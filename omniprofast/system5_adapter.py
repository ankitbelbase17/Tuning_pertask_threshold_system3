"""
system5_adapter.py — run the REAL system_5 (async-omni v2, icl_ingester_writer)
pipeline per sample.

FIDELITY CONTRACT
-----------------
This driver does NOT reimplement system_5. It spins up the exact three-thread
pipeline from run.py — encoder_thread -> input_ingester_thread -> controller_thread,
sharing one linear KV cache through KVCacheManager — and lets it run unmodified.
The controller self-paces in VIDEO time, so it MUST run realtime (batch would
fast-forward the whole clip before the controller could walk it).

The ONLY things set per-sample are: system_prompt (templated from the sample),
video_path, and max_seconds. Everything else keeps async_omni_v2's config defaults.

Emissions are captured via the evaluator hook the controller already invokes
(record_trigger / record_write) — no code in system_5 is touched.
"""
from __future__ import annotations

import dataclasses
import queue
import sys
import threading

from utils import SYSTEM5_DIR, log

if SYSTEM5_DIR not in sys.path:
    sys.path.insert(0, SYSTEM5_DIR)


class CaptureEvaluator:
    """Collects (vt, share) triggers and (vt, text, latency) writes for any
    video. Thread-safe (the controller calls it from its own thread)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.triggers: list[tuple[float, float]] = []
        self.writes: list[tuple[float, str, float]] = []
        self.gates: list[tuple[float, float, float | None]] = []

    def record_gate(self, vt, share, thr=None):
        with self._lock:
            self.gates.append((float(vt), float(share), thr))

    def record_trigger(self, vt, share):
        with self._lock:
            self.triggers.append((float(vt), float(share)))

    def record_write(self, vt, text, wall_latency):
        with self._lock:
            self.writes.append((float(vt), str(text), float(wall_latency)))

    def emissions(self) -> list[dict]:
        """Merge triggers with their writer text (matched by trigger vt)."""
        with self._lock:
            writes_by_vt = {}
            for vt, text, lat in self.writes:
                writes_by_vt.setdefault(vt, (text, lat))
            out = []
            for vt, share in sorted(self.triggers):
                text, lat = writes_by_vt.get(vt, ("", None))
                out.append({"t_sec": vt, "share": share, "raw": text,
                            "writer_latency_s": lat})
            return out


class System5Runner:
    def __init__(self):
        from config import AsyncOmniConfig
        from backend import Qwen3VLBackend
        from manager import KVCacheManager
        from vision_stream import encoder_thread
        from input_ingester import input_ingester_thread
        from controller import controller_thread
        from util import EncoderControl, VideoClock

        self._KVCacheManager = KVCacheManager
        self._encoder_thread = encoder_thread
        self._ingester_thread = input_ingester_thread
        self._controller_thread = controller_thread
        self._EncoderControl = EncoderControl
        self._VideoClock = VideoClock

        # Everything (model_id, dtype, device(s), kv_budget, fps, prompts, sampling,
        # ...) comes from config.py — the single source of truth. The eval injects
        # only per-video data (instruction, video_path) in run_sample.
        self.base_cfg = AsyncOmniConfig()
        cfg = self.base_cfg

        import dataclasses as _dc
        import torch
        self.torch = torch

        # Optional multi-GPU placement (config.writer_device / encoder_device). When
        # set, the controller's generation and/or the encoder's vision run on their
        # OWN GPU so decode doesn't time-share with frame encode/ingest.
        has_enc = bool(cfg.encoder_device and cfg.encoder_device != cfg.device)
        has_ctrl = bool(cfg.writer_device and cfg.writer_device != cfg.device)
        primary_role = "language" if has_enc else "full"   # primary needs vision only if it also encodes
        log(f"loading backends: primary role={primary_role} on {cfg.device}"
            f"{' | controller on '+cfg.writer_device if has_ctrl else ''}"
            f"{' | encoder on '+cfg.encoder_device if has_enc else ''}", tag="runner")

        self.backend = Qwen3VLBackend(cfg, role=primary_role)          # ingester + shared cache
        self.encoder_backend = self.backend
        if has_enc:
            self.encoder_backend = Qwen3VLBackend(_dc.replace(cfg, device=cfg.encoder_device), role="vision")
        self.controller_backend = self.backend
        if has_ctrl:
            self.controller_backend = Qwen3VLBackend(_dc.replace(cfg, device=cfg.writer_device), role="language")
        log("backends loaded; icl_ingester_writer pipeline (3 threads, shared cache).",
            tag="runner")

    def run_sample(self, sample, *, max_seconds: float | None = None) -> dict:
        """Run the real async pipeline on one video; return captured emissions.

        The eval injects ONLY per-video DATA — the task `instruction` and the
        `video_path` (plus how much of the clip to run). EVERYTHING behavioural
        (prompts, fps, realtime, sampling, kv_budget, timestamps) comes from
        async_omni_v2/config.py, which is the single source of truth."""
        # pick the task-specific controller ICL prompt if we have one for this task,
        # else fall back to the generic controller_prompt (both live in config.py)
        controller_prompt = self.base_cfg.task_controller_prompts.get(
            sample.task, self.base_cfg.controller_prompt)
        cfg = dataclasses.replace(
            self.base_cfg,
            instruction=sample.question,
            video_path=sample.video_path,
            video_id=sample.video_id,
            max_seconds=(max_seconds if max_seconds else 10 ** 9),
            controller_prompt=controller_prompt,
        )

        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=None)
        in_q = queue.Queue(maxsize=max(256, cfg.frame_q_size))
        stop = threading.Event()
        ctrl = self._EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
        clock = self._VideoClock()
        ev = CaptureEvaluator()

        threads = [
            threading.Thread(target=self._encoder_thread,
                             args=(cfg, self.encoder_backend, in_q, ctrl, stop, None),
                             name="encoder", daemon=True),
            threading.Thread(target=self._ingester_thread,
                             args=(cfg, mgr, in_q, ctrl, stop, None, clock),
                             name="input_ingester", daemon=True),
            threading.Thread(target=self._controller_thread,
                             args=(cfg, mgr, ctrl, clock, stop, None, ev, self.controller_backend),
                             name="controller", daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        emits = ev.emissions()
        return {
            "id": sample.id, "task": sample.task, "video_id": sample.video_id,
            "question": sample.question, "event": sample.event,
            "audio_dependency": sample.audio_dependency,
            "ground_truth": sample.ground_truth,
            "predictions": emits,
            "n_triggers": len(ev.triggers), "n_writes": len(ev.writes),
            "n_gates": len(ev.gates),
            "eval_mode": "online", "realtime": cfg.realtime,
        }
