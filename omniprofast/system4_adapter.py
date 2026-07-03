"""
system4_adapter.py — run the REAL system4 (async-omni v4, Qwen2.5-Omni) pipeline.

FIDELITY CONTRACT (same as the system_5 runner): this does NOT reimplement
system4. It replicates run.py:run_once's thread wiring — audio_thread +
encoder_thread (+ clock_thread) -> orchestrator_thread -> writer_thread, all
sharing one linear KV cache via KVCacheManager — and runs it unmodified. The
async behaviour (audio+vision merged queue, time-cadence event gate, vision gate
steering fps, MVCC writer) is identical to the original.

Only the requested things change: the three prompts are genericised per sample
(system_prompt / event_question / writer_prompt) and event_threshold is the
variant's threshold; gate_mode is pinned to "fixed" so that threshold is honoured;
video_path / max_seconds are per sample; realtime=False (batch). system4 IS
audio-capable, so audio_on=True (its whole point) — OmniPro audio-dependent
samples are fair game here, unlike vision-only system_5.

Emissions captured via the evaluator hook system4 already calls
(record_gate / record_trigger / record_write).
"""
from __future__ import annotations

import dataclasses
import queue
import sys
import threading

from utils import SYSTEM4_DIR, log

if SYSTEM4_DIR not in sys.path:
    sys.path.insert(0, SYSTEM4_DIR)


class CaptureEvaluator:
    """Matches system4's OnlineEvaluator record interface, for any video."""

    def __init__(self):
        self._lock = threading.Lock()
        self.triggers, self.writes, self.gates = [], [], []

    def record_gate(self, vt, share, thr=None):
        with self._lock:
            self.gates.append((float(vt), float(share), thr))

    def record_trigger(self, vt, share):
        with self._lock:
            self.triggers.append((float(vt), float(share)))

    def record_write(self, vt, text, wall_latency):
        with self._lock:
            self.writes.append((float(vt), str(text), float(wall_latency)))

    def emissions(self):
        with self._lock:
            wbv = {}
            for vt, text, lat in self.writes:
                wbv.setdefault(vt, (text, lat))
            out = []
            for vt, share in sorted(self.triggers):
                text, lat = wbv.get(vt, ("", None))
                out.append({"t_sec": vt, "share": share, "raw": text,
                            "writer_latency_s": lat})
            return out


class System4Runner:
    def __init__(self, *, model_id: str | None = None, dtype: str = "bfloat16",
                 device: str = "cuda", kv_budget: int | None = None):
        from config import AsyncOmniConfig
        from backend import build_backend
        from manager import KVCacheManager
        from vision_stream import encoder_thread
        from audio_stream import audio_thread
        from clock_stream import clock_thread
        from orchestrator import orchestrator_thread
        from writer import writer_thread
        from util import EncoderControl

        self._cfg_cls = AsyncOmniConfig
        self._KVCacheManager = KVCacheManager
        self._encoder_thread = encoder_thread
        self._audio_thread = audio_thread
        self._clock_thread = clock_thread
        self._orchestrator_thread = orchestrator_thread
        self._writer_thread = writer_thread
        self._EncoderControl = EncoderControl

        base_kwargs = dict(mode="video_audio", dtype=dtype, device=device, profile=False)
        if model_id:
            base_kwargs["model_id"] = model_id
        if kv_budget:
            base_kwargs["kv_budget"] = kv_budget
        self.base_cfg = AsyncOmniConfig(**base_kwargs)

        import torch
        self.torch = torch
        log(f"[s4] loading backend {self.base_cfg.model_id} ({dtype}); "
            f"audio_on={self.base_cfg.audio_on}", tag="s4")
        self.backend = build_backend(self.base_cfg)
        log(f"[s4] backend loaded (accepts_audio={self.backend.accepts_audio}).", tag="s4")

    def run_sample(self, sample, prompt_fields: dict, *, max_seconds: float | None,
                   realtime: bool = False, fps: float | None = None) -> dict:
        cfg = dataclasses.replace(
            self.base_cfg,
            system_prompt=prompt_fields["system_prompt"],
            event_question=prompt_fields["goal_question"],     # map goal_->event_
            writer_prompt=prompt_fields["writer_prompt"],
            event_threshold=prompt_fields["goal_threshold"],
            prompt_mode="task",          # use the injected prompts verbatim
            gate_mode="fixed",           # honour the variant threshold
            video_path=sample.video_path,
            max_seconds=(max_seconds if max_seconds else 10 ** 9),
            realtime=realtime,
            **({"fps": fps} if fps else {}),
        )

        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=None)
        in_q = queue.Queue(maxsize=max(256, cfg.frame_q_size + cfg.audio_q_size))
        writer_q = queue.Queue(maxsize=4)
        stop = threading.Event()
        done = threading.Event()
        ctrl = self._EncoderControl(cfg.fps, 0.0, cfg.encoder_focus_fps)
        ev = CaptureEvaluator()

        threads = []
        if cfg.audio_on and self.backend.accepts_audio:
            threads.append(threading.Thread(target=self._audio_thread,
                           args=(cfg, self.backend, in_q, stop, None),
                           name="audio", daemon=True))
        if not cfg.vision_off:
            threads.append(threading.Thread(target=self._encoder_thread,
                           args=(cfg, self.backend, in_q, ctrl, stop, None),
                           name="encoder", daemon=True))
        if cfg.timestamps:
            threads.append(threading.Thread(target=self._clock_thread,
                           args=(cfg, self.backend, in_q, stop, None),
                           name="clock", daemon=True))
        threads.append(threading.Thread(target=self._orchestrator_thread,
                       args=(cfg, mgr, in_q, writer_q, ctrl, stop, done, None, ev),
                       name="orchestrator", daemon=True))
        threads.append(threading.Thread(target=self._writer_thread,
                       args=(cfg, mgr, writer_q, done, None, ev, self.backend),
                       name="writer", daemon=True))
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
            "n_gates": len(ev.gates), "eval_mode": "online", "realtime": realtime,
            "system": "system4",
        }
