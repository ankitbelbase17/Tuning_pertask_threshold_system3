"""
system5_adapter.py — run the REAL system_5 (async-omni v2) pipeline per sample.

FIDELITY CONTRACT
-----------------
This driver does NOT reimplement system_5. It spins up the *exact* three-thread
pipeline from run.py — encoder_thread -> orchestrator_thread -> writer_thread,
sharing one linear KV cache through KVCacheManager — and lets it run unmodified.
The async behaviour is therefore identical to the original:
  * the vision INPUT gate still adapts encoder fps (focus vs idle);
  * eviction, the single-writer invariant, and the priority/lock model are intact;
  * the writer still works off an MVCC snapshot_clone in its own thread.

The ONLY things changed from the previous implementation are the ones requested:
  * the THREE prompts (system_prompt, goal_question, writer_prompt) are now
    generic, templated per-sample instead of football-hard-coded;
  * goal_threshold is varied for the two precision/recall sweep variants only;
  * video_path / max_seconds are set per sample;
  * realtime defaults to False (system_5's own supported batch mode, `run.sh
    batch`) so 60 runs don't sleep in wall-clock — the concurrency model is
    unchanged, only the encoder's wall-clock pacing sleep is skipped.
Everything else (kv_budget, gate cadences, vision_question, writer sampling
preset, fps bounds) keeps system_5's dataclass defaults.

Emissions are captured via the evaluator hook the threads already invoke
(record_trigger / record_write) — no code in system_5 is touched.
"""
from __future__ import annotations

import dataclasses
import os
import queue
import sys
import threading

from utils import SYSTEM5_DIR, log

if SYSTEM5_DIR not in sys.path:
    sys.path.insert(0, SYSTEM5_DIR)


def _env_overrides() -> dict:
    """Benchmark-matrix switches, toggled per run without touching config.py.
    Each maps an OMNIPRO_* env var (1/0/true/false) onto the matching
    AsyncOmniConfig field; unset vars fall through to the dataclass default."""
    def flag(name):
        v = os.environ.get(name)
        return None if v is None else v.strip().lower() in ("1", "true", "yes", "on")
    over = {
        "input_gate": flag("OMNIPRO_INPUT_GATE"),
        "output_gate": flag("OMNIPRO_OUTPUT_GATE"),
        "writer_cache": flag("OMNIPRO_WRITER_CACHE"),
        "timestamp_tokens": flag("OMNIPRO_TIMESTAMP_TOKENS"),
        "deterministic": flag("OMNIPRO_DETERMINISTIC"),
        "gate_hysteresis": flag("OMNIPRO_GATE_HYSTERESIS"),
    }
    return {k: v for k, v in over.items() if v is not None}


class CaptureEvaluator:
    """Drop-in for system_5's GroundTruthEvaluator: same record_* interface, but
    it just collects (vt, share) triggers and (vt, text, latency) writes for any
    video. Thread-safe (orchestrator and writer call it from different threads)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.triggers: list[tuple[float, float]] = []
        self.writes: list[tuple[float, str, float]] = []
        self.gates: list[tuple[float, float, float | None]] = []

    def record_gate(self, vt, share, thr=None):
        # Called by orchestrator.py on EVERY gate tick (before threshold test).
        # We only score triggers/writes, but the hook must exist or the pipeline
        # crashes with AttributeError.
        with self._lock:
            self.gates.append((float(vt), float(share), thr))

    def record_trigger(self, vt, share):
        with self._lock:
            self.triggers.append((float(vt), float(share)))

    def record_write(self, vt, text, wall_latency):
        with self._lock:
            self.writes.append((float(vt), str(text), float(wall_latency)))

    def emissions(self) -> list[dict]:
        """Merge triggers with their writer text (matched by trigger vt).

        Time metrics use trigger times (every fired gate is an emission, even if
        the bounded writer_q dropped its write under load). Content uses the
        writer line emitted for that trigger when present.
        """
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
    def __init__(self, *, model_id: str | None = None, dtype: str = "bfloat16",
                 device: str = "cuda", kv_budget: int | None = None):
        # This repo's system_5 package is `async_omni_v2`. Module/name seams vs the
        # original prosync layout: orchestrator -> input_ingester, build_backend ->
        # Qwen3VLBackend, no `mode`/`event_*`/`prompt_mode`/`gate_mode` fields.
        from config import AsyncOmniConfig
        from backend import Qwen3VLBackend
        from manager import KVCacheManager
        from vision_stream import encoder_thread
        from input_ingester import input_ingester_thread
        from writer import writer_thread
        from util import EncoderControl

        self._AsyncOmniConfig = AsyncOmniConfig
        self._KVCacheManager = KVCacheManager
        self._encoder_thread = encoder_thread
        self._ingester_thread = input_ingester_thread
        self._writer_thread = writer_thread
        self._EncoderControl = EncoderControl

        base_kwargs = dict(dtype=dtype, device=device, profile=False)
        if model_id:
            base_kwargs["model_id"] = model_id
        if kv_budget:
            base_kwargs["kv_budget"] = kv_budget
        # system_5 dataclass defaults for everything else
        self.base_cfg = AsyncOmniConfig(**base_kwargs)

        import torch
        self.torch = torch
        log(f"loading backend {self.base_cfg.model_id} ({dtype}); "
            f"kv_budget={self.base_cfg.kv_budget}", tag="runner")
        self.backend = Qwen3VLBackend(self.base_cfg)   # role='full' (single shared backend)
        log("backend loaded; async pipeline preserved (3 threads, shared cache).",
            tag="runner")

    def run_sample(self, sample, prompt_fields: dict, *, max_seconds: float | None,
                   realtime: bool = False, fps: float | None = None) -> dict:
        """Run the real async pipeline on one video; return captured emissions."""
        # system_5 uses goal_question/goal_threshold (not event_*), always uses its
        # prompts verbatim + a fixed threshold, and has no `mode`/prompt_mode/gate_mode.
        # Ablation-matrix switches are overridable per run via OMNIPRO_* env vars.
        cfg = dataclasses.replace(
            self.base_cfg,
            system_prompt=prompt_fields["system_prompt"],
            goal_question=prompt_fields["goal_question"],
            writer_prompt=prompt_fields["writer_prompt"],
            goal_threshold=prompt_fields["goal_threshold"],
            video_path=sample.video_path,
            max_seconds=(max_seconds if max_seconds else 10 ** 9),
            realtime=realtime,
            groundtruth=False,                # disable the football-only GT hook
            **({"fps": fps} if fps else {}),
            **_env_overrides(),               # OMNIPRO_INPUT_GATE / OUTPUT_GATE / ...
        )

        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=None)
        in_q = queue.Queue(maxsize=max(256, cfg.frame_q_size))
        writer_q = queue.Queue(maxsize=4)
        stop = threading.Event()   # system_5 coordinates all 3 threads on ONE event
        ctrl = self._EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
        ev = CaptureEvaluator()

        threads = [
            threading.Thread(target=self._encoder_thread,
                             args=(cfg, self.backend, in_q, ctrl, stop, None),
                             name="encoder", daemon=True),
            threading.Thread(target=self._ingester_thread,
                             args=(cfg, mgr, in_q, writer_q, ctrl, stop, None, ev),
                             name="input_ingester", daemon=True),
            threading.Thread(target=self._writer_thread,
                             args=(cfg, mgr, writer_q, stop, None, ev, self.backend),
                             name="writer", daemon=True),
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
            "eval_mode": "online", "realtime": realtime,
        }
