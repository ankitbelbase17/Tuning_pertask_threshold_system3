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
import os
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
        from backend import Qwen3VLBackend, Qwen2_5OmniBackend
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

        # FORK (system3_qwem_omni): honour cfg.backend the SAME way run.py:61 does,
        # instead of hardcoding the parent's vision-only class. Previously this
        # adapter always built Qwen3VLBackend while config.py's model_id had
        # already been switched to Qwen/Qwen2.5-Omni-7B, so the OmniPro eval was
        # feeding omni weights into Qwen3VLForConditionalGeneration -- it was not
        # evaluating the omni model at all. OMNIPRO_BACKEND overrides for A/B.
        BACKENDS = {"qwen2_5_omni": Qwen2_5OmniBackend, "qwen3_vl": Qwen3VLBackend}
        bname = os.environ.get("OMNIPRO_BACKEND", cfg.backend)
        if bname not in BACKENDS:
            raise SystemExit(f"OMNIPRO_BACKEND/cfg.backend must be one of "
                             f"{list(BACKENDS)}, got {bname!r}")
        BK = BACKENDS[bname]
        log(f"backend={bname} ({BK.__name__}) model_id={cfg.model_id} "
            f"use_audio={getattr(cfg,'use_audio',False)} "
            f"tmrope={getattr(cfg,'tmrope_positions',False)}", tag="runner")
        self.backend_name = bname

        self.backend = BK(cfg, role=primary_role)          # ingester + shared cache
        self.encoder_backend = self.backend
        if has_enc:
            self.encoder_backend = BK(_dc.replace(cfg, device=cfg.encoder_device), role="vision")
        self.controller_backend = self.backend
        if has_ctrl:
            self.controller_backend = BK(_dc.replace(cfg, device=cfg.writer_device), role="language")
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
        # A/B switch for the head-to-head (EXPERIENCE.md): OMNIPRO_GATE_MODE
        # overrides config.gate_mode so both systems run from one checkout.
        import os
        gate_mode = os.environ.get("OMNIPRO_GATE_MODE", self.base_cfg.gate_mode)
        # A/B switch for the schema-walk decoder: OMNIPRO_DECODE_MODE=schema|free
        # lets both decoders run from one checkout (same commit, same weights), so
        # the comparison is matched.
        decode_mode = os.environ.get("OMNIPRO_DECODE_MODE", self.base_cfg.decode_mode)
        # PER-TASK firing threshold. A single global 0.5 made two tasks score
        # time_f1 0.000 -- not a perception failure, the gate simply could never
        # fire. Fitted offline by auc.py; OMNIPRO_HIT_THRESHOLD forces one value
        # across all tasks (used to reproduce the old global-0.5 behaviour).
        hit_threshold = self.base_cfg.task_hit_thresholds.get(
            sample.task, self.base_cfg.hit_threshold)
        if os.environ.get("OMNIPRO_HIT_THRESHOLD"):
            hit_threshold = float(os.environ["OMNIPRO_HIT_THRESHOLD"])
        # A/B switch for the KV-cache read path: OMNIPRO_CACHE_MODE=snapshot|inplace.
        # "inplace" drops the per-tick deep copy of the cache. It must produce
        # BYTE-IDENTICAL output to "snapshot" -- see test_inplace.py, which asserts
        # exactly that -- so this is a memory/latency switch, never an accuracy one.
        cache_mode = os.environ.get("OMNIPRO_CACHE_MODE",
                                    self.base_cfg.controller_cache_mode)
        # A/B switch for the sampler: OMNIPRO_WRITER_GREEDY=1 forces argmax.
        # Seeded multinomial sampling is the leading suspect for the run-to-run
        # divergence that broke the ab_inplace null control (MISSION S6/S11): a
        # near-tie resolved differently by bf16 kernel jitter changes one token,
        # and the stream diverges from there. This knob is what tests that.
        _g = os.environ.get("OMNIPRO_WRITER_GREEDY")
        writer_greedy = (_g not in (None, "", "0", "false", "False")
                         if _g is not None else self.base_cfg.writer_greedy)
        cfg = dataclasses.replace(
            self.base_cfg,
            decode_mode=decode_mode,
            hit_threshold=hit_threshold,
            controller_cache_mode=cache_mode,
            writer_greedy=writer_greedy,
            # signal-quality sweep knobs (all change the forward pass, so none can
            # be screened offline the way gate strategies can)
            seen_mode=os.environ.get("OMNIPRO_SEEN_MODE", self.base_cfg.seen_mode),
            schema_max_seen_tokens=int(os.environ.get(
                "OMNIPRO_SEEN_TOKENS", self.base_cfg.schema_max_seen_tokens)),
            instruction=sample.question,
            event=(sample.event or sample.question),
            video_path=sample.video_path,
            video_id=sample.video_id,
            max_seconds=(max_seconds if max_seconds else 10 ** 9),
            controller_prompt=controller_prompt,
            writer_prompt=self.base_cfg.task_writer_prompts.get(
                sample.task, self.base_cfg.writer_prompt),
            gate_mode=gate_mode,
            # deterministic => frame-indexed lockstep walk: no wall-clock pacing
            # (batch), blocking queues, ingester waits on each due tick. This is
            # what makes runs bit-reproducible (the async snapshot race is gone).
            realtime=(self.base_cfg.realtime and not self.base_cfg.deterministic),
        )

        # re-seed per sample so every video starts from the same RNG state and
        # (deterministic=True) CUDA kernels are deterministic
        from util import seed_everything
        seed_everything(cfg.seed, cfg.deterministic)

        # TELEMETRY: a Profiler collects per-run counts + timings (encoder frames
        # emitted/dropped, ingester frames ingested, per-op GPU times, controller
        # gen time + tokens). Passed to the manager + all three threads; summary
        # printed below into the run log.
        from util import Profiler
        prof = Profiler(enabled=True)
        mgr = self._KVCacheManager(self.backend, kv_budget=cfg.kv_budget, prof=prof)
        in_q = queue.Queue(maxsize=max(256, cfg.frame_q_size))
        stop = threading.Event()
        feed_done = threading.Event()   # ingester -> controller: stream fully drained
        ctrl = self._EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
        clock = self._VideoClock()
        ev = CaptureEvaluator()

        if cfg.gate_mode == "probe":
            # PROBE-GATE head-to-head arm: yes/no logit gate inside the ingester
            # fires a separate writer. No controller/clock; in deterministic mode
            # the ingester blocks on writer_q.join() -> frame-indexed, reproducible.
            from writer import writer_thread
            writer_q = queue.Queue(maxsize=4)
            threads = [
                threading.Thread(target=self._encoder_thread,
                                 args=(cfg, self.encoder_backend, in_q, ctrl, stop, prof),
                                 name="encoder", daemon=True),
                threading.Thread(target=self._ingester_thread,
                                 args=(cfg, mgr, in_q, ctrl, stop, prof, None, feed_done,
                                       writer_q, ev),
                                 name="input_ingester", daemon=True),
                threading.Thread(target=writer_thread,
                                 args=(cfg, mgr, writer_q, stop, prof, ev,
                                       self.controller_backend, feed_done),
                                 name="writer", daemon=True),
            ]
        else:
            threads = [
                threading.Thread(target=self._encoder_thread,
                                 args=(cfg, self.encoder_backend, in_q, ctrl, stop, prof),
                                 name="encoder", daemon=True),
                threading.Thread(target=self._ingester_thread,
                                 args=(cfg, mgr, in_q, ctrl, stop, prof, clock, feed_done),
                                 name="input_ingester", daemon=True),
                threading.Thread(target=self._controller_thread,
                                 args=(cfg, mgr, ctrl, clock, stop, prof, ev,
                                       self.controller_backend, feed_done),
                                 name="controller", daemon=True),
            ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

        # dump the telemetry for THIS video into the run log
        log(f"TELEMETRY {sample.video_id}\n{prof.summary()}", tag="prof")

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
