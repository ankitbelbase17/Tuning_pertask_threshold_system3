"""
run.py — entrypoint: build config, load backend, wire the three threads.

FORK (system3_qwem_omni): backbone is Qwen2.5-Omni-7B (vision + audio) by
default, selected via cfg.backend -- see OMNI_EXTENSION.md for why this model,
OMNI_FEASIBILITY.md for the audio-path verification, backend.py's
Qwen2_5OmniBackend docstring for the engineering detail.

Pipeline (all sharing ONE model + ONE linear KV cache via the manager):
    vision_stream  --frames-->  input_ingester  --shared KV cache-->  controller
The encoder streams + encodes frames; the ingester prefills them into the shared
cache (AND, every audio_seconds_per_chunk, synchronously embeds + ingests the
matching audio window -- see input_ingester.py's "OPTION A" docstring); the
controller reads that cache each tick via an MVCC snapshot and emits a control
JSON (fps steer + answer). Each runs in its own thread at its own pace.
"""
import argparse
import dataclasses
import queue
import threading

from config import AsyncOmniConfig
from backend import Qwen3VLBackend, Qwen2_5OmniBackend
from manager import KVCacheManager
from vision_stream import encoder_thread
from input_ingester import input_ingester_thread
from controller import controller_thread
from util import log, Profiler, EncoderControl, VideoClock, seed_everything


def parse_args():
    cfg = AsyncOmniConfig()
    ap = argparse.ArgumentParser(description="async-omni v2 (Qwen3-VL) — icl_ingester_writer")
    for f in dataclasses.fields(cfg):
        if f.type in (list,):
            continue
        if f.type is bool:
            ap.add_argument(f"--{f.name}", action="store_true", default=getattr(cfg, f.name))
            ap.add_argument(f"--no_{f.name}", dest=f.name, action="store_false")
        else:
            ap.add_argument(f"--{f.name}", type=type(getattr(cfg, f.name)),
                            default=getattr(cfg, f.name))
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = AsyncOmniConfig(**{k: v for k, v in vars(args).items()
                             if k in {f.name for f in dataclasses.fields(AsyncOmniConfig)}})
    if not cfg.video_path:
        raise SystemExit("--video_path is required")

    seed_everything(cfg.seed, cfg.deterministic)   # before any model/RNG use

    # FORK (system3_qwem_omni): cfg.backend selects the frozen backbone.
    # "qwen3_vl" is kept for an explicit A/B against the parent checkout
    # (with cfg.use_audio=False, cfg.tmrope_positions=False to match it
    # exactly) -- run.py:48 remains the single construction site the
    # OMNI_FEASIBILITY.md engineering plan calls for.
    BACKENDS = {"qwen2_5_omni": Qwen2_5OmniBackend, "qwen3_vl": Qwen3VLBackend}
    if cfg.backend not in BACKENDS:
        raise SystemExit(f"--backend must be one of {list(BACKENDS)}, got {cfg.backend!r}")
    backend = BACKENDS[cfg.backend](cfg)
    prof = Profiler(enabled=cfg.profile)
    mgr = KVCacheManager(backend, kv_budget=cfg.kv_budget, prof=prof)

    vis_q = queue.Queue(maxsize=cfg.frame_q_size)   # encoder -> ingester
    stop = threading.Event()
    feed_done = threading.Event()                   # ingester -> controller: stream drained
    ctrl = EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
    clock = VideoClock()                            # ingester publishes vt; controller reads it

    threads = [
        threading.Thread(target=encoder_thread, args=(cfg, backend, vis_q, ctrl, stop, prof),
                         name="encoder", daemon=True),
        threading.Thread(target=input_ingester_thread,
                         args=(cfg, mgr, vis_q, ctrl, stop, prof, clock, feed_done),
                         name="input_ingester", daemon=True),
        threading.Thread(target=controller_thread,
                         args=(cfg, mgr, ctrl, clock, stop, prof, None, None, feed_done),
                         name="controller", daemon=True),
    ]
    log("main", 0.0, f"start: model={cfg.model_id} fps={cfg.fps} "
                     f"max={cfg.max_seconds}s budget={cfg.kv_budget} "
                     f"realtime={cfg.realtime} speed={cfg.speed}")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(prof.summary(), flush=True)
    log("main", 0.0, "done")


if __name__ == "__main__":
    main()
