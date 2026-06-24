"""
run.py — entrypoint: build config, load backend, wire the three threads.

Pipeline (all sharing ONE Qwen3-VL model + ONE linear KV cache via the manager):
    vision_stream  --frames-->  orchestrator  --trigger-->  writer
                                     |                          |
                                     +------ shared KV cache ----+
Each runs in its own thread at its own pace; the manager serializes GPU/cache
access by priority so the writer never blocks the orchestrator.
"""
import argparse
import dataclasses
import queue
import threading

from config import AsyncOmniConfig
from backend import Qwen3VLBackend
from manager import KVCacheManager
from vision_stream import encoder_thread
from orchestrator import orchestrator_thread
from writer import writer_thread
from util import log, Profiler, EncoderControl
from eval_gt import make_evaluator


def parse_args():
    cfg = AsyncOmniConfig()
    ap = argparse.ArgumentParser(description="async-omni v2 (Qwen3-VL)")
    # expose every dataclass field as --flag with the dataclass default
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

    backend = Qwen3VLBackend(cfg)
    prof = Profiler(enabled=cfg.profile)
    mgr = KVCacheManager(backend, kv_budget=cfg.kv_budget, prof=prof)

    vis_q = queue.Queue(maxsize=cfg.frame_q_size)   # encoder -> orchestrator
    writer_q = queue.Queue(maxsize=1)
    stop = threading.Event()
    ctrl = EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
    evaluator = make_evaluator(cfg.video_path, cfg.gt_window, cfg.groundtruth) #NOTE: just to check for the france vs senegal match. not important
    if evaluator is not None:
        log("main", 0.0, "ground-truth eval ON (France 3-1 Senegal highlight)")

    threads = [
        threading.Thread(target=encoder_thread, args=(cfg, backend, vis_q, ctrl, stop, prof),
                         name="encoder", daemon=True),
        threading.Thread(target=orchestrator_thread,
                         args=(cfg, mgr, vis_q, writer_q, ctrl, stop, prof, evaluator),
                         name="orchestrator", daemon=True),
        threading.Thread(target=writer_thread, args=(cfg, mgr, writer_q, stop, prof, evaluator),
                         name="writer", daemon=True),
    ]
    log("main", 0.0, f"start: model={cfg.model_id} fps={cfg.fps} "
                     f"max={cfg.max_seconds}s budget={cfg.kv_budget} "
                     f"realtime={cfg.realtime} speed={cfg.speed}")
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(prof.summary(), flush=True)
    if evaluator is not None:
        print(evaluator.report(), flush=True)
    log("main", 0.0, "done")


if __name__ == "__main__":
    main()
