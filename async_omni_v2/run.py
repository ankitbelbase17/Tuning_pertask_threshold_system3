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
from input_ingester import input_ingester_thread
from writer import writer_thread
from controller import controller_thread
from util import log, Profiler, EncoderControl, VideoClock, seed_everything
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

    seed_everything(cfg.seed, cfg.deterministic)   # before any model/RNG use

    # The primary backend always serves the orchestrator (language). It only also
    # needs the vision half if the encoder shares it (no dedicated encoder GPU).
    has_encoder_replica = bool(cfg.encoder_device and cfg.encoder_device != cfg.device)
    primary_role = "language" if has_encoder_replica else "full"
    backend = Qwen3VLBackend(cfg, role=primary_role)
    prof = Profiler(enabled=cfg.profile)
    mgr = KVCacheManager(backend, kv_budget=cfg.kv_budget, prof=prof)

    # Optional writer replica on a 2nd GPU (language-only); else share the primary.
    # The MVCC snapshot is shipped to its GPU inside the writer thread.
    writer_backend = backend
    if cfg.writer_device and cfg.writer_device != cfg.device:
        log("main", 0.0, f"loading writer replica (language-only) on {cfg.writer_device}")
        writer_backend = Qwen3VLBackend(dataclasses.replace(cfg, device=cfg.writer_device), role="language")
    else:
        log("main", 0.0, f"writer shares the model on {cfg.device}")

    # Optional encoder replica on a 3rd GPU (vision-only, ~1.2 GB); else share the
    # primary. Projected tokens are moved to the orchestrator's GPU in forward().
    encoder_backend = backend
    if has_encoder_replica:
        log("main", 0.0, f"loading encoder replica (vision-only) on {cfg.encoder_device}")
        encoder_backend = Qwen3VLBackend(dataclasses.replace(cfg, device=cfg.encoder_device), role="vision")
    else:
        log("main", 0.0, f"encoder shares the model on {cfg.device}")

    vis_q = queue.Queue(maxsize=cfg.frame_q_size)   # encoder -> ingester
    writer_q = queue.Queue(maxsize=1)
    stop = threading.Event()
    ctrl = EncoderControl(cfg.fps, cfg.encoder_idle_fps, cfg.encoder_focus_fps)
    evaluator = make_evaluator(cfg.video_path, cfg.gt_window, cfg.groundtruth) #NOTE: just to check for the france vs senegal match. not important
    if evaluator is not None:
        log("main", 0.0, "ground-truth eval ON (France 3-1 Senegal highlight)")

    # probe_scheduler="model": ingester is pure prefill (publishes vt via clock) and
    # the controller replaces the writer as the agentic orchestrator. "fixed"=current.
    model_mode = cfg.probe_scheduler == "model"
    clock = VideoClock() if model_mode else None
    if model_mode:
        driver = threading.Thread(target=controller_thread,
                                  args=(cfg, mgr, ctrl, clock, stop, prof, evaluator, writer_backend),
                                  name="controller", daemon=True)
    else:
        driver = threading.Thread(target=writer_thread,
                                  args=(cfg, mgr, writer_q, stop, prof, evaluator, writer_backend),
                                  name="writer", daemon=True)
    threads = [
        threading.Thread(target=encoder_thread, args=(cfg, encoder_backend, vis_q, ctrl, stop, prof),
                         name="encoder", daemon=True),
        threading.Thread(target=input_ingester_thread,
                         args=(cfg, mgr, vis_q, writer_q, ctrl, stop, prof, evaluator, clock),
                         name="input_ingester", daemon=True),
        driver,
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
