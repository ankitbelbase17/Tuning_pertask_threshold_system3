"""
evaluate.py — minimal OmniPro ONLINE eval for the icl_ingester_writer pipeline.

Runs the real async pipeline (encoder -> ingester -> controller, shared KV cache)
on each sample, captures its emissions, and scores them (greedy temporal match
within ±tolerance, plus content via the Gemini judge / exact-match).

All model behaviour + prompts come from async_omni_v2/config.py. The eval injects
only per-video DATA (task instruction + video path). Resumable per-sample by id.

  python evaluate.py --tasks event_narration --limit 1 --shortest --out ./output
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

from utils import OUTPUT_DIR, BENCHMARK_JSON, DATASET_DIR, log, read_jsonl, set_seed, write_json
from dataset import ALL_TASKS, load_samples


def _done_ids(path: str) -> set:
    return {r["id"] for fp in glob.glob(path) for r in read_jsonl(fp)}


def _append_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--audio", default="none_helpful",
                    choices=["none", "helpful", "required", "none_helpful", "all"])
    ap.add_argument("--limit", type=int, default=0, help="samples per task; 0=all")
    ap.add_argument("--shortest", action="store_true",
                    help="with --limit, keep the SHORTEST videos per task (fast subset)")
    ap.add_argument("--benchmark_json", default=None)
    ap.add_argument("--dataset_dir", default=None)
    ap.add_argument("--max_seconds", type=float, default=0, help="0=full video")
    ap.add_argument("--tolerance", type=float, default=3.0, help="temporal-match window (s)")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--out", default=OUTPUT_DIR)
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(1234)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    samples = load_samples(tasks=tasks, audio=args.audio,
                           limit_per_task=(args.limit or None), max_duration=None,
                           shortest=args.shortest,
                           benchmark_json=(args.benchmark_json or BENCHMARK_JSON),
                           dataset_dir=(args.dataset_dir or DATASET_DIR))
    if not samples:
        log("no samples; aborting.")
        return

    from system5_adapter import System5Runner
    from metrics import ContentJudge, aggregate, score_sample

    judge = ContentJudge()
    runner = System5Runner()      # all model config from async_omni_v2/config.py
    max_seconds = None if (args.max_seconds or 0) <= 0 else args.max_seconds

    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "online_pred.jsonl")
    done = _done_ids(pred_path) if args.resume else set()
    log(f"{len(done)} done, {len(samples)} samples", tag="online")

    for i, s in enumerate(samples):
        if s.id in done:
            continue
        t0 = time.time()
        pred = runner.run_sample(s, max_seconds=max_seconds)
        pred["wall_s"] = round(time.time() - t0, 2)
        _append_jsonl(pred_path, [pred])
        sc = score_sample(pred, tolerance=args.tolerance, judge=judge)
        log(f"[online] {i+1}/{len(samples)} {s.task} emits={sc['n_emits']} "
            f"gt={sc['n_gt']} tp={sc['tp_time']} fp={sc['fp']} fn={sc['fn']} "
            f"({pred['wall_s']}s)")

    rows = read_jsonl(pred_path)
    agg = aggregate([score_sample(r, tolerance=args.tolerance, judge=judge) for r in rows])
    agg["n"] = len(rows)
    write_json(os.path.join(args.out, "online_metrics.json"), agg)
    log(f"time_f1={agg['overall']['time_f1']} joint_f1={agg['overall']['joint_f1']} "
        f"content_acc={agg['overall']['content_acc']} (n={len(rows)})")


if __name__ == "__main__":
    main()
