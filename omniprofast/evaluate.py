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
    # WINDOW-FIT FILTER (omni_s3_eval/chain_state.py). A debug-qos generation at
    # n nodes gets a (90/n)-minute wall, and resume is per-SAMPLE: a video whose
    # eval cannot finish inside the window is started, SIGKILLed, and retried
    # forever. The chain hands each shape only the samples that fit it, so a
    # 4-node/22-min generation never picks up a 595 s video. 0 = no cap.
    ap.add_argument("--max_dur", type=float, default=0,
                    help="skip samples longer than this many seconds (0=no cap)")
    ap.add_argument("--tolerance", type=float, default=3.0, help="temporal-match window (s)")
    ap.add_argument("--shard", type=int, default=0, help="this lane's index")
    ap.add_argument("--nshards", type=int, default=1, help="total lanes")
    ap.add_argument("--subset_every", type=int, default=1,
                    help="keep 1 sample in N (deterministic by sorted id); 4 = 25%")
    ap.add_argument("--subset_off", type=int, default=0, help="offset within --subset_every")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--done_glob", default=None,
                    help="glob of every lane's online_pred.jsonl; resume skips ids "
                         "finished by ANY lane (needed when the lane count changes)")
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(1234)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    samples = load_samples(tasks=tasks, audio=args.audio,
                           limit_per_task=(args.limit or None),
                           max_duration=(args.max_dur or None),
                           shortest=args.shortest,
                           benchmark_json=(args.benchmark_json or BENCHMARK_JSON),
                           dataset_dir=(args.dataset_dir or DATASET_DIR))
    if not samples:
        log("no samples; aborting.")
        return

    # LANE SHARDING. Sort by id first so the split is identical on every lane and
    # every restart regardless of load_samples' internal ordering, then take a
    # round-robin stripe. Round-robin (not contiguous blocks) keeps long and short
    # videos mixed evenly across lanes, so no single lane inherits the whole tail.
    samples.sort(key=lambda s: s.id)
    if args.subset_every > 1:
        samples = [s for i, s in enumerate(samples)
                   if i % args.subset_every == args.subset_off]
        log(f"subset 1-in-{args.subset_every} (off {args.subset_off}): "
            f"{len(samples)} samples", tag="online")
    if args.nshards > 1:
        samples = [s for i, s in enumerate(samples) if i % args.nshards == args.shard]
        log(f"shard {args.shard}/{args.nshards}: {len(samples)} samples "
            f"({sum(s.duration for s in samples)/3600:.2f} h of video)", tag="online")

    # SHORTEST-FIRST inside the lane. The debug QoS gives a fixed wall window and
    # resume is per-SAMPLE, so a long video that cannot finish inside one window
    # would be retried forever and block everything behind it. Running short ones
    # first banks the cheap samples in early generations and isolates the long
    # tail into the final ones, where run_chain.sh reshapes the job to a longer
    # window. Nothing is dropped -- only reordered.
    samples.sort(key=lambda s: s.duration)

    from system5_adapter import System5Runner
    from metrics import ContentJudge, aggregate, score_sample

    judge = ContentJudge()
    runner = System5Runner()      # all model config from async_omni_v2/config.py
    max_seconds = None if (args.max_seconds or 0) <= 0 else args.max_seconds

    os.makedirs(args.out, exist_ok=True)
    pred_path = os.path.join(args.out, "online_pred.jsonl")
    # --no_resume => fresh run: truncate stale predictions so the aggregate isn't
    # polluted by earlier runs (the file is APPENDED to per sample).
    if not args.resume and os.path.exists(pred_path):
        os.remove(pred_path)
    # RESUME IS GLOBAL ACROSS LANES, not per-lane. The chain reshapes 4 -> 2 -> 1
    # nodes to give the long tail a longer window, which changes the lane COUNT and
    # therefore the round-robin shard assignment: lane 0 of a 4-lane split covers
    # different samples than lane 0 of a 16-lane split. Resuming from this lane's
    # own file only would silently re-run samples another lane already finished.
    # Globbing every sibling lane makes the shard split a work-distribution hint
    # while completion stays a property of the run as a whole.
    done = _done_ids(args.done_glob or pred_path) if args.resume else set()
    log(f"{len(done)} done, {len(samples)} samples", tag="online")

    for i, s in enumerate(samples):
        if s.id in done:
            continue
        log(f"===== [{i+1}/{len(samples)}] {s.id} "
            f"gt_times={s.gt_times} question={s.question!r}", tag="online")
        t0 = time.time()
        pred = runner.run_sample(s, max_seconds=max_seconds)
        wall = time.time() - t0
        # TIMING BLOCK (per video): everything needed for the wall-clock-vs-length
        # table, recorded per sample so a resumed run keeps its numbers.
        eff_len = min(s.duration, max_seconds) if max_seconds else s.duration
        pred["wall_s"] = round(wall, 2)
        pred["video_len_s"] = round(s.duration, 2)
        pred["eval_len_s"] = round(eff_len, 2)
        # >1.0 = SLOWER than the video plays; <1.0 = faster than realtime.
        pred["realtime_factor"] = round(wall / eff_len, 4) if eff_len > 0 else None
        lats = [e.get("writer_latency_s") for e in pred.get("predictions", [])
                if e.get("writer_latency_s") is not None]
        pred["emit_latency_s"] = {
            "n": len(lats),
            "mean": round(sum(lats) / len(lats), 4) if lats else None,
            "max": round(max(lats), 4) if lats else None,
            "min": round(min(lats), 4) if lats else None,
            "all": [round(x, 4) for x in lats],
        }
        pred["backend"] = getattr(runner, "backend_name", None)
        pred["model_id"] = getattr(runner.base_cfg, "model_id", None)
        pred["use_audio"] = getattr(runner.base_cfg, "use_audio", None)
        pred["arm"] = os.environ.get("OMNIPRO_ARM", "")
        pred["shard"] = args.shard
        _append_jsonl(pred_path, [pred])
        sc = score_sample(pred, tolerance=args.tolerance, judge=judge)
        log(f"[online] {i+1}/{len(samples)} {s.task} video={s.video_id} "
            f"emits={sc['n_emits']} gt={sc['n_gt']} tp={sc['tp_time']} fp={sc['fp']} "
            f"fn={sc['fn']} ({pred['wall_s']}s wall / {pred['video_len_s']}s video "
            f"= {pred['realtime_factor']}x)")

    rows = read_jsonl(pred_path)
    agg = aggregate([score_sample(r, tolerance=args.tolerance, judge=judge) for r in rows])
    agg["n"] = len(rows)
    write_json(os.path.join(args.out, "online_metrics.json"), agg)
    o = agg["overall"]
    log(f"time_f1={o['time_f1']} joint_f1={o['joint_f1']} "
        f"content_acc={o['content_acc']} (n={len(rows)})")
    if not o.get("content_complete", True):
        # loud, because a withheld content metric means this run's headline
        # joint-F1 does not exist yet -- it is not zero, it is unmeasured.
        log(f"[judge] CONTENT WITHHELD: {o['n_unjudged']}/{o['n_matched']} matched "
            f"emits went UNJUDGED (coverage {o['content_coverage']:.1%}). "
            f"time_* are final; run judge_offline.py to fill in content.")


if __name__ == "__main__":
    main()
