"""
evaluate.py — full OmniPro × system_5 benchmark: ALL tasks, ALL prompt variants,
both metric families.

Two engines, two processes (their packages share flat module names and must not
co-import):
  * ONLINE metrics  -> system_5 (untouched async pipeline)  via System5Runner
  * PROBE  metrics  -> system_5_probe (editable copy)        via ProbeRunner

Modes:
  --mode online   run the async pipeline, write per-variant online predictions+metrics
  --mode probe    run the GT-probe protocol, write per-variant probe records+metrics
  --mode merge     read both, write results.md + summary.json + wandb (one run/variant)
  --mode all       online phase + probe phase (as subprocesses) + merge

Resumable (per-sample, by id) and shardable (--shard i --nshards N) so the full
2,700 × 10 sweep can run as a SLURM array and restart without redoing work.

Outputs under eval/output/<variant>/: online_pred*.jsonl, online_metrics.json,
probe_rec*.jsonl, probe_metrics.json; and top-level results.md, summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

from utils import (OUTPUT_DIR, WANDB_ENTITY, WANDB_PROJECT, WandbRun, log,
                   read_jsonl, set_seed, write_json)
from dataset import ALL_TASKS, load_samples
from prompts import get_variants


# --------------------------------------------------------------------------
# resume helpers
# --------------------------------------------------------------------------
def _done_ids(vdir: str, stem: str) -> set:
    done = set()
    for fp in glob.glob(os.path.join(vdir, f"{stem}*.jsonl")):
        for r in read_jsonl(fp):
            done.add(r["id"])
    return done


def _append_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


# --------------------------------------------------------------------------
# ONLINE phase
# --------------------------------------------------------------------------
def run_online(args, variants, samples):
    if args.system == "system4":
        from system4_adapter import System4Runner as Runner
    else:
        from system5_adapter import System5Runner as Runner
    from metrics import ContentJudge, aggregate, score_sample
    judge = ContentJudge()
    runner = Runner(model_id=args.model_id, dtype=args.dtype,
                    kv_budget=(args.kv_budget or None))
    max_seconds = None if (args.max_seconds or 0) <= 0 else args.max_seconds
    shard_tag = "" if args.nshards <= 1 else f".shard{args.shard}"

    for v in variants:
        vdir = os.path.join(args.out, v.key)
        os.makedirs(vdir, exist_ok=True)
        pred_path = os.path.join(vdir, f"online_pred{shard_tag}.jsonl")
        done = _done_ids(vdir, "online_pred") if args.resume else set()
        log(f"[online] variant {v.key}: {len(done)} done, "
            f"{len(samples)} in shard", tag="online")
        for i, s in enumerate(samples):
            if s.id in done:
                continue
            fields = v.fill(s.question, s.event)
            t0 = time.time()
            pred = runner.run_sample(s, fields, max_seconds=max_seconds,
                                     realtime=args.realtime, fps=args.fps)
            pred["wall_s"] = round(time.time() - t0, 2)
            pred["variant"] = v.key
            _append_jsonl(pred_path, [pred])
            sc = score_sample(pred, tolerance=args.tolerance, judge=judge)
            log(f"[online] {v.key} {i+1}/{len(samples)} {s.task} emits={sc['n_emits']} "
                f"gt={sc['n_gt']} tp={sc['tp_time']} fp={sc['fp']} fn={sc['fn']} "
                f"({pred['wall_s']}s)")
        # (re)compute metrics from ALL shards for this variant
        rows = []
        for fp in glob.glob(os.path.join(vdir, "online_pred*.jsonl")):
            rows += read_jsonl(fp)
        per = [score_sample(r, tolerance=args.tolerance, judge=judge) for r in rows]
        agg = aggregate(per)
        agg.update(variant=v.key, desc=v.desc, n=len(rows))
        write_json(os.path.join(vdir, "online_metrics.json"), agg)
        log(f"[online] {v.key} -> time_f1={agg['overall']['time_f1']} "
            f"joint_f1={agg['overall']['joint_f1']} (n={len(rows)})")


# --------------------------------------------------------------------------
# PROBE phase
# --------------------------------------------------------------------------
def run_probe(args, variants, samples):
    if args.system == "system4":
        from system4_probe_adapter import System4ProbeRunner as ProbeRunner
    else:
        from system5_probe_adapter import ProbeRunner
    from metrics import ContentJudge, probe_metrics
    judge = ContentJudge()
    runner = ProbeRunner(model_id=args.model_id, dtype=args.dtype,
                         kv_budget=(args.kv_budget or None))
    max_seconds = None if (args.max_seconds or 0) <= 0 else args.max_seconds
    shard_tag = "" if args.nshards <= 1 else f".shard{args.shard}"

    for v in variants:
        vdir = os.path.join(args.out, v.key)
        os.makedirs(vdir, exist_ok=True)
        rec_path = os.path.join(vdir, f"probe_rec{shard_tag}.jsonl")
        done = _done_ids(vdir, "probe_rec") if args.resume else set()
        log(f"[probe] variant {v.key}: {len(done)} done, {len(samples)} in shard",
            tag="probe")
        for i, s in enumerate(samples):
            if s.id in done:
                continue
            fields = v.fill(s.question, s.event)
            t0 = time.time()
            recs = runner.run_sample(s, fields, fps=args.fps, max_seconds=max_seconds,
                                     content=True)
            _append_jsonl(rec_path, recs)
            log(f"[probe] {v.key} {i+1}/{len(samples)} {s.task} triggers={len(recs)} "
                f"({round(time.time()-t0,2)}s)")
        rows = []
        for fp in glob.glob(os.path.join(vdir, "probe_rec*.jsonl")):
            rows += read_jsonl(fp)
        pm = probe_metrics(rows, threshold=v.goal_threshold, judge=judge)
        pm.update(variant=v.key, desc=v.desc, n=len(rows))
        write_json(os.path.join(vdir, "probe_metrics.json"), pm)
        log(f"[probe] {v.key} -> paired_acc={pm['overall']['paired_accuracy']} "
            f"content_acc={pm['overall']['content_accuracy']} (n={len(rows)})")


# --------------------------------------------------------------------------
# MERGE phase  (no heavy imports -> safe to run in the orchestrator process)
# --------------------------------------------------------------------------
def run_merge(args, variants, meta):
    # Recompute metrics from ALL shard prediction files (authoritative — avoids
    # the per-shard recompute race). Needs only metrics.py (no model import).
    from metrics import ContentJudge, aggregate, probe_metrics, score_sample
    judge = ContentJudge()
    rows = []
    for v in variants:
        vdir = os.path.join(args.out, v.key)

        on = None
        opreds = []
        for fp in glob.glob(os.path.join(vdir, "online_pred*.jsonl")):
            opreds += read_jsonl(fp)
        if opreds:
            per = [score_sample(r, tolerance=args.tolerance, judge=judge) for r in opreds]
            on = aggregate(per)
            on.update(variant=v.key, desc=v.desc, n=len(opreds))
            write_json(os.path.join(vdir, "online_metrics.json"), on)

        pr = None
        precs = []
        for fp in glob.glob(os.path.join(vdir, "probe_rec*.jsonl")):
            precs += read_jsonl(fp)
        if precs:
            pr = probe_metrics(precs, threshold=v.goal_threshold, judge=judge)
            pr.update(variant=v.key, desc=v.desc, n=len(precs))
            write_json(os.path.join(vdir, "probe_metrics.json"), pr)

        rows.append({"variant": v.key, "desc": v.desc, "online": on, "probe": pr})

        wb = WandbRun(not args.no_wandb, project=WANDB_PROJECT, entity=WANDB_ENTITY,
                      name=f"{args.run_name}-{v.key}", group="prompt-sweep",
                      config={"variant": v.key, "desc": v.desc,
                              "goal_threshold": v.goal_threshold, **meta},
                      tags=["omnipro", "system5", "prompt-sweep"])
        if on:
            wb.summary({f"online/{k}": val for k, val in on["overall"].items()})
        if pr:
            wb.summary({f"probe/{k}": val for k, val in pr["overall"].items()})
        if on:
            wb.log_table("online_per_task",
                         ["task", "n", "time_f1", "joint_f1", "content_acc"],
                         [[t, m["n_samples"], m["time_f1"], m["joint_f1"], m["content_acc"]]
                          for t, m in on["per_task"].items()])
        if pr:
            wb.log_table("probe_per_task",
                         ["task", "n", "paired_acc", "post_acc", "content_acc"],
                         [[t, m["n"], m["paired_accuracy"], m["post_accuracy"],
                           m["content_accuracy"]] for t, m in pr["per_task"].items()])
        wb.finish()

    write_json(os.path.join(args.out, "summary.json"), {"meta": meta, "variants": rows})
    # results md always lands in the top-level OUTPUT_DIR under the requested name
    _write_md(os.path.join(OUTPUT_DIR, args.results_name), rows, meta)


def _write_md(path, rows, meta):
    L = ["# OmniPro × system_5 — Full Benchmark (all tasks, all prompts)", "",
         f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ", "",
         "## Configuration"]
    for k, val in meta.items():
        L.append(f"- **{k}**: {val}")
    L += ["",
          "> **Online** metrics: system_5's real async pipeline (3 threads, shared "
          "KV cache) — unmodified. **Probe** metrics: the `system_5_probe` copy, "
          "synchronous GT-probe protocol. system_5 is vision-only; OmniPro `required` "
          "samples need audio. Free-text content judge: "
          f"`{meta.get('content_judge')}`.", ""]

    def og(r, k):
        return (r["online"]["overall"][k] if r.get("online") else None)

    def pg(r, k):
        return (r["probe"]["overall"][k] if r.get("probe") else None)

    ranked = sorted(rows, key=lambda r: (og(r, "time_f1") or -1), reverse=True)
    L += ["## Leaderboard", "",
          "| Rank | Variant | Online Time F1 | Online Joint F1 | Online Content | "
          "Probe Paired-Acc | Probe Post-Acc | Probe Content |",
          "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(ranked, 1):
        L.append(f"| {i} | `{r['variant']}` | {og(r,'time_f1')} | {og(r,'joint_f1')} | "
                 f"{og(r,'content_acc')} | {pg(r,'paired_accuracy')} | "
                 f"{pg(r,'post_accuracy')} | {pg(r,'content_accuracy')} |")
    L.append("")
    best = ranked[0]
    L += ["## Best variant (by Online Time F1)",
          f"**`{best['variant']}`** — {best['desc']}", ""]

    L.append("## Per-task breakdown")
    for r in ranked:
        L.append(f"### `{r['variant']}` — {r['desc']}")
        if r.get("online"):
            L += ["**Online** (time P/R/F1, joint F1, content acc):",
                  "| Task | n | Time P | Time R | Time F1 | Joint F1 | Content |",
                  "|---|---|---|---|---|---|---|"]
            for t, m in r["online"]["per_task"].items():
                L.append(f"| {t} | {m['n_samples']} | {m['time_precision']} | "
                         f"{m['time_recall']} | {m['time_f1']} | {m['joint_f1']} | "
                         f"{m['content_acc']} |")
        if r.get("probe"):
            L += ["", "**Probe** (paired/pre/post accuracy, content acc):",
                  "| Task | n | Paired | Pre | Post | Content |",
                  "|---|---|---|---|---|---|"]
            for t, m in r["probe"]["per_task"].items():
                L.append(f"| {t} | {m['n']} | {m['paired_accuracy']} | "
                         f"{m['pre_accuracy']} | {m['post_accuracy']} | "
                         f"{m['content_accuracy']} |")
        L.append("")
    open(path, "w").write("\n".join(L))
    log(f"wrote {path}")


# --------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["online", "probe", "merge", "all"])
    ap.add_argument("--system", default="system5", choices=["system5", "system4"])
    ap.add_argument("--results_name", default="results.md")
    ap.add_argument("--tasks", default=",".join(ALL_TASKS))
    ap.add_argument("--audio", default="none_helpful",
                    choices=["none", "helpful", "required", "none_helpful", "all"])
    ap.add_argument("--limit", type=int, default=0, help="samples per task; 0=all")
    ap.add_argument("--shortest", action="store_true",
                    help="with --limit, keep the SHORTEST videos per task (fast subset)")
    ap.add_argument("--benchmark_json", default=None,
                    help="annotation file to score; default = bundled benchmark_mini.json")
    ap.add_argument("--dataset_dir", default=None,
                    help="root the video_path entries are joined onto")
    ap.add_argument("--max_seconds", type=float, default=0, help="0=full video")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--variants", default="")
    ap.add_argument("--model_id", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--kv_budget", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", dest="resume", action="store_false")
    ap.add_argument("--out", default=OUTPUT_DIR)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--run_name", default=None)
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(1234)
    args.run_name = args.run_name or time.strftime("full-%m%d-%H%M")
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    variants = get_variants([v.strip() for v in args.variants.split(",") if v.strip()] or None)

    from utils import BENCHMARK_JSON, DATASET_DIR
    samples = load_samples(tasks=tasks, audio=args.audio,
                           limit_per_task=(args.limit or None), max_duration=None,
                           shortest=args.shortest,
                           benchmark_json=(args.benchmark_json or BENCHMARK_JSON),
                           dataset_dir=(args.dataset_dir or DATASET_DIR))
    if args.nshards > 1:
        samples = samples[args.shard::args.nshards]
        log(f"shard {args.shard}/{args.nshards}: {len(samples)} samples")
    if not samples and args.mode != "merge":
        log("no samples; aborting.")
        return

    meta = {"system": args.system,
            "tasks": tasks, "audio_subset": args.audio, "limit_per_task": args.limit or "all",
            "n_samples_total": len(samples), "n_variants": len(variants),
            "fps": args.fps, "realtime": args.realtime, "max_seconds": args.max_seconds or "full",
            "tolerance": args.tolerance, "dtype": args.dtype,
            "content_judge": ("llm" if (os.environ.get("OPENAI_API_KEY") or
                                        os.environ.get("GEMINI_API_KEY")) else "lexical"),
            "wandb": "off" if args.no_wandb else f"{WANDB_ENTITY}/{WANDB_PROJECT}"}

    if args.mode == "online":
        run_online(args, variants, samples)
    elif args.mode == "probe":
        run_probe(args, variants, samples)
    elif args.mode == "merge":
        run_merge(args, variants, meta)
    elif args.mode == "all":
        # run the two engines as separate processes (package isolation), then merge
        base = [sys.executable, os.path.abspath(__file__)]
        passthrough = ["--tasks", args.tasks, "--audio", args.audio,
                       "--limit", str(args.limit), "--max_seconds", str(args.max_seconds),
                       "--fps", str(args.fps), "--tolerance", str(args.tolerance),
                       "--variants", args.variants, "--dtype", args.dtype,
                       "--kv_budget", str(args.kv_budget), "--out", args.out,
                       "--system", args.system, "--results_name", args.results_name,
                       "--shard", str(args.shard), "--nshards", str(args.nshards)]
        if args.benchmark_json:
            passthrough += ["--benchmark_json", args.benchmark_json]
        if args.dataset_dir:
            passthrough += ["--dataset_dir", args.dataset_dir]
        if args.shortest:
            passthrough.append("--shortest")
        if args.realtime:
            passthrough.append("--realtime")
        if not args.resume:
            passthrough.append("--no_resume")
        log("=== ONLINE phase (subprocess) ===")
        subprocess.run(base + ["--mode", "online"] + passthrough, check=True)
        log("=== PROBE phase (subprocess) ===")
        subprocess.run(base + ["--mode", "probe"] + passthrough, check=True)
        log("=== MERGE ===")
        run_merge(args, variants, meta)
    log("DONE.")


if __name__ == "__main__":
    main()
