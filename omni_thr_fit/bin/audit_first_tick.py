#!/usr/bin/env python
"""audit_first_tick.py -- how much of the threshold grid is DEGENERATE, and why.

THE OBSERVATION THAT PROMPTED THIS. In pass 1, every instant_event_alert sample
fired exactly once, at vid 1.0 s, while ground truth sat at 26-98 s -- so
time-F1 was structurally 0 at every threshold tried. The gate trace explains it:

    vid 1.0s  p_hit=0.792  rise=True  fire=True     <- almost no context yet
    vid 24.0s p_hit=0.958  rise=True  fire=False    <- the REAL event, suppressed

p_hit is high on the first tick (the model answers from the prompt before it has
seen anything), `prev_above` starts False so that tick counts as a RISING EDGE,
and instant_event_alert's inherited 600 s refractory then locks out the rest of
the video. Threshold cannot matter: any value below the tick-1 p_hit produces the
identical single emit at t=1.

This script measures that over the finished phase-A run, offline and free, and
compares three gate definitions:

  as_is    prev_above=False           -- what the live gate and resweep.py do now
  edge1    prev_above=first tick      -- an edge REQUIRES a predecessor; tick 1
                                        has none, so it cannot be a rising edge
  warmup   suppress fires < W seconds -- also covers `level` mode, but adds an
                                        unfitted fourth knob

"Distinct outcomes" is the number of different emit-sequences the 10-point grid
produces. 1 means the threshold is inert for that task -- the same defect as the
0.5 hysteresis rail this experiment was built to correct.
"""
from __future__ import annotations
import os, sys, json, argparse
from collections import defaultdict

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
from resweep import greedy_match, TOL, _HEADER, _GATE          # noqa: E402
import glob


def parse_logs(pattern):
    """resweep.parse_run, but over an explicit glob.

    parse_run hardcodes `**/run_*.log`; the phase-A fleet wrote `lane<N>.log` at
    the top level, so the built-in glob matches nothing and silently returns an
    empty dict -- an empty audit that looks like a clean one. Same parsing rules:
    ticks are attributed to the header that precedes them, last occurrence wins.
    """
    samples = {}
    for path in sorted(glob.glob(pattern)):
        cur = None
        with open(path, errors="replace") as fh:
            for line in fh:
                h = _HEADER.search(line)
                if h:
                    task, vid, sid, gts = h.group(1), h.group(2), h.group(3), h.group(4)
                    cur = {"task": task, "video": vid, "ticks": [],
                           "gt": [float(x) for x in gts.split(",") if x.strip()]}
                    samples[f"{task}::{vid}::{sid}"] = cur
                    continue
                if cur is None:
                    continue
                g = _GATE.search(line)
                if g:
                    cur["ticks"].append((float(g.group(1)), float(g.group(3))))
    return samples

GRID = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]


def simulate(ticks, thr, mode, refractory, rule="as_is", warmup=0.0):
    ticks = sorted(ticks)
    if not ticks:
        return []
    emits, last = [], None
    # THE ONE LINE THIS AUDIT IS ABOUT.
    prev = (ticks[0][1] >= thr) if rule == "edge1" else False
    for vt, p in ticks:
        cur = p >= thr
        want = (cur and not prev) if mode == "edge" else cur
        if want and vt >= warmup and (last is None or vt - last >= refractory):
            emits.append(vt)
            last = vt
        prev = cur
    return emits


def f1(samples, thr, mode, refr, rule, warmup):
    tp = fp = fn = 0
    for s in samples:
        e = simulate(s["ticks"], thr, mode, refr, rule, warmup)
        a, b, c = greedy_match(e, s["gt"], TOL)
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), tp + fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/iopsstor/scratch/cscs/dbartaula/omni_s3_eval/"
                                     "results/phaseA_omni_full/lane*.log",
                    help="glob of run logs carrying ctrl.gate traces")
    ap.add_argument("--warmup", type=float, default=5.0)
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(ROOT, "repo", "async_omni_v2"))
    from config import AsyncOmniConfig as C
    cfg = C()

    samples = parse_logs(a.run)
    by_task = defaultdict(list)
    for s in samples.values():
        by_task[s["task"]].append(s)
    print(f"parsed {len(samples)} samples from {a.run}\n")

    # how often is the FIRST tick already above a low threshold?
    print("first-tick p_hit distribution (the artifact itself):")
    print(f"  {'task':<30}{'n':>5}{'med p1':>9}{'p1>=.05':>9}{'p1>=.45':>9}{'med all':>9}")
    for t in sorted(by_task):
        ss = by_task[t]
        firsts = [sorted(s["ticks"])[0][1] for s in ss if s["ticks"]]
        allp = [p for s in ss for _, p in s["ticks"]]
        if not firsts:
            continue
        med = sorted(firsts)[len(firsts) // 2]
        meda = sorted(allp)[len(allp) // 2]
        print(f"  {t:<30}{len(ss):>5}{med:>9.3f}"
              f"{sum(1 for x in firsts if x >= 0.05) / len(firsts):>9.2f}"
              f"{sum(1 for x in firsts if x >= 0.45) / len(firsts):>9.2f}{meda:>9.3f}")

    print("\ngrid degeneracy and best time-F1, per gate definition")
    print("(distinct = how many different emit-sets the 10-point grid produces;"
          " 1 = threshold inert)")
    print(f"\n  {'task':<28}{'refr':>6}{'mode':>6}"
          f"{'as_is':>18}{'edge1':>18}{f'warmup{a.warmup:g}s':>18}")
    print(f"  {'':<28}{'':>6}{'':>6}" + f"{'dist  F1@best':>18}" * 3)
    out = {}
    for t in sorted(by_task):
        ss = by_task[t]
        refr = cfg.task_refractory_s.get(t, 0.0)
        mode = cfg.task_gate_modes.get(t, "edge")
        row = f"  {t:<28}{refr:>6.0f}{mode:>6}"
        out[t] = {}
        for rule, wu in (("as_is", 0.0), ("edge1", 0.0), ("as_is", a.warmup)):
            sigs, best, bthr = set(), -1.0, None
            for thr in GRID:
                sig = tuple(tuple(simulate(s["ticks"], thr, mode, refr, rule, wu))
                            for s in ss)
                sigs.add(sig)
                v, _ = f1(ss, thr, mode, refr, rule, wu)
                if v > best:
                    best, bthr = v, thr
            key = rule if wu == 0.0 else f"warmup{a.warmup:g}"
            out[t][key] = {"distinct": len(sigs), "best_f1": round(best, 4),
                           "best_thr": bthr}
            row += f"{len(sigs):>7}  {best:>9.4f}"
        print(row)

    with open(os.path.join(ROOT, "FIRST_TICK_AUDIT.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {ROOT}/FIRST_TICK_AUDIT.json")


if __name__ == "__main__":
    main()
