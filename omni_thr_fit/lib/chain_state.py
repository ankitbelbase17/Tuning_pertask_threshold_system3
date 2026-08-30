#!/usr/bin/env python
"""chain_state.py -- what is left, and what job shape can still finish it.

Prints shell-eval-able KEY=VALUE lines. The debug chain calls this between
generations to decide (a) whether the pass is done and (b) how many nodes the
next generation should ask for.

THE RESHAPE PROBLEM (inherited from omni_s3_eval, and re-verified 2026-08-28:
`sacctmgr show qos debug-qos` gives MaxTRESMins node=90, MaxTRES node=4).
debug-qos budgets 90 NODE-MINUTES per job and caps a job at 4 nodes, so:

    4 nodes = 16 GPUs, 22 min window
    2 nodes =  8 GPUs, 45 min window
    1 node  =  4 GPUs, 90 min window

All three buy identical compute; only the WINDOW differs. Resume is per-SAMPLE,
so a sample whose eval exceeds the window can never checkpoint -- it is started,
killed at the wall, and retried forever, blocking the pass. OmniPro's longest
video is 594.6 s, so at the measured p95 xRT of 3.34 the worst sample needs
~33 min and does not fit a 22-min window.

So: pick the shape from a BAND, not from the worst sample. Each shape is handed
only the samples that fit its own window (MAX_DUR, enforced by evaluate.py
--max_dur AFTER the subset stride), and the chain steps 4 -> 2 -> 1 as each band
empties. The step-down is monotone -- a band only empties once -- so there is no
oscillation, and the long tail is reached at the one shape that can clear it.

WHAT DIFFERS FROM omni_s3_eval's VERSION. There the unit of work was a sample of
one big run. Here it is a CELL (task x threshold), each holding the same frozen
15 samples, so "remaining" is computed over cells x their frozen ids rather than
over one benchmark sweep.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worklist as W                                            # noqa: E402

LOAD_S = float(os.environ.get("CHAIN_LOAD_S", 300))    # model load + startup per generation
SAFETY = float(os.environ.get("CHAIN_SAFETY", 0.80))   # only trust 80% of the window

# COLD-START PRIOR. A pass with zero completed samples has no measured xRT, so the
# band selection has nothing to size a window with. The prior is a p95, not a
# mean, because the window must clear the WORST sample it might pick up. 3.34 is
# the p95 measured over this fork's own completed 2,700-sample phase-A run
# (mean 1.947, p95 3.343). Deliberately conservative: too high only costs a
# tighter cap for one generation; too low means a sample that cannot finish its
# window and is retried forever.
XRT_PRIOR_P95 = float(os.environ.get("CHAIN_XRT_PRIOR", 3.343))


def cell_rows(cell):
    """Every prediction row banked in a cell, across all its lanes."""
    rows = []
    for fp in glob.glob(os.path.join(cell, "lane*", "online_pred.jsonl")):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass          # a lane SIGKILLed mid-write leaves a torn line
        except OSError:
            pass                      # /iopsstor flaps; treat as "nothing known yet"
    return rows


def default_worklist():
    """Which pass is ACTIVE is read from a FILE, not baked into the chain's env.

    A self-chaining job re-submits itself for days and reshapes 4->2->1 nodes.  An
    env var set at the first submit is one dropped --export away from vanishing,
    and that failure is SILENT in the worst way: the chain falls back to pass 1,
    finds it already COMPLETE, prints "nothing to do" and stands down -- looking
    like a finished run rather than a broken one.  A file is re-read every
    generation, survives any relaunch by any fleet, and can be switched while the
    chain is running (the same reason LOGIN_GPUS is a file):
        echo worklist_p2.tsv > $THR_ROOT/ACTIVE_WORKLIST
    Precedence: --worklist > $THR_WORKLIST > ACTIVE_WORKLIST > worklist_p1.tsv.
    """
    root = os.environ["THR_ROOT"]
    env = os.environ.get("THR_WORKLIST", "").strip()
    if env:
        return env if os.path.isabs(env) else os.path.join(root, env)
    try:
        name = open(os.path.join(root, "ACTIVE_WORKLIST")).read().strip()
    except OSError:
        name = ""
    if name:
        return name if os.path.isabs(name) else os.path.join(root, name)
    return os.path.join(root, "worklist_p1.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worklist", default=default_worklist())
    ap.add_argument("--nodes", type=int, default=4,
                    help="the CURRENT generation's node count (reporting only)")
    ap.add_argument("--max_nodes", type=int,
                    default=int(os.environ.get("CHAIN_MAX_NODES", 4)),
                    help="ceiling on the shape; 4 = debug-qos maximum")
    a = ap.parse_args()

    # id -> duration, for the band selection. The frozen splits carry the ids;
    # the benchmark carries the durations.
    dur = {}
    with open(os.environ["OMNIPRO_BENCHMARK_JSON"]) as f:
        for e in json.load(f):
            dur[e["id"]] = float(e.get("duration", 0.0))

    units = W.read_worklist(a.worklist)
    cells = sorted({(p, t, thr) for p, t, thr, _ in units})

    target = done = 0
    remaining_durs, xs = [], []
    cells_done = 0
    for (p, t, thr) in cells:
        want = W.frozen_ids(t)
        cd = W.cell_dir(p, t, thr)
        have = W.done_ids(cd)
        target += len(want)
        done += len(want & have)
        cells_done += (want <= have)
        remaining_durs += [dur.get(i, 0.0) for i in (want - have)]
        for r in cell_rows(cd):
            if r.get("realtime_factor") and not r.get("skipped_oom"):
                xs.append(r["realtime_factor"])

    xrt = (sum(xs) / len(xs)) if xs else None
    # size the window off the SLOW end, not the mean: one window has to clear the
    # worst sample it might pick up, not the average one.
    xrt_p95 = sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else None
    durs = sorted(remaining_durs)
    longest = durs[-1] if durs else 0.0
    xrt_shape = xrt_p95 if xrt_p95 else (XRT_PRIOR_P95 if durs else None)

    def cap(n):
        return ((90.0 / n) * 60.0 * SAFETY - LOAD_S) / xrt_shape

    rec, max_dur, fit_n = a.nodes, 0.0, len(durs)
    if xrt_shape and durs:
        for n in (4, 2, 1):
            if n > a.max_nodes:
                continue
            c = cap(n)
            k = sum(1 for d in durs if d <= c)
            if n == 1:                     # the last shape takes everything left
                rec, max_dur, fit_n = 1, 0.0, len(durs)
                break
            if k > 0:                      # this shape still has work it can clear
                rec, max_dur, fit_n = n, c, k
                break

    print(f"TARGET={target}")
    print(f"DONE={done}")
    print(f"REMAIN={len(durs)}")
    print(f"CELLS_TOTAL={len(cells)}")
    print(f"CELLS_DONE={cells_done}")
    print(f"COMPLETE={1 if not durs else 0}")
    print(f"LONGEST_REMAIN_S={longest:.1f}")
    print(f"XRT_MEAN={xrt:.3f}" if xrt else "XRT_MEAN=")
    print(f"XRT_P95={xrt_p95:.3f}" if xrt_p95 else "XRT_P95=")
    print(f"REC_NODES={rec}")
    print(f"MAX_DUR={max_dur:.0f}")        # 0 = no cap (the 1-node shape takes all)
    print(f"FIT_N={fit_n}")
    print(f"REMAIN_GPUH={sum(durs) * (xrt or XRT_PRIOR_P95) / 3600.0:.1f}")


if __name__ == "__main__":
    main()
