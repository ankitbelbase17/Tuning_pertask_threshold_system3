#!/usr/bin/env python
"""stage3_state.py -- progress and job shape for the FULL 2,700-sample eval (sec.6).

Same contract as chain_state.py (shell-eval-able KEY=VALUE), same band-selection
maths, but a different notion of "the work":

  chain_state.py : work = CELLS (task x threshold), each holding the frozen 15 ids
  stage3_state.py: work = the WHOLE benchmark, once, with the fitted thresholds

WHY A SEPARATE MODULE INSTEAD OF A FLAG. The two differ in what "remaining" means
and in nothing else, but that one difference touches every line of the loop. A
flag would put two experiments' definitions of done-ness in one function, which is
exactly the shape of bug sec.6.1 is trying to detect.

THE TARGET SET IS TAKEN FROM THE HARNESS, NOT REBUILT. load_samples() is what
evaluate.py itself will run; asking it for the id list means the denominator here
cannot drift from the numerator on disk. sec.6 forbids capping long videos OUT of
stage 3 -- every id must eventually be evaluated -- so `--max_dur` here only ever
defers a sample to a longer-window generation, and REMAIN counts it throughout.
"""
from __future__ import annotations
import argparse, glob, json, os, sys


LOAD_S = float(os.environ.get("CHAIN_LOAD_S", 300))
SAFETY = float(os.environ.get("CHAIN_SAFETY", 0.80))
XRT_PRIOR_P95 = float(os.environ.get("CHAIN_XRT_PRIOR", 3.343))

RESULTS = os.path.join(os.environ["THR_ROOT"], "results", "full2700")


def target_samples():
    """(id -> duration) for every sample stage 3 must evaluate.

    Imported from the harness so this is the same list evaluate.py will build:
    all 9 tasks, --audio all, no subset stride, no duration cap.
    """
    sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "repo", "omniprofast"))
    from dataset import ALL_TASKS, load_samples                  # noqa: E402
    s = load_samples(tasks=list(ALL_TASKS), audio="all",
                     limit_per_task=None, max_duration=None, shortest=False,
                     benchmark_json=os.environ["OMNIPRO_BENCHMARK_JSON"],
                     dataset_dir=os.environ["OMNIPRO_DATASET_DIR"])
    return {x.id: float(x.duration) for x in s}


def banked():
    """ids already evaluated by ANY lane, plus their realtime factors."""
    ids, xs = set(), []
    for fp in glob.glob(os.path.join(RESULTS, "lane*", "online_pred.jsonl")):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # torn line from a SIGKILLed lane
                    if "id" in r:
                        ids.add(r["id"])
                    if r.get("realtime_factor") and not r.get("skipped_oom"):
                        xs.append(r["realtime_factor"])
        except OSError:
            pass                          # /iopsstor flaps; "nothing known yet"
    return ids, xs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--max_nodes", type=int,
                    default=int(os.environ.get("CHAIN_MAX_NODES", 4)))
    a = ap.parse_args()

    dur = target_samples()
    have, xs = banked()
    remaining = {i: d for i, d in dur.items() if i not in have}

    xrt = (sum(xs) / len(xs)) if xs else None
    xrt_p95 = sorted(xs)[int(0.95 * (len(xs) - 1))] if xs else None
    durs = sorted(remaining.values())
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
            if n == 1:                 # the last shape takes everything left
                rec, max_dur, fit_n = 1, 0.0, len(durs)
                break
            if k > 0:
                rec, max_dur, fit_n = n, c, k
                break

    print(f"TARGET={len(dur)}")
    print(f"DONE={len(dur) - len(remaining)}")
    print(f"REMAIN={len(durs)}")
    print(f"CELLS_TOTAL=1")
    print(f"CELLS_DONE={0 if durs else 1}")
    print(f"COMPLETE={1 if not durs else 0}")
    print(f"LONGEST_REMAIN_S={durs[-1] if durs else 0.0:.1f}")
    print(f"XRT_MEAN={xrt:.3f}" if xrt else "XRT_MEAN=")
    print(f"XRT_P95={xrt_p95:.3f}" if xrt_p95 else "XRT_P95=")
    print(f"REC_NODES={rec}")
    print(f"MAX_DUR={max_dur:.0f}")
    print(f"FIT_N={fit_n}")
    print(f"REMAIN_GPUH={sum(durs) * (xrt or XRT_PRIOR_P95) / 3600.0:.1f}")


if __name__ == "__main__":
    main()
