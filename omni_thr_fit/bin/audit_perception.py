#!/usr/bin/env python
"""audit_perception.py -- is p_hit informative about event TIMING at all?

  python bin/audit_perception.py                    # -> PERCEPTION_AUDIT.json
  python bin/audit_perception.py --pass p1 --stage3 results/stage3

THE QUESTION. The whole threshold fit assumes p_hit ranks event-adjacent ticks
above quiet ones, and that the only open question is WHERE to cut. If that
assumption is false the fit is optimising a coin, every threshold performs the
same up to sampling noise, and the honest deliverable is that negative result --
not a fitted number presented as a tuning win.

WHY THIS EXISTS SEPARATELY FROM Fig 2. Fig 2 reports AUC at the operating point.
A near-chance AUC there has three innocent explanations that must be excluded
before it can be reported as a property of the model:

  (a) A CLOCK OFFSET between the `vid Ns` stamp in the log and the benchmark's
      trigger_time_sec. A constant shift destroys AUC while leaving a perfectly
      informative score. Excluded by sweeping the offset: a real offset shows up
      as a clear peak away from zero.
  (b) TOO TIGHT A TOLERANCE. If the model anticipates or lags events by more than
      +-3 s, the +-3 s label calls its correct high scores negatives. Excluded by
      re-running at +-10 s.
  (c) A DEGENERATE SCORE -- p_hit pinned at one value, or quantised to a handful
      of levels, in which case AUC is near 0.5 by construction and says nothing
      about perception. Excluded by reporting the distribution and the count of
      distinct values.

Only when all three are excluded does "AUC = chance" mean what it appears to mean.

MEASURED 2026-08-29 on the five pass-1 tasks that had a complete reference cell:
AUC stayed inside 0.39-0.55 at every offset in +-10 s and at both tolerances,
with no peak anywhere, while p_hit spanned 0.000-0.997 with 230-284 distinct
values and an interquartile range of 0.2-0.7. So the score is well spread and
genuinely uninformative about WHEN, not merely mis-aligned or saturated.
semantic_condition_alert came out BELOW chance and fell monotonically as the
offset went positive (0.487 at -10 s to 0.389 at +10 s): its confidence is
systematically LOWER near an event.

CAVEAT THAT MUST TRAVEL WITH THE NUMBER. n = 15 videos per task on the frozen fit
subset. The bootstrap CIs straddle 0.5, so the claim this supports is
"indistinguishable from chance at this n", never "proven to be exactly chance".
Point --stage3 at the full run once it exists and the same audit has n = 300.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
sys.path.insert(0, os.path.join(ROOT, "repo", "async_omni_v2"))

import worklist as W                                             # noqa: E402
from auc import auc_score, load_gt, parse_scores                 # noqa: E402

OFFSETS = list(range(-10, 11, 2))
TOLS = (3.0, 10.0)


def label(scores, gt, tol, off):
    """Same rule as auc.label_ticks, plus a shift applied to the TICK clock so a
    constant misalignment between the two time bases becomes visible."""
    y, p = [], []
    for vid, ticks in scores.items():
        g = gt.get(vid)
        if not g:
            continue
        for vt, ph in ticks:
            y.append(1 if any(abs(vt + off - t) <= tol for t in g) else 0)
            p.append(ph)
    return y, p


def cell_for(pass_, task):
    thrs = sorted({r[2] for r in W.read_worklist(
        os.path.join(ROOT, f"worklist_{pass_}.tsv")) if r[1] == task})
    if not thrs:
        return None, None
    thr = thrs[len(thrs) // 2]
    cell = W.cell_dir(pass_, task, thr)
    if len(W.done_ids(cell)) < len(W.frozen_ids(task)):
        return None, None          # incomplete cell = a different video set
    return cell, thr


def audit(paths, gt, tol_offsets=True):
    scores = parse_scores(paths)
    if not scores:
        return None
    allp = sorted(ph for ticks in scores.values() for _, ph in ticks)
    if not allp:
        return None
    q = lambda f: allp[int(f * (len(allp) - 1))]
    out = {"n_videos": len(scores), "n_ticks": len(allp),
           "p_hit": {"min": allp[0], "p25": q(.25), "med": q(.5),
                     "p75": q(.75), "max": allp[-1],
                     "n_distinct": len(set(allp))},
           "auc": {}}
    for tol in TOLS:
        row = {}
        for off in (OFFSETS if tol_offsets else [0]):
            y, p = label(scores, gt, tol, off)
            a = auc_score(y, p)
            row[str(off)] = a
            if off == 0:
                out.setdefault("positive_rate", {})[str(tol)] = (
                    sum(y) / len(y) if y else None)
        out["auc"][str(tol)] = row
    # the headline: the BEST AUC any offset/tolerance could buy. If the score
    # were merely misaligned this would be well above the zero-offset value.
    best = max((v for r in out["auc"].values() for v in r.values()
                if v is not None), default=None)
    out["auc_best_over_offsets"] = best
    out["auc_at_zero_tol3"] = out["auc"]["3.0"]["0"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--stage3", default=None,
                    help="score the full run instead of the fit-subset cells")
    ap.add_argument("--out", default=os.path.join(ROOT, "PERCEPTION_AUDIT.json"))
    a = ap.parse_args()

    bench = os.environ.get("OMNIPRO_BENCHMARK_JSON")
    if not bench or not os.path.exists(bench):
        sys.exit("no OMNIPRO_BENCHMARK_JSON")

    tasks = sorted({r[1] for r in W.read_worklist(
        os.path.join(ROOT, f"worklist_{a.pass_}.tsv"))})
    res = {}
    for t in tasks:
        if a.stage3:
            paths = sorted(glob.glob(os.path.join(a.stage3, "**", "run.log"),
                                     recursive=True))
            src = a.stage3
        else:
            cell, thr = cell_for(a.pass_, t)
            if not cell:
                res[t] = {"skipped": "no complete reference cell"}
                print(f"{t:30s} SKIP (no complete cell)")
                continue
            paths = sorted(glob.glob(os.path.join(cell, "lane*", "run.log")))
            src = f"{a.pass_} thr {thr:.2f}"
        r = audit(paths, load_gt(bench, task=t))
        if r is None:
            res[t] = {"skipped": "no p_hit ticks in logs"}
            print(f"{t:30s} SKIP (no ticks)")
            continue
        r["source"] = src
        res[t] = r
        print(f"{t:30s} n={r['n_videos']:3d}vid {r['n_ticks']:5d}tick  "
              f"AUC@0/tol3 {r['auc_at_zero_tol3']:.3f}  "
              f"best over offsets {r['auc_best_over_offsets']:.3f}  "
              f"p_hit {r['p_hit']['min']:.3f}-{r['p_hit']['max']:.3f} "
              f"({r['p_hit']['n_distinct']} distinct)")
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
