#!/usr/bin/env python
"""audit_fit_noise.py -- is the per-task threshold FIT reproducible, or is it noise?

  python bin/audit_fit_noise.py --pass p1        # -> FIT_NOISE_AUDIT.json

THE QUESTION, AND WHY IT IS NOT THE ONE pick.py ANSWERS. pick.py reports WHICH
threshold won and by how much. That is not the same as whether the win is real.
A grid of 10-12 cells scored on 15 videos will always produce a maximum; the
question is whether the SAME threshold would win again on a different sample of
videos from the same distribution. If it would not, the "fitted" value is a draw
from the noise and reporting it as a tuned per-task parameter is a lie about
where the number came from.

THE STATISTIC: SELECTION STABILITY. Resample the frozen video ids with
replacement; on each draw recompute the pooled time_f1 of EVERY cell in the grid
and record which threshold wins. The frequency of the originally-fitted threshold
in that distribution is the honest answer to "did we fit anything". A stable fit
re-selects its own winner in most draws; a fit of noise scatters the argmax across
the grid, and a grid of k cells with no signal lands near 1/k.

WHY THE BOOTSTRAP IS PAIRED, AND WHY THAT MATTERS HERE. Every cell of a task runs
the SAME fifteen frozen ids, so a draw can hold the video set fixed across cells
and compare thresholds on identical videos. That removes between-video variance --
which dominates, since videos differ far more from each other than thresholds do
on any one video -- and makes this a much sharper test than comparing two
independent confidence intervals. It cuts BOTH ways: it is the most favourable
test the fit could ask for. Whatever it fails to establish here, it would fail
harder unpaired.

RESAMPLE VIDEOS, NOT EVENTS. A video contributes several GT events whose outcomes
are correlated through one shared KV cache and one gate trajectory. Resampling
events i.i.d. would treat them as independent and return an interval several
times too narrow -- the same error lib/figs_auc.py documents for ticks.

POOLING MATCHES metrics.aggregate: micro-average tp/fp/fn over the drawn ids and
take F1 of the pooled counts, NOT the mean of per-video F1s. The two differ, and
the pooled form is the one every other number in this study reports.

Scoring itself is metrics.score_sample VERBATIM (spec sec.8) -- this module only
groups and resamples.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter, defaultdict

# NO JUDGE. This audit is pure timing. score_sample calls the judge on every
# MATCH regardless, and ContentJudge.offline is set after __init__ has already
# armed the client -- so unset the keys the way bin/worker.sh does, which is the
# documented way to make the judge report "unavailable" and return None with no
# network at all. Doing this BEFORE the import is what makes it stick.
for _k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GEMINI_API_BASE", "OPENAI_API_BASE"):
    os.environ.pop(_k, None)

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "omniprofast"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "async_omni_v2"))

import worklist as W                                                # noqa: E402
import pick as P                                                    # noqa: E402
from score_cells import cell_predictions                            # noqa: E402
from metrics import ContentJudge, score_sample, _prf                # noqa: E402

NOISE = 0.03          # sec.6.2's measured run-to-run band on time_f1
B = 2000


def pooled(counts, ids):
    """One cell's micro-averaged row over `ids`, in the shape pick.rank expects."""
    tp = fp = fn = ne = ng = 0
    for i in ids:
        c = counts.get(i)
        if c:
            tp += c[0]; fp += c[1]; fn += c[2]; ne += c[3]; ng += c[4]
    return {"time_f1": _prf(tp, fp, fn)[2],
            "emit_per_gt": (ne / ng) if ng else None}


def select(counts, grid, ids, metric="time_f1"):
    """The threshold pick.py WOULD choose on this sample -- its own rank(), not a
    bare argmax. The two differ: sec.3's tie-band prefers the emission volume
    closest to the truth whenever the top two are within 0.02, so on three of the
    nine tasks the shipped pick is the runner-up by raw F1. Measuring the
    stability of an argmax we do not actually use would answer a question nobody
    asked."""
    rows = []
    for thr in grid:
        r = pooled(counts[thr], ids)
        r["thr"] = thr
        rows.append(r)
    return P.rank(rows, metric)[0]["thr"], {r["thr"]: r["time_f1"] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--boot", type=int, default=B)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default=os.path.join(ROOT, "FIT_NOISE_AUDIT.json"))
    a = ap.parse_args()

    judge = ContentJudge()          # keys already unset above -> returns None
    judge.offline = True

    units = W.read_worklist(os.path.join(ROOT, f"worklist_{a.pass_}.tsv"))
    by_task = defaultdict(set)
    for _p, t, thr, _sh in units:
        by_task[t].add(round(float(thr), 4))

    picks = {}
    pf = os.path.join(ROOT, "P1_PICKS.json" if a.pass_ == "p1" else "FINAL_THRESHOLDS.json")
    if os.path.exists(pf):
        picks = json.load(open(pf))

    out = {}
    for task in sorted(by_task):
        grid = sorted(by_task[task])
        # id -> (tp,fp,fn) per cell, and the id set common to every cell so the
        # paired draw compares thresholds on identical videos.
        counts, common = {}, None
        for thr in grid:
            c = {}
            for p in cell_predictions(W.cell_dir(a.pass_, task, thr)):
                s = score_sample(p, tolerance=3.0, judge=judge)
                c[s["id"]] = (s["tp_time"], s["fp"], s["fn"],
                              s["n_emits"], s["n_gt"])
            counts[thr] = c
            common = set(c) if common is None else (common & set(c))
        common = sorted(common or [])
        if len(common) < 2 or len(grid) < 2:
            out[task] = {"skipped": f"{len(common)} common ids over {len(grid)} cells"}
            print(f"{task:30s} SKIP"); continue

        sel, obs = select(counts, grid, common)
        order = sorted(grid, key=lambda t: -obs[t])
        best = order[0]
        # The two passes record their pick in DIFFERENT SHAPES and the audit has
        # to read both: P1_PICKS.json maps task -> {"best": thr, ...} while
        # FINAL_THRESHOLDS.json maps task -> thr. Assuming the p1 shape raises on
        # p2, which is the good failure mode, but it still has to be handled.
        pk = picks.get(task)
        pk = pk.get("best") if isinstance(pk, dict) else pk
        fitted = round(float(sel if pk is None else pk), 4)
        # sec.4 lets a COARSE pass-1 candidate win the finalise ranking, so the
        # shipped threshold is not guaranteed to be a point on the pass-2 grid.
        # When it is not, audit the rule's own pick on this grid rather than
        # inventing a cell that was never run.
        if fitted not in obs:
            fitted = sel
        # the contrast that means something: the best cell that is NOT the fitted
        # one. Differencing the fitted cell against "the runner-up by raw F1" is
        # a no-op whenever the tie-band already moved the pick to that cell.
        rival = next(t for t in order if t != fitted)
        tie = [t for t in grid if obs[best] - obs[t] <= NOISE]

        rng = random.Random(a.seed)
        wins, d_second, d_shipped = Counter(), [], []
        for _ in range(a.boot):
            draw = [common[rng.randrange(len(common))] for _ in common]
            w, fs = select(counts, grid, draw)
            wins[w] += 1
            d_second.append(fs[fitted] - fs[rival])
        lo = sorted(d_second)[int(.025 * len(d_second))]
        hi = sorted(d_second)[int(.975 * len(d_second))]
        out[task] = {
            "n_videos": len(common), "grid": grid, "n_cells": len(grid),
            "observed_f1": {str(t): round(obs[t], 4) for t in grid},
            "fitted_thr": fitted, "best_thr": best, "rule_selects": sel,
            "rival_thr": rival,
            "gap_to_rival": round(obs[fitted] - obs[rival], 4),
            "grid_span": round(obs[order[0]] - obs[order[-1]], 4),
            "n_within_noise_of_best": len(tie),
            "frac_grid_within_noise": round(len(tie) / len(grid), 3),
            # the headline
            "reselect_rate": round(wins[fitted] / a.boot, 3),
            "chance_reselect_rate": round(1.0 / len(grid), 3),
            "n_distinct_winners": len(wins),
            "winner_hist": {str(t): wins[t] for t in sorted(wins, key=lambda x: -wins[x])},
            "delta_vs_rival_ci95": [round(lo, 4), round(hi, 4)],
            "p_delta_le_0": round(sum(1 for d in d_second if d <= 0) / len(d_second), 3),
        }
        r = out[task]
        print(f"{task:30s} thr {fitted:.3f}  reselected {r['reselect_rate']:.0%} "
              f"(chance {r['chance_reselect_rate']:.0%}, {r['n_distinct_winners']}/"
              f"{len(grid)} cells ever win)  span {r['grid_span']:.3f}  "
              f"{r['n_within_noise_of_best']}/{len(grid)} within noise  "
              f"d2nd CI [{lo:+.3f},{hi:+.3f}]")

    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
