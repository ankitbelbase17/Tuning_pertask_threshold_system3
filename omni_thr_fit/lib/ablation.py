#!/usr/bin/env python
"""ablation.py -- Table 4: does per-task fitting beat a single global threshold?

  python lib/ablation.py --pass p1                  # screen, on the fit subset
  python lib/ablation.py --pass p2 --picks FINAL_THRESHOLDS.json

FOUR ARMS (spec sec.9.2 Table 4):
  global_0.5          every task at 0.5 -- the shipped gate's EFFECTIVE behaviour,
                      since `gate_strategy="hysteresis"` tested p_hit >= a fixed
                      global 0.5 and ignored `hit_threshold` entirely (RUNBOOK 2).
  best_single_global  the ONE threshold, swept over the grid all tasks share, that
                      maximises pooled time_f1. This is the arm per-task fitting
                      has to beat to justify its own existence.
  shipped_per_task    config.py's `task_hit_thresholds`.
  fitted_per_task     this study's picks.

WHY THIS REPORTS DIFFERENCES WITH CONFIDENCE INTERVALS AND NOT FOUR BARE NUMBERS.
RUNBOOK sec.2.6 measured that no task's fitted threshold beats its own best rival
outside noise. A Table 4 of four point estimates would invite exactly the reading
that finding rules out -- "fitted is highest, therefore fitting works". The arms
run on the SAME videos, so the difference between two arms can be bootstrapped
PAIRED, which is both sharper and the only form that answers the question asked.
An arm's advantage is real only if its interval against `best_single_global`
excludes zero.

OFF-GRID ARMS ARE FLAGGED, NEVER INTERPOLATED. `shipped_per_task` uses values like
0.992 and 0.925 that no cell was ever run at, and 0.5 itself is not a grid point
(the grid steps 0.45 -> 0.55). Such an arm is served by the NEAREST cell actually
on disk and every substitution is recorded in `off_grid` with its distance, so a
reader can see which rows are approximations. Reporting a nearest-cell value as if
it were the real one is the failure mode this guards against.

Scoring is metrics.score_sample / _prf VERBATIM (spec sec.8); this module only
groups, pools and resamples.
"""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import defaultdict

# no judge: pure timing. See bin/audit_fit_noise.py for why this must precede the
# import rather than setting ContentJudge.offline afterwards.
for _k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "GEMINI_API_BASE", "OPENAI_API_BASE"):
    os.environ.pop(_k, None)

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "omniprofast"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "async_omni_v2"))

import worklist as W                                                # noqa: E402
from score_cells import cell_predictions                            # noqa: E402
from metrics import ContentJudge, score_sample, _prf                # noqa: E402

SHIPPED = {"cumulative_counting": 0.925, "dedup_counting": 0.992,
           "event_narration": 0.10, "explicit_target_grounding": 0.50,
           "instant_event_alert": 0.45, "realtime_state_monitor": 0.80,
           "semantic_condition_alert": 0.98, "sequential_step_instruction": 0.01,
           "snapshot_counting": 0.985}


def load(pass_, judge):
    """task -> thr -> id -> (tp, fp, fn). Only cells that exist on disk."""
    units = W.read_worklist(os.path.join(ROOT, f"worklist_{pass_}.tsv"))
    grids = defaultdict(set)
    for _p, t, thr, _sh in units:
        grids[t].add(round(float(thr), 4))
    data = {}
    for task, grid in grids.items():
        cells = {}
        for thr in sorted(grid):
            preds = cell_predictions(W.cell_dir(pass_, task, thr))
            if not preds:
                continue
            cells[thr] = {s["id"]: (s["tp_time"], s["fp"], s["fn"])
                          for s in (score_sample(p, tolerance=3.0, judge=judge)
                                    for p in preds)}
        if cells:
            data[task] = cells
    return data


def nearest(grid, want):
    return min(grid, key=lambda t: (abs(t - want), t))


def arm_map(data, kind, picks=None):
    """task -> (threshold_used, requested, is_off_grid)."""
    out = {}
    for task, cells in data.items():
        grid = sorted(cells)
        if kind == "global_0.5":
            want = 0.5
        elif kind == "shipped_per_task":
            want = SHIPPED[task]
        elif kind == "fitted_per_task":
            rec = (picks or {}).get(task, {})
            # `final` is what pick.py would actually ship: it falls back to the
            # shipped value on a FLAT task rather than inventing a winner.
            want = float(rec.get("final", rec.get("best", SHIPPED[task])))
        else:
            raise ValueError(kind)
        got = nearest(grid, want)
        out[task] = (got, want, abs(got - want) > 1e-9)
    return out


def pooled_f1(data, amap, ids_by_task):
    """Micro-average tp/fp/fn across every task, matching metrics.aggregate's
    pooled `overall` block -- NOT the mean of per-task F1s."""
    tp = fp = fn = 0
    for task, (thr, _w, _o) in amap.items():
        cell = data[task][thr]
        for i in ids_by_task[task]:
            c = cell.get(i)
            if c:
                tp += c[0]; fp += c[1]; fn += c[2]
    return _prf(tp, fp, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--picks", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default=os.path.join(ROOT, "ABLATION.json"))
    a = ap.parse_args()

    judge = ContentJudge()
    judge.offline = True
    pf = a.picks or os.path.join(
        ROOT, "P1_PICKS.json" if a.pass_ == "p1" else "FINAL_THRESHOLDS.json")
    picks = json.load(open(pf)) if os.path.exists(pf) else {}

    data = load(a.pass_, judge)
    if not data:
        sys.exit("no scored cells")
    # the id set every cell of a task shares, so all arms compare on identical videos
    ids = {t: sorted(set.intersection(*(set(c) for c in cells.values())))
           for t, cells in data.items()}

    arms = {k: arm_map(data, k, picks)
            for k in ("global_0.5", "shipped_per_task", "fitted_per_task")}

    # best_single_global: sweep the grid the tasks have in COMMON. A threshold only
    # some tasks were run at is not a global setting, it is a partial one.
    common = sorted(set.intersection(*(set(c) for c in data.values())))
    sweep = {}
    for thr in common:
        m = {t: (thr, thr, False) for t in data}
        sweep[thr] = pooled_f1(data, m, ids)[2]
    bsg = max(common, key=lambda t: sweep[t]) if common else None
    if bsg is not None:
        arms["best_single_global"] = {t: (bsg, bsg, False) for t in data}

    res = {"pass": a.pass_, "n_tasks": len(data),
           "n_videos_total": sum(len(v) for v in ids.values()),
           "common_grid": common, "single_global_sweep": {str(k): round(v, 4)
                                                          for k, v in sweep.items()},
           "best_single_global_thr": bsg, "arms": {}}
    for name, m in arms.items():
        p, r, f = pooled_f1(data, m, ids)
        res["arms"][name] = {
            "time_p": round(p, 4), "time_r": round(r, 4), "time_f1": round(f, 4),
            "thresholds": {t: v[0] for t, v in m.items()},
            "off_grid": {t: {"requested": v[1], "used": v[0],
                             "abs_error": round(abs(v[0] - v[1]), 4)}
                         for t, v in m.items() if v[2]},
        }

    # paired bootstrap of every arm against best_single_global
    if bsg is not None:
        rng = random.Random(a.seed)
        base = arms["best_single_global"]
        diffs = defaultdict(list)
        # best_single_global is itself an argmax taken over this same grid on these
        # same videos, so it carries the identical winner's curse sec.2.6 measured
        # for the per-task fit. Reporting "a global threshold beats 0.5 by +0.02"
        # without checking whether that argmax reproduces would be the same error
        # one level up. So the sweep is re-run inside every draw and the winner
        # recorded: if 0.15 is a noise spike on a flat plateau, it will not hold.
        bsg_wins = defaultdict(int)
        for _ in range(a.boot):
            draw = {t: [v[rng.randrange(len(v))] for _ in v] for t, v in ids.items()}
            b = pooled_f1(data, base, draw)[2]
            for name, m in arms.items():
                if name != "best_single_global":
                    diffs[name].append(pooled_f1(data, m, draw)[2] - b)
            w = max(common, key=lambda t: pooled_f1(
                data, {tt: (t, t, False) for tt in data}, draw)[2])
            bsg_wins[w] += 1
        res["bsg_reselect_rate"] = round(bsg_wins[bsg] / a.boot, 3)
        res["bsg_chance_rate"] = round(1.0 / len(common), 3)
        res["bsg_winner_hist"] = {str(k): v for k, v in
                                  sorted(bsg_wins.items(), key=lambda kv: -kv[1])}
        for name, d in diffs.items():
            d.sort()
            res["arms"][name]["delta_vs_best_single_global"] = {
                "point": round(res["arms"][name]["time_f1"]
                               - res["arms"]["best_single_global"]["time_f1"], 4),
                "ci95": [round(d[int(.025 * len(d))], 4),
                         round(d[int(.975 * len(d))], 4)],
                "p_le_0": round(sum(1 for x in d if x <= 0) / len(d), 3),
            }

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"pass {a.pass_}  {len(data)} tasks  "
          f"{sum(len(v) for v in ids.values())} videos  "
          f"best_single_global thr={bsg}")
    print(f"{'arm':22s} {'time_P':>7s} {'time_R':>7s} {'time_F1':>8s} "
          f"{'d vs BSG':>10s} {'95% CI':>18s}  off-grid")
    for name in ("global_0.5", "best_single_global", "shipped_per_task",
                 "fitted_per_task"):
        if name not in res["arms"]:
            continue
        v = res["arms"][name]
        d = v.get("delta_vs_best_single_global")
        ds = f"{d['point']:+.4f}" if d else "     --"
        ci = f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]" if d else ""
        print(f"{name:22s} {v['time_p']:7.4f} {v['time_r']:7.4f} "
              f"{v['time_f1']:8.4f} {ds:>10s} {ci:>18s}  {len(v['off_grid'])}/{len(data)}")
    if "bsg_reselect_rate" in res:
        print(f"\nbest_single_global={bsg} re-selected {res['bsg_reselect_rate']:.0%} "
              f"of draws (chance {res['bsg_chance_rate']:.0%}); winners: " +
              "  ".join(f"{k}:{v}" for k, v in list(res["bsg_winner_hist"].items())[:5]))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
