#!/usr/bin/env python
"""pick.py -- apply the sec.3 selection rule; emit pass 2's worklist or the final thresholds.

  python lib/pick.py --pass p1     # -> P1_PICKS.json + worklist_p2.tsv
  python lib/pick.py --pass p2     # -> FINAL_THRESHOLDS.json

THE RULE (sec.3), and it is deliberately NOT a bare argmax:
  rank by the selection metric, `time_f1` as tie-break; if the top two are within
  0.02 of each other, treat them as tied and prefer the threshold whose emit/gt is
  closer to 1.0 -- the one not winning by luck of emission volume.

WHY time_f1 IS THE SELECTION METRIC HERE. sec.3 ranks by joint_f1, but the lanes
run with the judge offline (RUNBOOK sec.3), so joint_f1 is WITHHELD for every cell
and ranking by it would rank by None. time_f1 is judge-free and exact. This is
recorded as a deviation, not smuggled in: --metric joint_f1 switches back once
verdicts exist, and re-running this script is free.

sec.6.2's warning is enforced, not just quoted: at ~2.4 GT events per sample a
15-sample cell holds ~36 scoring events, and a 0.02-0.03 gap between adjacent
thresholds is NOT a real difference. So a task whose curve is flat or
non-monotonic across the whole grid is reported as FLAT and KEEPS ITS SHIPPED
VALUE rather than having a winner manufactured for it.
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "lib"))
import worklist as W                                            # noqa: E402

ROOT = os.environ["THR_ROOT"]
TIE_BAND = 0.02
SHIPPED = {"cumulative_counting": 0.925, "dedup_counting": 0.992,
           "event_narration": 0.10, "explicit_target_grounding": 0.50,
           "instant_event_alert": 0.45, "realtime_state_monitor": 0.80,
           "semantic_condition_alert": 0.98, "sequential_step_instruction": 0.01,
           "snapshot_counting": 0.985}


def load_cells(pass_):
    with open(os.path.join(ROOT, "results", pass_, "CELLS.json")) as f:
        return json.load(f)


def rank(rows, metric):
    """sec.3: rank by `metric`, time_f1 as tie-break, then the 0.02 tie-band."""
    def key(r):
        m = r.get(metric)
        return (-(m if m is not None else -1.0), -(r["time_f1"] or 0.0))
    ranked = sorted(rows, key=key)
    if len(ranked) >= 2:
        a, b = ranked[0], ranked[1]
        ma, mb = (a.get(metric) or 0.0), (b.get(metric) or 0.0)
        if abs(ma - mb) <= TIE_BAND:
            # tied: prefer the one whose emission volume is closest to the truth
            def dist(r):
                e = r.get("emit_per_gt")
                return abs((e if e is not None else 1e9) - 1.0)
            if dist(b) < dist(a):
                ranked[0], ranked[1] = b, a
    return ranked


def flat(rows, metric):
    """True when the whole grid sits inside one tie-band -- no real winner."""
    vals = [(r.get(metric) or 0.0) for r in rows]
    return (max(vals) - min(vals)) <= TIE_BAND if vals else True


def main():
    ap = argparse.ArgumentParser()
    # Not restricted to {p1,p2}: a synthetic pass name is how the selection rule
    # gets tested against peaked / flat / rail-pinned curves without waiting for
    # -- or clobbering -- a real pass. "p2" selects the finalise branch; anything
    # else takes the pass-1 branch and writes under its own name.
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--metric", default="time_f1",
                    choices=["time_f1", "joint_f1"])
    a = ap.parse_args()
    picks_name = "P1_PICKS.json" if a.pass_ == "p1" else f"{a.pass_.upper()}_PICKS.json"
    next_wl = ("worklist_p2.tsv" if a.pass_ == "p1"
               else f"worklist_{a.pass_}_next.tsv")

    cells = load_cells(a.pass_)
    by_task = {}
    for r in cells:
        by_task.setdefault(r["task"], []).append(r)

    picks, p2_grid = {}, {}
    print(f"{'task':<30}{'best':>7}{'2nd':>7}{'m1':>8}{'m2':>8}{'gap':>7}"
          f"{'e/gt':>7}  note")
    for task in W.TASKS:
        rows = [r for r in by_task.get(task, []) if r["reliable"]]
        dropped = len(by_task.get(task, [])) - len(rows)
        if not rows:
            print(f"{task:<30}{'-':>7}  no reliable cells")
            continue
        ranked = rank(rows, a.metric)
        b1, b2 = ranked[0], (ranked[1] if len(ranked) > 1 else ranked[0])
        m1 = b1.get(a.metric) or 0.0
        m2 = b2.get(a.metric) or 0.0
        is_flat = flat(rows, a.metric)
        note = []
        if is_flat:
            note.append(f"FLAT (span {max((r.get(a.metric) or 0) for r in rows) - min((r.get(a.metric) or 0) for r in rows):.3f}) -> keep shipped")
        if dropped:
            note.append(f"{dropped} unreliable cell(s) excluded")
        # RAIL CHECK (GATE_TUNING.md sec.3 rule 2): a winner at the edge of the
        # searched range means the grid truncated the search, not that the edge is
        # optimal. Flag it; do not silently accept it.
        grid = sorted(r["thr"] for r in rows)
        if b1["thr"] in (grid[0], grid[-1]):
            note.append(f"RAIL at {b1['thr']:.3f} -- widen the grid")

        picks[task] = {
            "best": b1["thr"], "second": b2["thr"],
            f"best_{a.metric}": m1, f"second_{a.metric}": m2,
            "gap": round(m1 - m2, 4),
            "best_emit_per_gt": b1.get("emit_per_gt"),
            "best_n": b1["n"], "best_n_gt": b1["n_gt"], "best_n_emit": b1["n_emit"],
            "shipped": SHIPPED[task], "flat": is_flat,
            "rail": b1["thr"] in (grid[0], grid[-1]),
            "final": SHIPPED[task] if is_flat else b1["thr"],
            "metric": a.metric,
        }
        print(f"{task:<30}{b1['thr']:>7.3f}{b2['thr']:>7.3f}{m1:>8.4f}{m2:>8.4f}"
              f"{m1 - m2:>7.4f}{(b1.get('emit_per_gt') or 0):>7.2f}  {'; '.join(note)}")

        # sec.4: 5 interior points splitting [min(b1,b2), max(b1,b2)] into 6
        lo, hi = sorted((b1["thr"], b2["thr"]))
        if hi > lo:
            p2_grid[task] = [round(lo + (hi - lo) * k / 6, 4) for k in range(1, 6)]
        else:
            p2_grid[task] = []      # best == second: nothing to refine

    if a.pass_ != "p2":
        with open(os.path.join(ROOT, picks_name), "w") as f:
            json.dump(picks, f, indent=1)
        rows = W.gen_worklist(os.path.join(ROOT, next_wl),
                              pass_="p2", per_task=p2_grid,
                              tasks=[t for t in W.TASKS if p2_grid.get(t)])
        print(f"\n{picks_name} + {next_wl} ({len(rows)} units, "
              f"{len(rows) // W.NSHARD} cells)")
    else:
        # sec.4: the winner of THR_P2 union {b1, b2}. b1/b2 are already measured,
        # so re-running them would be waste -- but excluding them could pick a
        # refined point worse than the coarse one.
        with open(os.path.join(ROOT, "P1_PICKS.json")) as f:
            p1 = json.load(f)
        final = {}
        for task in W.TASKS:
            cand = [r for r in by_task.get(task, []) if r["reliable"]]
            p1cells = [r for r in load_cells("p1")
                       if r["task"] == task and r["reliable"]
                       and r["thr"] in (p1[task]["best"], p1[task]["second"])]
            allc = cand + p1cells
            if not allc:
                final[task] = SHIPPED[task]
                continue
            if p1.get(task, {}).get("flat"):
                final[task] = SHIPPED[task]           # flat stays shipped
                continue
            final[task] = rank(allc, a.metric)[0]["thr"]
        with open(os.path.join(ROOT, "FINAL_THRESHOLDS.json"), "w") as f:
            json.dump(final, f, indent=1)
        print("\nFINAL_THRESHOLDS.json:")
        for t, v in final.items():
            d = v - SHIPPED[t]
            print(f"  {t:<30}{v:>7.4f}  (shipped {SHIPPED[t]:.3f}, delta {d:+.3f})")


if __name__ == "__main__":
    main()
