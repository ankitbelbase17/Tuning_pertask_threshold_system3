#!/usr/bin/env python
"""audit_refractory.py -- what does holding the vision-only refractory cost?

THE QUESTION. This experiment fits ONE of the gate's three knobs (threshold);
`mode` and `refractory_s` keep the values fitted for the VISION-ONLY Qwen3-VL
backbone. RUNBOOK sec.2 records that as a limitation. Pass 1's early cells make it
concrete: `instant_event_alert` carries a 600 s refractory, so a >=233 s video
gets exactly ONE emission, and time-F1 is 0 at every threshold in the grid --
because the single shot is spent before the event happens, not because the
threshold is wrong.

This bounds the cost offline and for free, on the finished 2,700-sample phase-A
tick stream. It is a SCREEN, not a verdict (GATE_TUNING.md): the replay cannot
model the live `bool(answer)` requirement and so over-counts emissions. Read the
COLUMN COMPARISON, never the absolute numbers.
"""
from __future__ import annotations
import os, sys, json, argparse

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
sys.path.insert(0, os.path.join(ROOT, "repo", "async_omni_v2"))
sys.path.insert(0, os.path.join(ROOT, "bin"))
from resweep import greedy_match, TOL                            # noqa: E402
from audit_first_tick import parse_logs, simulate, GRID           # noqa: E402
from config import AsyncOmniConfig as C                           # noqa: E402

REFRACS = [0.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0]


def f1(ss, thr, mode, refr):
    tp = fp = fn = 0
    for s in ss:
        e = simulate(s["ticks"], thr, mode, refr, rule="edge1")
        a, b, c = greedy_match(e, s["gt"], TOL)
        tp += a; fp += b; fn += c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), tp + fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/iopsstor/scratch/cscs/dbartaula/omni_s3_eval/"
                                     "results/phaseA_omni_full/lane*.log")
    a = ap.parse_args()
    cfg = C()
    samples = parse_logs(a.run)
    by = {}
    for s in samples.values():
        by.setdefault(s["task"], []).append(s)
    print(f"parsed {len(samples)} samples\n")
    print("Best time-F1 over the 10-point threshold grid, per refractory.")
    print("'shipped' = the refractory currently in force (vision-only fit).\n")
    hdr = f"  {'task':<28}{'ship':>6}" + "".join(f"{r:>7.0f}" for r in REFRACS) + f"{'best':>8}{'gain':>8}"
    print(hdr)
    out = {}
    for t in sorted(by):
        ss = by[t]
        mode = cfg.task_gate_modes.get(t, "edge")
        ship = cfg.task_refractory_s.get(t, 0.0)
        row, best, bestr = [], -1.0, None
        for r in REFRACS:
            v = max(f1(ss, thr, mode, r)[0] for thr in GRID)
            row.append(v)
            if v > best:
                best, bestr = v, r
        cur = max(f1(ss, thr, mode, ship)[0] for thr in GRID)
        out[t] = {"shipped_refractory": ship, "f1_at_shipped": round(cur, 4),
                  "best_refractory": bestr, "f1_at_best": round(best, 4),
                  "gain": round(best - cur, 4),
                  "curve": {str(r): round(v, 4) for r, v in zip(REFRACS, row)}}
        star = " <--" if best - cur > 0.03 else ""
        print(f"  {t:<28}{ship:>6.0f}" + "".join(f"{v:>7.3f}" for v in row)
              + f"{bestr:>8.0f}{best - cur:>+8.3f}{star}")
    with open(os.path.join(ROOT, "REFRACTORY_AUDIT.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {ROOT}/REFRACTORY_AUDIT.json")
    print("'<--' marks a gain beyond the sec.6.2 noise band (0.03).")


if __name__ == "__main__":
    main()
