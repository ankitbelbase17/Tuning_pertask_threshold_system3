#!/usr/bin/env python
"""score_cells.py -- one row per (pass, task, threshold) cell.

Scoring is metrics.score_sample / metrics.aggregate VERBATIM (spec sec.8: "Do not
reimplement the scorer; a second implementation is a second set of bugs"). This
module only groups predictions into cells and lays the aggregate out as a table.

joint_f1 / content_acc are computed but will read as WITHHELD while the judge is
offline: the lanes run with the judge keys unset, so `_content_correct` returns
None for the two free-text tasks and those matches land in `n_unjudged` rather
than being guessed. Selection runs on time_f1, which never touches a judge.

  python lib/score_cells.py                 # -> results/p1/CELLS.{json,csv}
  python lib/score_cells.py --pass p2
"""
from __future__ import annotations
import argparse, csv, json, os, sys, glob

sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "lib"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "omniprofast"))
sys.path.insert(0, os.path.join(os.environ["REPO"], "async_omni_v2"))
import worklist as W                                            # noqa: E402
from metrics import ContentJudge, aggregate, score_sample       # noqa: E402

FIELDS = ["pass", "task", "thr", "n", "n_gt", "n_emit", "emit_per_gt",
          "time_p", "time_r", "time_f1", "content_acc", "joint_f1",
          "content_acc_lb", "joint_f1_lb", "n_unjudged", "xrt_mean", "reliable"]


def cell_predictions(cell):
    # a resumed lane can re-bank an id; keep the LAST occurrence, which is the run
    # the on-disk predictions actually reflect.
    dedup = {}
    for fp in sorted(glob.glob(os.path.join(cell, "lane*", "online_pred.jsonl"))):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue       # torn line from a SIGKILLed lane
                    if r.get("id"):
                        dedup[r["id"]] = r
        except OSError:
            pass                       # /iopsstor flaps; a missing lane is not zero
    return list(dedup.values())


def score_cell(pass_, task, thr, judge, tol=3.0):
    cell = W.cell_dir(pass_, task, thr)
    preds = cell_predictions(cell)
    want = W.frozen_ids(task)
    if not preds:
        return None
    per = [score_sample(p, tolerance=tol, judge=judge) for p in preds]
    # a cell is ONE task, so aggregate's pooled "overall" block is that task's
    # block; per_task would be a dict of one and buys nothing here.
    g = aggregate(per)["overall"]
    xs = [p["realtime_factor"] for p in preds if p.get("realtime_factor")]
    n_emit, n_gt = g["n_emits"], g["n_gt"]
    return {
        "pass": pass_, "task": task, "thr": round(thr, 4),
        "n": len(preds), "n_gt": n_gt, "n_emit": n_emit,
        "emit_per_gt": round(n_emit / n_gt, 4) if n_gt else None,
        "time_p": g["time_precision"], "time_r": g["time_recall"],
        "time_f1": g["time_f1"],
        # None while the judge is offline -- WITHHELD, never guessed. The _lb
        # columns are the lower bounds aggregate() publishes under a name that
        # says what they are, and they are NOT a substitute for the real thing.
        "content_acc": g["content_acc"], "joint_f1": g["joint_f1"],
        "content_acc_lb": g["content_acc_lb"], "joint_f1_lb": g["joint_f1_lb"],
        "n_unjudged": g["n_unjudged"],
        "xrt_mean": round(sum(xs) / len(xs), 3) if xs else None,
        # A cell scored on fewer samples than it was supposed to have is NOT
        # comparable to a complete one. sec.9.4 requires such panels be hatched;
        # this is the flag that drives it.
        "reliable": len(preds) >= len(want),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--worklist", default=None)
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--judge", action="store_true",
                    help="allow live LLM judging (default OFF, see below)")
    a = ap.parse_args()
    wl = a.worklist or os.path.join(os.environ["THR_ROOT"],
                                    f"worklist_{a.pass_}.tsv")

    # JUDGE OFF BY DEFAULT, and it must be default-off rather than opt-out. env.sh
    # exports the API keys, so ContentJudge would otherwise come up live and try
    # Gemini for EVERY free-text emit across 94 cells -- against an endpoint that
    # 404s on this key. That is thousands of doomed round-trips, and any verdict it
    # did return would be mixed into the same cache namespace as a later judge's
    # (metrics.py:353-360), which is exactly the contamination sec.5 forbids.
    # Selection runs on time_f1, which never touches a judge.
    if not a.judge:
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY",
                  "GEMINI_API_BASE", "OPENAI_API_BASE"):
            os.environ.pop(k, None)
    judge = ContentJudge()
    print(f"[score] judge mode = {getattr(judge, 'mode', '?')}")
    cells = sorted({(p, t, thr) for p, t, thr, _ in W.read_worklist(wl)},
                   key=lambda c: (c[1], c[2]))
    out = []
    for (p, t, thr) in cells:
        row = score_cell(p, t, thr, judge, a.tolerance)
        if row:
            out.append(row)

    d = os.path.join(os.environ["THR_ROOT"], "results", a.pass_)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "CELLS.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(d, "CELLS.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    hdr = f"{'task':<30}{'thr':>7}{'n':>4}{'n_gt':>6}{'n_emit':>8}{'e/gt':>7}" \
          f"{'time_f1':>9}{'joint_f1':>10}{'xRT':>7}  ok"
    print(hdr); print("-" * len(hdr))
    for r in out:
        jf = "WITHHELD" if r["joint_f1"] is None else f"{r['joint_f1']:.4f}"
        print(f"{r['task']:<30}{r['thr']:>7.3f}{r['n']:>4}{r['n_gt']:>6}"
              f"{r['n_emit']:>8}{(r['emit_per_gt'] or 0):>7.2f}"
              f"{r['time_f1']:>9.4f}{jf:>10}"
              f"{(r['xrt_mean'] or 0):>7.2f}  {'y' if r['reliable'] else 'N'}")
    print(f"\n{len(out)}/{len(cells)} cells scored -> {d}/CELLS.json")


if __name__ == "__main__":
    main()
