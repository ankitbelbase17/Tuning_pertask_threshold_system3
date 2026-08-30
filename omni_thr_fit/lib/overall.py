#!/usr/bin/env python
"""overall.py -- sec.6.1's three numbers, each across the three OmniPro strata.

  python lib/overall.py                       # stage 3, judge off (time-F1 only)
  python lib/overall.py --judge               # after verdicts have been ingested
  python lib/overall.py --pred_dir results/full2700 --json OVERALL.json

THE THREE NUMBERS, and why they are not interchangeable (sec.6.1):

  in-sample ceiling  the winning 5% cells, scored on the very samples that chose
                     them. Optimistically biased BY CONSTRUCTION -- it is a
                     headroom figure, and is labelled as one, exactly as
                     system_3/METHODOLOGY_FOR_PAPER.md labels grid_best.json.
  HEADLINE           stage 3, all 2,700. Comparable to OmniPro Table 2 Online.
  fit-disjoint       stage 3 rescored with the 135 fitting ids removed. Free --
                     the same predictions, filtered -- and it is the clean
                     generalization estimate, because stage 3's 2,700 CONTAIN the
                     135, making the headline ~5% in-sample.

If headline and fit-disjoint differ by more than the sec.6.2 noise band, the fit
overfitted its 15-sample cells; that comparison is the point of computing both.

STRATA. `gross` is not a summary of the other two: 65% of OmniPro is
audio_dependency=required, so gross is dominated by it. All three are printed.

SCORING IS metrics.py, CALLED, NOT REIMPLEMENTED (CLAUDE.md sec.3) -- a second
implementation of the scorer is a second set of scorer bugs.
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
sys.path.insert(0, os.path.join(ROOT, "lib"))

STRATA = {"audio_required": lambda a: a == "required",
          "audio_not_required": lambda a: a in ("none", "helpful"),
          "gross": lambda a: True}


def read_preds(pred_dir):
    """Every prediction row under pred_dir, deduplicated by sample id.

    Deduplication is not cosmetic. Resume is global but not transactional: a lane
    SIGKILLed after writing its row but before the next lane's --done_glob refresh
    can leave the same id banked twice, and a duplicated sample would be counted
    twice in the pooled tp/fp/fn. Last row wins.
    """
    rows, torn = {}, 0
    for fp in sorted(glob.glob(os.path.join(pred_dir, "lane*", "online_pred.jsonl"))):
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    torn += 1
                    continue
                rows[r["id"]] = r
    return list(rows.values()), torn


def fit_ids():
    """The 135 frozen fitting ids -- the contamination to exclude."""
    import worklist as W
    ids = set()
    for t in W.TASKS:
        ids |= W.frozen_ids(t)
    return ids


def ceiling_rows(pass_dirs, finals):
    """Prediction rows of the WINNING cell of each task -- the in-sample ceiling.

    These are the same 15 samples per task that chose the threshold, so the
    resulting F1 is optimistically biased by construction. It is reported as
    headroom, never as a result (sec.6.1).
    """
    rows = {}
    for task, thr in finals.items():
        for pd in pass_dirs:
            cell = os.path.join(ROOT, "results", pd, task, f"thr_{float(thr):.4f}")
            if not os.path.isdir(cell):
                continue
            got, _ = read_preds(cell)
            for r in got:
                rows[r["id"]] = r
    return list(rows.values())


def table(rows, judge, tolerance):
    """{stratum: aggregate(...)} for one set of prediction rows."""
    from metrics import aggregate, score_sample
    out = {}
    for name, keep in STRATA.items():
        sub = [r for r in rows if keep(r.get("audio_dependency", "none"))]
        if not sub:
            out[name] = None
            continue
        out[name] = aggregate([score_sample(r, tolerance=tolerance, judge=judge)
                               for r in sub])
    return out


def show(title, tab, note=""):
    print(f"\n=== {title} ===" + (f"   [{note}]" if note else ""))
    print(f"{'stratum':<22}{'n':>6}{'n_gt':>7}{'n_emit':>8}"
          f"{'tP':>8}{'tR':>8}{'timeF1':>8}{'macroT':>8}"
          f"{'cAcc':>8}{'jointF1':>9}{'macroJ':>8}")
    for name in STRATA:
        a = tab.get(name)
        if a is None:
            print(f"{name:<22}{'-- no samples in this stratum --':>40}")
            continue
        o = a["overall"]
        def f(v):
            return "WITHHELD" if v is None else f"{v:.4f}"
        print(f"{name:<22}{o['n_samples']:>6}{o['n_gt']:>7}{o['n_emits']:>8}"
              f"{o['time_precision']:>8.4f}{o['time_recall']:>8.4f}"
              f"{o['time_f1']:>8.4f}{o['macro_time_f1']:>8.4f}"
              f"{f(o['content_acc']):>8}{f(o['joint_f1']):>9}{f(o['macro_joint_f1']):>8}")
    g = tab.get("gross")
    if g and g["overall"]["n_unjudged"]:
        o = g["overall"]
        print(f"  content WITHHELD: {o['n_unjudged']}/{o['n_matched']} matches "
              f"unjudged (coverage {o['content_coverage']:.2f}); "
              f"lower bounds joint_f1_lb={o['joint_f1_lb']:.4f} "
              f"content_acc_lb={o['content_acc_lb']:.4f}")


def per_task_block(tab):
    a = tab.get("gross")
    if not a:
        return
    print(f"\n  {'task':<30}{'n':>5}{'n_gt':>6}{'n_emit':>8}{'timeF1':>9}{'jointF1':>9}")
    for t, b in a["per_task"].items():
        j = "WITHHELD" if b["joint_f1"] is None else f"{b['joint_f1']:.4f}"
        print(f"  {t:<30}{b['n_samples']:>5}{b['n_gt']:>6}{b['n_emits']:>8}"
              f"{b['time_f1']:>9.4f}{j:>9}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=os.path.join(ROOT, "results", "full2700"))
    ap.add_argument("--tolerance", type=float, default=3.0)
    ap.add_argument("--judge", action="store_true",
                    help="enable the LLM judge; without it content/joint are WITHHELD")
    ap.add_argument("--json", default=os.path.join(ROOT, "OVERALL.json"))
    ap.add_argument("--per_task", action="store_true")
    ap.add_argument("--ceiling", action="store_true",
                    help="also compute the in-sample ceiling from the winning cells")
    a = ap.parse_args()

    # Judge OFF unless asked. env.sh exports the API keys, so an unguarded run
    # would fire a judge request per matched emit -- thousands, against an
    # endpoint README records as 404/429/503 -- and poison the shared verdict
    # cache namespace on the way (CLAUDE.md sec.3).
    if not a.judge:
        for k in ("GEMINI_API_KEY", "OPENAI_API_KEY",
                  "GEMINI_API_BASE", "OPENAI_API_BASE"):
            os.environ.pop(k, None)
    from metrics import ContentJudge
    judge = ContentJudge()

    rows, torn = read_preds(a.pred_dir)
    if not rows:
        sys.exit(f"no predictions under {a.pred_dir}")
    print(f"{len(rows)} unique samples from {a.pred_dir}"
          + (f" ({torn} torn lines skipped)" if torn else ""))
    print(f"judge: {'ON -> ' + str(judge.mode) if a.judge else 'OFF (content WITHHELD)'}")

    fits = fit_ids()
    disjoint = [r for r in rows if r["id"] not in fits]

    out = {}
    out["full"] = table(rows, judge, a.tolerance)
    show("HEADLINE -- full OmniPro", out["full"],
         f"{len(rows)} samples; contains {len(rows) - len(disjoint)} fitting ids")
    if a.per_task:
        per_task_block(out["full"])

    out["fit_disjoint"] = table(disjoint, judge, a.tolerance)
    show("fit-disjoint", out["fit_disjoint"],
         f"{len(disjoint)} samples, the {len(rows) - len(disjoint)} fitting ids removed")

    # The delta that sec.6.1 asks for, stated rather than left to the reader.
    for s in STRATA:
        f_, d_ = out["full"].get(s), out["fit_disjoint"].get(s)
        if f_ and d_:
            df = f_["overall"]["time_f1"] - d_["overall"]["time_f1"]
            flag = "  <-- exceeds the sec.6.2 noise band" if abs(df) > 0.03 else ""
            print(f"  time-F1 in-sample bias, {s}: {df:+.4f}{flag}")

    # in-sample ceiling: only meaningful once a winner exists per task.
    fin = os.path.join(ROOT, "FINAL_THRESHOLDS.json")
    if a.ceiling and os.path.exists(fin):
        with open(fin) as f:
            finals = json.load(f)
        crows = ceiling_rows(["p2", "p1"], finals)
        if crows:
            out["in_sample_ceiling"] = table(crows, judge, a.tolerance)
            show("in-sample ceiling (OPTIMISTICALLY BIASED -- headroom, not a result)",
                 out["in_sample_ceiling"],
                 f"{len(crows)} samples, scored on the cells that chose the thresholds")
        else:
            print("\n(no winning-cell predictions found for the ceiling)")
    elif a.ceiling:
        print("\n(FINAL_THRESHOLDS.json absent -- skipping the in-sample ceiling)")

    with open(a.json, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
