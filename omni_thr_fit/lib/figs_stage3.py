#!/usr/bin/env python
"""figs_stage3.py -- Figures 6-8: the stage-3 result, drawn from banked predictions.

  python lib/figs_stage3.py --out figs/

  Fig 6  per-task time-F1 on the full benchmark, with the emission ratio that
         explains it (n_emit / n_gt on a second axis).
  Fig 7  paired-bootstrap forest: every task's dF1 against its best rival, pass 1
         and pass 2, with the zero line. This is finding 2/4 in one picture.
  Fig 8  head-to-head against the vision-only parent, per task.

WHY THESE THREE. Figures 1-5 describe the FIT (grid curves, refinement). Nothing
described the OUTCOME per task, and the outcome is where the two failure regimes
separate: six tasks over-emit 3-5x, three emit once per video. A reader cannot see
that in an F1 bar alone, which is why Fig 6 draws the emission ratio beside it.

Scoring is metrics.aggregate, called, not reimplemented (CLAUDE.md sec.3).
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.environ.get(
    "THR_ROOT", "/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit")
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import figstyle as S                                              # noqa: E402

SHORT = {"cumulative_counting": "cumul", "dedup_counting": "dedup",
         "event_narration": "narrate", "explicit_target_grounding": "ground",
         "instant_event_alert": "alert", "realtime_state_monitor": "state",
         "semantic_condition_alert": "cond", "sequential_step_instruction": "step",
         "snapshot_counting": "snap"}


def stage3_per_task(pred_dir):
    """(task -> aggregate block) over every banked stage-3 prediction."""
    from metrics import aggregate, score_sample
    rows = {}
    for fp in sorted(glob.glob(os.path.join(pred_dir, "lane*", "online_pred.jsonl"))):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue                      # torn line from a SIGKILLed lane
            if r.get("id"):
                rows[r["id"]] = r             # dedup: last wins, as score_cells does
    per = [score_sample(p, tolerance=3.0, judge=None) for p in rows.values()]
    return aggregate(per)


def fig6(out, pt):
    tasks = sorted(pt, key=lambda t: -pt[t]["time_f1"])
    f1 = [pt[t]["time_f1"] for t in tasks]
    ratio = [pt[t]["n_emits"] / max(pt[t]["n_gt"], 1) for t in tasks]
    fig, ax = plt.subplots(figsize=(S.COL_FULL, 2.7))
    x = range(len(tasks))
    # zorder above the grid: the default draws gridlines THROUGH the bars,
    # which reads as banding on a solid fill.
    ax.bar(x, f1, color=S.VIOLET, width=0.62, label="time-F1", zorder=3)
    ax.set_axisbelow(True)
    ax.set_ylabel("time-F1"); ax.set_ylim(0, 0.45)
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[t] for t in tasks])
    # The emission ratio is the explanation, so it shares the panel rather than
    # sitting in a second figure the reader has to hold in their head.
    ax2 = ax.twinx()
    ax2.plot(x, ratio, "o-", color=S.ORANGE, lw=1.4, ms=4, label="emits per GT event")
    ax2.axhline(1.0, color=S.MUTED, lw=0.9, ls=":")
    ax2.set_ylabel("emits per GT event"); ax2.set_ylim(0, 6.5)
    ax.set_title("Fig 6  Stage 3, per task: score and the emission ratio behind it",
                 loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
    S.tidy(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig6_stage3_per_task.pdf"))
    fig.savefig(os.path.join(out, "fig6_stage3_per_task.png"), dpi=200)
    plt.close(fig)
    return "fig6_stage3_per_task.pdf"


def fig7(out):
    p1 = json.load(open(os.path.join(ROOT, "FIT_NOISE_AUDIT.json")))
    p2 = json.load(open(os.path.join(ROOT, "FIT_NOISE_AUDIT_P2.json")))
    tasks = sorted(set(p1) & set(p2))
    fig, ax = plt.subplots(figsize=(S.COL_FULL, 3.1))
    for i, t in enumerate(tasks):
        for d, colour, off, lab in ((p1, S.VIOLET, -0.16, "pass 1"),
                                    (p2, S.BLUE, 0.16, "pass 2")):
            ci = d[t].get("delta_vs_rival_ci95")
            gap = d[t].get("gap_to_rival")
            if ci is None or gap is None:
                continue
            y = i + off
            ax.plot(ci, [y, y], color=colour, lw=1.6,
                    label=lab if i == 0 else None)
            ax.plot([gap], [y], "o", color=colour, ms=4)
    ax.axvline(0, color=S.INK, lw=1.0)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([SHORT[t] for t in tasks])
    ax.set_xlabel(r"$\Delta$ time-F1 of the fitted threshold vs its best rival"
                  "  (95% paired bootstrap CI)")
    ax.set_title("Fig 7  Every interval crosses zero, in both passes", loc="left")
    ax.legend(loc="lower right", frameon=False)
    S.tidy(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig7_bootstrap_forest.pdf"))
    fig.savefig(os.path.join(out, "fig7_bootstrap_forest.png"), dpi=200)
    plt.close(fig)
    return "fig7_bootstrap_forest.pdf"


def fig8(out, pt, parent_json):
    """Head-to-head. Drawn ONLY if the parent's table is readable -- it lives in
    another account's scratch, and a figure invented from a remembered number is
    exactly what this project's honesty rules exist to prevent."""
    if not os.path.exists(parent_json):
        return None
    par = json.load(open(parent_json))["per_task"]
    tasks = sorted(set(pt) & set(par), key=lambda t: -pt[t]["time_f1"])
    fig, ax = plt.subplots(figsize=(S.COL_FULL, 2.7))
    x = range(len(tasks))
    w = 0.38
    ax.set_axisbelow(True)
    ax.bar([i - w / 2 for i in x], [par[t]["time_f1"] for t in tasks],
           width=w, color=S.MUTED, label="system_3  Qwen3-VL-8B (vision)", zorder=3)
    ax.bar([i + w / 2 for i in x], [pt[t]["time_f1"] for t in tasks],
           width=w, color=S.VIOLET, label="omni  Qwen2.5-Omni-7B (A+V)", zorder=3)
    # n is uneven on the parent side and that is the table's main weakness, so it
    # is printed on the figure rather than left to the caption.
    for i, t in enumerate(tasks):
        ax.text(i - w / 2, par[t]["time_f1"] + 0.008, f"n={par[t]['n_samples']}",
                ha="center", fontsize=5, color=S.MUTED)
    ax.set_xticks(list(x)); ax.set_xticklabels([SHORT[t] for t in tasks])
    ax.set_ylabel("time-F1"); ax.set_ylim(0, 0.46)
    ax.set_title("Fig 8  Head to head (time-F1 only; different subsets, "
                 "see caption)", loc="left")
    ax.legend(loc="upper right", frameon=False)
    S.tidy(ax)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig8_head_to_head.pdf"))
    fig.savefig(os.path.join(out, "fig8_head_to_head.png"), dpi=200)
    plt.close(fig)
    return "fig8_head_to_head.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--pred_dir", default=os.path.join(ROOT, "results", "full2700"))
    ap.add_argument("--parent", default="/iopsstor/scratch/cscs/dbartaula/system_3/"
                                        "omniprofast/output_full9/fitted_full_table.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S.use()
    agg = stage3_per_task(a.pred_dir)
    pt = agg["per_task"]
    print("  n_samples =", agg["overall"]["n_samples"],
          " gross time-F1 =", round(agg["overall"]["time_f1"], 4))
    for f in (fig6(a.out, pt), fig7(a.out), fig8(a.out, pt, a.parent)):
        print("  ->", f if f else "(skipped: parent table not readable)")
    json.dump({"overall": agg["overall"], "per_task": pt},
              open(os.path.join(ROOT, "STAGE3_PARTIAL.json"), "w"), indent=1)
    print("  wrote STAGE3_PARTIAL.json (partial -- NOT the final OVERALL.json)")


if __name__ == "__main__":
    main()
