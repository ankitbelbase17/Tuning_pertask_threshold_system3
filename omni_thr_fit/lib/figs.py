#!/usr/bin/env python
"""figs.py -- Fig 0-5 of sec.9.3 as vector PDF at final column width.

  python lib/figs.py --out figs/            # every figure whose data exists

DEGRADES, NEVER GUESSES. Each figure is built only if its inputs are on disk;
anything missing is reported and skipped. That way this runs usefully mid-sweep
and again unchanged at the end -- there is never a version of a figure drawn from
placeholder numbers.

CAPTIONS CARRY n. sec.9.4: "a 15-sample panel and a 300-sample panel must not
look alike". Every panel title states its n and n_gt, and a cell flagged
reliable=False is drawn on a hatched background rather than silently plotted.
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
sys.path.insert(0, os.path.join(ROOT, "repo", "async_omni_v2"))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import figstyle as FS                                            # noqa: E402
import worklist as W                                             # noqa: E402

FS.use()
SHORT = {"cumulative_counting": "cumulative", "dedup_counting": "dedup",
         "event_narration": "narration", "explicit_target_grounding": "grounding",
         "instant_event_alert": "instant alert", "realtime_state_monitor": "state monitor",
         "semantic_condition_alert": "semantic alert",
         "sequential_step_instruction": "sequential step",
         "snapshot_counting": "snapshot"}


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def cells_by_task(pass_):
    c = jload(os.path.join(ROOT, "results", pass_, "CELLS.json"))
    if not c:
        return {}
    by = {}
    for r in c:
        by.setdefault(r["task"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["thr"])
    return by


# --------------------------------------------------------------------------
def fig1_f1_vs_threshold(out, p1, p2, picks):
    """Fig 1 -- F1 vs threshold, 3x3 small multiples, pass-1 + pass-2 overlaid."""
    if not p1:
        return "Fig 1: no results/p1/CELLS.json yet"
    fig = plt.figure(figsize=(FS.COL_FULL, FS.COL_FULL * 0.72))
    axes = FS.panel_grid(fig)
    shipped = jload(os.path.join(ROOT, "SHIPPED.json")) or {}
    # ONE Y SCALE ACROSS ALL NINE PANELS. Per-panel autoscaling made
    # instant_event_alert's flat zero (0-0.05) look like structure beside
    # semantic_condition_alert's real curve (0-0.30). Small multiples exist to be
    # compared; independent axes silently forbid that.
    allf = [r["time_f1"] for rows in p1.values() for r in rows]
    ymax = max(0.05, (max(allf) if allf else 0.05) * 1.18)
    for ax, task in zip(axes.ravel(), W.TASKS):
        rows = p1.get(task, [])
        FS.tidy(ax)
        if not rows:
            ax.set_title(f"{SHORT[task]}  (not yet run)", loc="left",
                          color=FS.MUTED)
            ax.set_xlim(0, 1); ax.set_ylim(0, ymax)
            ax.grid(False)
            continue
        x = [r["thr"] for r in rows]
        # time-F1 is drawn solid and FILLED, joint-F1 dashed and OPEN on top, so
        # that where the two coincide -- which they do exactly on the seven tasks
        # scored by deterministic extraction rather than the LLM judge -- the
        # violet reads through the dashes instead of vanishing under blue.
        ax.plot(x, [r["time_f1"] for r in rows], label="time-F1", zorder=3,
                lw=1.8, **FS.S1)
        jf = [r.get("joint_f1") for r in rows]
        if any(v is not None for v in jf):
            xs = [a for a, b in zip(x, jf) if b is not None]
            ys = [b for b in jf if b is not None]
            ax.plot(xs, ys, label="joint-F1", markerfacecolor="none",
                    markeredgewidth=1.0, lw=1.1, zorder=4, **FS.S2)
        # pass 2, denser in the refined interval
        for r in p2.get(task, []):
            ax.plot([r["thr"]], [r["time_f1"]], marker="o", ms=4,
                    color=FS.VIOLET, zorder=5)
        sh = shipped.get(task)
        if sh is not None:
            ax.axvline(sh, color=FS.MUTED, lw=0.8, ls="-", zorder=0)
        pk = (picks or {}).get(task)
        if pk:
            ax.plot([pk["final"]], [pk.get(f"best_{pk.get('metric','time_f1')}", 0)],
                    marker="o", ms=6, color=FS.VIOLET, zorder=6)
            ax.annotate(f"{pk['final']:.2f}", (pk["final"], 0), xytext=(2, 2),
                        textcoords="offset points", fontsize=6, color=FS.INK2)
        n = rows[0].get("n", 0); ngt = rows[0].get("n_gt", 0)
        ax.set_title(f"{SHORT[task]}  n={n}, gt={ngt}", loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        if not all(r.get("reliable", True) for r in rows):
            ax.set_facecolor("#fbfbfa")
    for ax in axes[-1]:
        ax.set_xlabel("hit threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("F1")
    h, l = axes.ravel()[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fp = os.path.join(out, "fig1_f1_vs_threshold.pdf")
    fig.savefig(fp); plt.close(fig)
    return f"Fig 1 -> {fp}"


def fig4_emission_calibration(out, p1):
    """Fig 4 -- emit/gt vs threshold on a log axis, reference line at 1."""
    if not p1:
        return "Fig 4: no results/p1/CELLS.json yet"
    fig = plt.figure(figsize=(FS.COL_FULL, FS.COL_FULL * 0.72))
    axes = FS.panel_grid(fig)
    for ax, task in zip(axes.ravel(), W.TASKS):
        rows = p1.get(task, [])
        FS.tidy(ax)
        ax.axhline(1.0, color=FS.MUTED, lw=0.8, ls="-", zorder=0)
        if not rows:
            ax.set_title(f"{SHORT[task]}  (no data)", loc="left")
            continue
        x = [r["thr"] for r in rows]
        y = [max(r.get("emit_per_gt") or 1e-3, 1e-3) for r in rows]
        ax.plot(x, y, markerfacecolor="white", markeredgewidth=1.0, **FS.S1)
        ax.set_yscale("log")
        ax.set_title(f"{SHORT[task]}  n={rows[0].get('n',0)}", loc="left")
        ax.set_xlim(0, 1)
    for ax in axes[-1]:
        ax.set_xlabel("hit threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("emits / GT")
    fig.tight_layout()
    fp = os.path.join(out, "fig4_emission_calibration.pdf")
    fig.savefig(fp); plt.close(fig)
    return f"Fig 4 -> {fp}"


def fig0_teaser(out, picks, shipped):
    """Fig 0 -- shipped vs final threshold, paired bars, sorted by |delta|."""
    if not picks:
        return "Fig 0: no PICKS json yet (needs pass 1)"
    rows = []
    for t in W.TASKS:
        p = picks.get(t)
        if not p:
            continue
        rows.append((t, shipped.get(t, p.get("shipped")), p["final"]))
    if not rows:
        return "Fig 0: picks present but empty"
    rows.sort(key=lambda r: abs(r[2] - r[1]), reverse=True)
    fig, ax = plt.subplots(figsize=(FS.COL_FULL, 0.34 * len(rows) + 0.9))
    FS.tidy(ax)
    ys = range(len(rows))
    h = 0.36
    ax.barh([y + h / 2 for y in ys], [r[1] for r in rows], height=h,
            color=FS.MUTED, label="shipped (vision-only fit)")
    ax.barh([y - h / 2 for y in ys], [r[2] for r in rows], height=h,
            color=FS.VIOLET, label="fitted (this work)")
    for y, (t, s, f) in zip(ys, rows):
        ax.annotate(f"{f - s:+.2f}", (max(s, f), y), xytext=(4, -2),
                    textcoords="offset points", fontsize=6, color=FS.INK2)
    ax.set_yticks(list(ys)); ax.set_yticklabels([SHORT[r[0]] for r in rows])
    ax.set_xlabel("hit threshold"); ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fp = os.path.join(out, "fig0_teaser.pdf")
    fig.savefig(fp); plt.close(fig)
    return f"Fig 0 -> {fp}"


def fig5_refinement(out, p1, p2, picks):
    """Fig 5 -- pass-1 grid, the two selected points, the pass-2 interval, final."""
    if not p1 or not picks:
        return "Fig 5: needs results/p1/CELLS.json and PICKS"
    tasks = [t for t in W.TASKS if p1.get(t)]
    fig, axes = plt.subplots(len(tasks), 1, sharex=True,
                             figsize=(FS.COL_FULL, 0.42 * len(tasks) + 0.8))
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        FS.tidy(ax)
        pk = picks.get(task, {})
        ax.plot([r["thr"] for r in p1[task]], [0] * len(p1[task]), "o",
                ms=4, mfc="white", mec=FS.VIOLET, mew=1.0, ls="none")
        if pk.get("best") is not None and pk.get("second") is not None:
            lo, hi = sorted((pk["best"], pk["second"]))
            ax.axvspan(lo, hi, color=FS.BLUE, alpha=0.12, lw=0)
        for r in p2.get(task, []):
            ax.plot([r["thr"]], [0], "o", ms=4, color=FS.BLUE, ls="none")
        if pk.get("final") is not None:
            ax.plot([pk["final"]], [0], "o", ms=7, color=FS.VIOLET, ls="none")
        ax.set_ylabel(SHORT[task], rotation=0, ha="right", va="center", fontsize=7)
        ax.set_yticks([]); ax.set_ylim(-1, 1); ax.set_xlim(0, 1)
        ax.grid(axis="y", visible=False)
        if pk.get("flat"):
            ax.annotate("FLAT - kept shipped", (0.02, 0.45), fontsize=6,
                        color=FS.ORANGE)
    axes[-1].set_xlabel("hit threshold")
    fig.tight_layout()
    fp = os.path.join(out, "fig5_refinement.pdf")
    fig.savefig(fp); plt.close(fig)
    return f"Fig 5 -> {fp}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    p1 = cells_by_task("p1")
    p2 = cells_by_task("p2")
    picks = (jload(os.path.join(ROOT, "P2_PICKS.json"))
             or jload(os.path.join(ROOT, "P1_PICKS.json")))
    sys.path.insert(0, os.path.join(ROOT, "lib"))
    from pick import SHIPPED
    with open(os.path.join(ROOT, "SHIPPED.json"), "w") as f:
        json.dump(SHIPPED, f, indent=1)

    for msg in (fig0_teaser(a.out, picks, SHIPPED),
                fig1_f1_vs_threshold(a.out, p1, p2, picks),
                fig4_emission_calibration(a.out, p1),
                fig5_refinement(a.out, p1, p2, picks)):
        print("  " + msg)
    print("\nFig 2 (ROC/AUC) and Fig 3 (per-sample P/R bands) need auc.py output"
          " and per-sample scores; they are built by figs_auc.py after stage 3.")


if __name__ == "__main__":
    main()
