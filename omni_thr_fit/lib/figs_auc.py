#!/usr/bin/env python
"""figs_auc.py -- Fig 2 (ROC / AUC) and Fig 3 (precision-recall bands) of sec.9.3.

  python lib/figs_auc.py --out figs/            # both, whichever has data
  python lib/figs_auc.py --out figs/ --only 2

Split out of figs.py because these two are the only figures whose inputs are the
raw per-tick `p_hit` stream and the per-sample score vectors rather than the
CELLS.json summary table. Same contract as figs.py: DEGRADES, NEVER GUESSES --
a task with no usable data is drawn as an explicitly empty panel saying so, and
is never interpolated from its neighbours.

WHAT THE TWO FIGURES ARE FOR (and why both)
-------------------------------------------
Fig 2 asks "is PERCEPTION right?" -- given the continuous p_hit, can the model
rank an event-adjacent tick above a quiet one at all? That is a property of the
backbone and the prompt, and it is invariant to where the threshold is put.

Fig 3 asks "is the GATE tuned?" -- at each threshold the grid actually ran, what
precision and recall did the live pipeline deliver, and how much of the movement
between thresholds is inside sampling noise?

A high AUC with a bad F1 means a mistuned gate. A low AUC means no threshold can
save it. Before this pair existed the two failures looked identical (AUC_DIAGNOSIS.md).

TWO SAMPLING SUBTLETIES THAT THE OBVIOUS IMPLEMENTATION GETS WRONG
------------------------------------------------------------------
1. THE BOOTSTRAP RESAMPLES VIDEOS, NOT TICKS. A 400-second video contributes
   ~400 ticks whose p_hit values are strongly serially correlated -- consecutive
   seconds of the same clip are very nearly the same observation. Resampling
   ticks i.i.d. treats them as 400 independent facts and returns a confidence
   interval several times too narrow. The unit of independent sampling here is
   the video (Fig 2) / the benchmark sample (Fig 3), so that is what is drawn
   with replacement, all of its rows travelling together (a cluster bootstrap).

2. THE ROC USES ONE REFERENCE CELL PER TASK, NOT THE POOLED GRID. It is tempting
   to pool p_hit from all ten threshold cells for ten times the ticks, and it
   would be wrong twice over: the cells re-run the SAME fifteen frozen ids, so
   the extra ticks are duplicates rather than new evidence; and p_hit is not
   actually threshold-invariant downstream of the first fire, because emitting
   decodes an utterance into the shared KV cache and every later tick is
   conditioned on it. So each task contributes exactly one cell -- its fitted
   threshold if the fit has produced one, else the mid-grid cell -- and the
   caption says which.
"""
from __future__ import annotations
import argparse, glob, json, os, random, sys
from collections import defaultdict

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
sys.path.insert(0, os.path.join(ROOT, "repo", "omniprofast"))
sys.path.insert(0, os.path.join(ROOT, "repo", "async_omni_v2"))

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import figstyle as FS                                            # noqa: E402
import worklist as W                                             # noqa: E402
import score_cells as SC                                         # noqa: E402
from auc import auc_score, label_ticks, load_gt, parse_scores    # noqa: E402
from metrics import ContentJudge, score_sample                   # noqa: E402

FS.use()

SHORT = {"cumulative_counting": "cumulative", "dedup_counting": "dedup",
         "event_narration": "narration", "explicit_target_grounding": "grounding",
         "instant_event_alert": "instant alert", "realtime_state_monitor": "state monitor",
         "semantic_condition_alert": "semantic alert",
         "sequential_step_instruction": "sequential step",
         "snapshot_counting": "snapshot"}
TASKS = sorted(SHORT)
NBOOT = 2000
SEED = 20260829          # fixed: the CI must not move when the figure is redrawn


def jload(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# reference-cell selection
# --------------------------------------------------------------------------
def grid_thresholds(pass_, task):
    wl = os.path.join(ROOT, f"worklist_{pass_}.tsv")
    try:
        rows = W.read_worklist(wl)
    except OSError:
        return []
    return sorted({r[2] for r in rows if r[1] == task})


def reference_cell(pass_, task):
    """-> (cell_dir, thr, provenance) for the ONE cell whose ticks feed the ROC.

    Preference order: the fitted threshold (so Fig 2 describes the system as it
    is actually configured), then the mid-grid cell. Either way the cell must
    hold every frozen id -- a half-filled cell would put a different, shorter set
    of videos on the ROC than the one the rest of the paper reports."""
    thrs = grid_thresholds(pass_, task)
    if not thrs:
        return None, None, None
    fitted = jload(os.path.join(ROOT, "FINAL_THRESHOLDS.json")) or {}
    want = fitted.get(task)
    order = ([(want, "fitted")] if want is not None else []) + \
            [(thrs[len(thrs) // 2], "mid-grid")]
    for thr, why in order:
        # snap to the grid: FINAL_THRESHOLDS holds the fitted value, the cell
        # directory is named by the grid point it was run at.
        near = min(thrs, key=lambda t: abs(t - thr))
        if abs(near - thr) > 1e-6 and why == "fitted":
            continue
        cell = W.cell_dir(pass_, task, near)
        if len(W.done_ids(cell)) >= len(W.frozen_ids(task)):
            return cell, near, why
    return None, None, None


# --------------------------------------------------------------------------
# Fig 2 -- ROC + AUC with a cluster bootstrap over videos
# --------------------------------------------------------------------------
def roc_points(y, p):
    """Full ROC as a step function. Written out rather than pulled from sklearn,
    which is not in this environment (auc.py makes the same choice for AUC)."""
    order = sorted(range(len(p)), key=lambda i: -p[i])
    P = sum(y)
    N = len(y) - P
    if not P or not N:
        return None
    tp = fp = 0
    xs, ys = [0.0], [0.0]
    i = 0
    while i < len(order):
        j = i                                   # consume the whole tie group:
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1                              # a tie is ONE operating point
        for k in range(i, j + 1):
            if y[order[k]]:
                tp += 1
            else:
                fp += 1
        xs.append(fp / N)
        ys.append(tp / P)
        i = j + 1
    return xs, ys


def boot_auc(per_video, rng, nboot=NBOOT):
    """95% CI on AUC, resampling VIDEOS with replacement (see module docstring)."""
    vids = list(per_video)
    if len(vids) < 2:
        return None, None
    out = []
    for _ in range(nboot):
        yy, pp = [], []
        for v in (vids[rng.randrange(len(vids))] for _ in vids):
            y, p = per_video[v]
            yy += y
            pp += p
        a = auc_score(yy, pp)
        if a is not None:
            out.append(a)
    if len(out) < nboot // 4:      # mostly degenerate resamples -> no honest CI
        return None, None
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]


def fig2(pass_, out):
    rng = random.Random(SEED)
    bench = os.environ.get("OMNIPRO_BENCHMARK_JSON")
    if not bench or not os.path.exists(bench):
        print("  fig2: SKIP (no benchmark json; set OMNIPRO_BENCHMARK_JSON)")
        return False

    panels, any_data = {}, False
    for task in TASKS:
        cell, thr, why = reference_cell(pass_, task)
        if not cell:
            panels[task] = None
            continue
        # the lanes write run.log; auc.py's own --all globs run_*.log, which is
        # the phase-A naming and matches nothing here (same trap as
        # resweep.parse_run -- see bin/audit_first_tick.py).
        paths = sorted(glob.glob(os.path.join(cell, "lane*", "run.log")))
        scores = parse_scores(paths)
        gt = load_gt(bench, task=task)
        if not scores or not gt:
            panels[task] = None
            continue
        y, p, per_video = label_ticks(scores, gt, 3.0)
        if not y or not sum(y) or sum(y) == len(y):
            panels[task] = None
            continue
        panels[task] = dict(y=y, p=p, per_video=per_video, thr=thr, why=why)
        any_data = True
    if not any_data:
        print("  fig2: SKIP (no cell has both p_hit ticks and ground truth)")
        return False

    fig = plt.figure(figsize=(FS.COL_FULL, FS.COL_FULL * 0.78))
    axes = FS.panel_grid(fig)
    for ax, task in zip(axes.ravel(), TASKS):
        FS.tidy(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        d = panels[task]
        if d is None:
            ax.set_title(f"{SHORT[task]}\n(no data)", color=FS.MUTED)
            ax.text(0.5, 0.5, "no complete cell", ha="center", va="center",
                    fontsize=6, color=FS.MUTED, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        ax.plot([0, 1], [0, 1], color=FS.MUTED, lw=0.8, ls=(0, (2, 2)), zorder=1)
        pts = roc_points(d["y"], d["p"])
        if pts:
            ax.plot(pts[0], pts[1], color=FS.VIOLET, lw=1.5, zorder=3)
            ax.fill_between(pts[0], pts[1], color=FS.VIOLET, alpha=0.10, zorder=2)
        a = auc_score(d["y"], d["p"])
        lo, hi = boot_auc(d["per_video"], rng)
        ci = f"\n[{lo:.2f}, {hi:.2f}]" if lo is not None else "\n[CI n/a]"
        # n is on every panel: sec.9.4 forbids a 15-video panel looking like a
        # 300-video one. pos% is there because AUC is meaningless without it.
        ax.set_title(f"{SHORT[task]}  AUC {a:.3f}{ci}", fontsize=7)
        # three SHORT lines, not two long ones: at this panel size a two-line
        # block ran off the left edge and printed over the y-axis labels.
        ax.text(0.96, 0.04,
                f"{len(d['per_video'])} vid, {len(d['y'])} ticks\n"
                f"{100.0*sum(d['y'])/len(d['y']):.1f}% positive\n"
                f"thr {d['thr']:.2f} ({d['why']})",
                ha="right", va="bottom", fontsize=5.5, color=FS.INK2,
                linespacing=1.35, transform=ax.transAxes)
    for ax in axes[-1]:
        ax.set_xlabel("false positive rate")
    for ax in axes[:, 0]:
        ax.set_ylabel("true positive rate")
    fig.tight_layout(pad=0.4)
    path = os.path.join(out, "fig2_roc.pdf")
    fig.savefig(path)
    # PDF is what LaTeX includes; the PNG exists only so the figure can actually
    # be LOOKED at -- there is no pdftoppm on this cluster and an unrendered
    # figure is an unchecked figure.
    fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    print(f"  fig2: {path}")
    return True


# --------------------------------------------------------------------------
# Fig 3 -- per-sample precision / recall bands across the fitted grid
# --------------------------------------------------------------------------
def boot_pr(per_sample, rng, nboot=NBOOT):
    """Median and 95% band for (precision, recall), resampling SAMPLES.

    The point estimate pools tp/fp/fn exactly the way metrics.aggregate does --
    micro-averaged, not the mean of per-sample ratios, which would weight a
    one-event video the same as a twelve-event one."""
    n = len(per_sample)
    if n < 2:
        return None
    ps, rs = [], []
    for _ in range(nboot):
        tp = fp = fn = 0
        for s in (per_sample[rng.randrange(n)] for _ in range(n)):
            tp += s["tp_time"]; fp += s["fp"]; fn += s["fn"]
        ps.append(tp / (tp + fp) if tp + fp else 0.0)
        rs.append(tp / (tp + fn) if tp + fn else 0.0)
    ps.sort(); rs.sort()
    q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
    return dict(p=q(ps, .5), p_lo=q(ps, .025), p_hi=q(ps, .975),
                r=q(rs, .5), r_lo=q(rs, .025), r_hi=q(rs, .975))


def fig3(pass_, out):
    rng = random.Random(SEED)
    # Precision and recall are pure timing -- no content verdict is involved --
    # but score_sample calls the judge on every MATCH regardless. offline=True is
    # the documented cache-only switch: without it, redrawing this figure fires
    # one API call per matched emit per threshold cell and writes a few thousand
    # failures into the judge cache for numbers this figure does not use.
    judge = ContentJudge()
    judge.offline = True
    series, any_data = {}, False
    for task in TASKS:
        rows = []
        for thr in grid_thresholds(pass_, task):
            cell = W.cell_dir(pass_, task, thr)
            preds = SC.cell_predictions(cell)
            if not preds:
                continue
            per = [score_sample(p, tolerance=3.0, judge=judge) for p in preds]
            b = boot_pr(per, rng)
            if b:
                # complete = every frozen id present. An incomplete cell is still
                # plotted (it is real data) but marked, per sec.9.4's rule that a
                # reliable=False cell is never silently indistinguishable.
                b.update(thr=thr, n=len(per),
                         complete=len(preds) >= len(W.frozen_ids(task)))
                rows.append(b)
        rows.sort(key=lambda r: r["thr"])
        series[task] = rows
        any_data = any_data or bool(rows)
    if not any_data:
        print("  fig3: SKIP (no scored cells yet)")
        return False

    fig = plt.figure(figsize=(FS.COL_FULL, FS.COL_FULL * 0.72))
    axes = FS.panel_grid(fig)
    for ax, task in zip(axes.ravel(), TASKS):
        FS.tidy(ax)
        ax.set_ylim(0, 1.02)
        rows = series[task]
        if not rows:
            ax.set_title(f"{SHORT[task]}\n(no data)", color=FS.MUTED)
            ax.text(0.5, 0.5, "no scored cell", ha="center", va="center",
                    fontsize=6, color=FS.MUTED, transform=ax.transAxes)
            continue
        x = [r["thr"] for r in rows]
        ax.fill_between(x, [r["p_lo"] for r in rows], [r["p_hi"] for r in rows],
                        color=FS.VIOLET, alpha=0.15, lw=0, zorder=2)
        ax.fill_between(x, [r["r_lo"] for r in rows], [r["r_hi"] for r in rows],
                        color=FS.BLUE, alpha=0.15, lw=0, zorder=2)
        # precision solid+filled, recall dashed+open on top -- the same redundant
        # encoding figs.py uses, so that where the two coincide the dashes let
        # the lower series read through instead of erasing it.
        ax.plot(x, [r["p"] for r in rows], label="precision", zorder=4,
                lw=1.6, **FS.S1)
        ax.plot(x, [r["r"] for r in rows], label="recall", zorder=5, lw=1.4,
                markerfacecolor="white", markeredgewidth=1.0, **FS.S2)
        # An incomplete cell gets a tick on a rug along the BOTTOM, not a
        # full-height hatched span. The span version was drawn first and was
        # unreadable: on dedup/narration/sequential most cells are still filling,
        # so eight overlapping stripes made the panel look uniformly hatched and
        # buried the curves it was supposed to annotate.
        for r in rows:
            if not r["complete"]:
                ax.plot([r["thr"]], [0.012], marker="|", markersize=5,
                        color=FS.MUTED, lw=0, markeredgewidth=1.1, zorder=6,
                        clip_on=False)
        ns = sorted({r["n"] for r in rows})
        ax.set_title(f"{SHORT[task]}  n={ns[0] if len(ns)==1 else f'{ns[0]}-{ns[-1]}'}",
                     fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("hit threshold")
    for ax in axes[:, 0]:
        ax.set_ylabel("rate")
    fig.tight_layout(pad=0.4)
    # tight_layout AFTER the axes, legend AFTER that: placed before, the legend
    # was laid out against the pre-tightened figure and landed on top of the
    # middle panel's x-axis label.
    h, l = axes[0][0].get_legend_handles_labels()
    if h:
        h.append(plt.Line2D([], [], marker="|", lw=0, color=FS.MUTED,
                            markeredgewidth=1.1))
        l.append("cell still filling")
        fig.subplots_adjust(bottom=0.14)
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.005))
    path = os.path.join(out, "fig3_pr_bands.pdf")
    fig.savefig(path)
    # PDF is what LaTeX includes; the PNG exists only so the figure can actually
    # be LOOKED at -- there is no pdftoppm on this cluster and an unrendered
    # figure is an unchecked figure.
    fig.savefig(path[:-4] + ".png", dpi=200)
    plt.close(fig)
    print(f"  fig3: {path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", default="p1")
    ap.add_argument("--out", default=os.path.join(ROOT, "figs"))
    ap.add_argument("--only", choices=["2", "3"], default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.only in (None, "2"):
        fig2(a.pass_, a.out)
    if a.only in (None, "3"):
        fig3(a.pass_, a.out)


if __name__ == "__main__":
    main()
