#!/usr/bin/env python
"""tables.py -- sec.9.2's Tables 1-4 as LaTeX fragments.

  python lib/tables.py --out paper/tables/

Each table is written only if its inputs exist; a missing table becomes a stub
that says WHAT IS MISSING, so a premature build produces a paper with an honest
hole rather than invented numbers.

sec.6.2 IS ENFORCED IN THE MARKUP, not left to the author: `n`, `n_gt` and
`n_emit` travel beside every F1 in Table 1, and a metric that the judge could not
supply prints as \\WITHHELD rather than as a number or a dash that could be read
as zero.
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.environ["THR_ROOT"]
sys.path.insert(0, os.path.join(ROOT, "lib"))
import worklist as W                                             # noqa: E402
from pick import SHIPPED                                         # noqa: E402

SHORT = {"cumulative_counting": "cumulative", "dedup_counting": "dedup",
         "event_narration": "narration", "explicit_target_grounding": "grounding",
         "instant_event_alert": "instant alert", "realtime_state_monitor": "state monitor",
         "semantic_condition_alert": "semantic alert",
         "sequential_step_instruction": "sequential step",
         "snapshot_counting": "snapshot"}


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def fmt(v, nd=4):
    """A number, or the WITHHELD macro. Never a bare dash: sec.3 of CLAUDE.md --
    content metrics are WITHHELD, never guessed, and a dash reads as zero."""
    return "\\WITHHELD" if v is None else f"{v:.{nd}f}"


def stub(name, why, label=None):
    """A placeholder that still ANCHORS its label.

    Without the label, every \\ref to a not-yet-built table renders as "??" and
    the prose silently loses its cross-references -- which reads as a LaTeX bug
    rather than as an unfinished run, and hides which numbers are still pending.
    """
    lab = f"\\label{{{label}}}" if label else ""
    return (f"% {name}: NOT BUILT\n"
            f"\\begin{{center}}\\fbox{{\\parbox{{0.9\\linewidth}}"
            f"{{\\textbf{{{name} not yet available.}} {why}}}}}{lab}\\end{{center}}\n")


def table1(picks, cells):
    """Finalized per-task thresholds. sec.9.2: never a bare F1."""
    if not picks or not cells:
        return stub("Table 1", "Requires P1/P2 picks and scored cells.")
    by = {}
    for r in cells:
        by.setdefault(r["task"], {})[round(r["thr"], 4)] = r
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Finalized per-task gate thresholds. $n$, $n_{gt}$ and $n_{emit}$"
         r" are reported beside every F1: at $\sim$2.4 GT events per sample a"
         r" 15-sample cell holds $\sim$36 scoring events, and a 0.02--0.03 gap"
         r" between adjacent thresholds is not a real difference.}",
         r"\label{tab:thresholds}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"task & ship & p1 & 2nd & \textbf{final} & $\Delta$ & time-F1 & joint-F1"
         r" & $n$ & e/gt \\", r"\midrule"]
    for t in W.TASKS:
        p = picks.get(t)
        if not p:
            continue
        c = by.get(t, {}).get(round(p["final"], 4))
        d = p["final"] - SHIPPED[t]
        # Build the measured columns separately. Folding them into one f-string
        # with a trailing `if c else "--"` made the ternary bind to the WHOLE
        # string, so a task whose winning cell was missing collapsed its entire
        # row to a single "--" -- a silently malformed table, not a visible gap.
        tf = fmt(c["time_f1"]) if c else "--"
        jf = fmt(c.get("joint_f1")) if c else "--"
        nn = str(c["n"]) if c else "--"
        eg = f"{c['emit_per_gt']:.2f}" if c and c.get("emit_per_gt") is not None else "--"
        row = (f"{SHORT[t]} & {SHIPPED[t]:.3f} & {p['best']:.3f} & "
               f"{p['second']:.3f} & \\textbf{{{p['final']:.3f}}} & {d:+.3f} & "
               f"{tf} & {jf} & {nn} & {eg} \\\\")
        if p.get("flat"):
            row += "  % FLAT: kept shipped"
        L.append(row)
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def table2(overall):
    """Complete OmniPro Online, 2,700 samples, three strata x three blocks."""
    if not overall:
        return stub("Table 2", label="tab:full", why= "Requires stage 3 (OVERALL.json).")
    blocks = [("full", "full OmniPro, fitted thresholds (the claim)"),
              ("in_sample_ceiling", "5\\% in-sample ceiling (optimistically biased)"),
              ("fit_disjoint", "fit-disjoint rescore")]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Complete OmniPro Online. The in-sample block is a headroom"
         r" figure, not a result: it is scored on the very samples that chose the"
         r" thresholds. Baselines use a different subset and scoring and are not"
         r" directly comparable.}",
         r"\label{tab:full}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r" & stratum & $n$ & $n_{gt}$ & $n_{emit}$ & t-P & t-R & time-F1 & joint-F1 \\"]
    for key, label in blocks:
        blk = overall.get(key)
        if not blk:
            continue
        L.append(r"\midrule")
        L.append(rf"\multicolumn{{9}}{{l}}{{\textit{{{label}}}}} \\")
        for s in ("gross", "audio_not_required", "audio_required"):
            a = blk.get(s)
            if not a:
                continue
            o = a["overall"]
            L.append(f" & {s.replace('_',' ')} & {o['n_samples']} & {o['n_gt']} & "
                     f"{o['n_emits']} & {o['time_precision']:.4f} & "
                     f"{o['time_recall']:.4f} & {o['time_f1']:.4f} & "
                     f"{fmt(o['joint_f1'])} \\\\")
    L += [r"\midrule",
          r"\multicolumn{9}{l}{\textit{published baselines (different subset /"
          r" scoring)}} \\",
          r" & MiniCPM-o 4.5 (9B), Online & & & & & & & 0.209 \\",
          r" & MMDuet2, Online & & & & & & & 0.113 \\",
          r" & LiveStar, Online & & & & & & & 0.036 \\",
          r" & Qwen2.5-Omni-7B, \emph{Probe} & & & & & & & 0.201 \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def table3(overall):
    """Per-task complete eval at the final thresholds."""
    if not overall or not overall.get("full", {}).get("gross"):
        return stub("Table 3", label="tab:pertask", why= "Requires stage 3 (OVERALL.json).")
    pt = overall["full"]["gross"]["per_task"]
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Per-task results on the complete 2{,}700-sample benchmark at"
         r" the fitted thresholds, with the shipped threshold beside each so the"
         r" effect of fitting is visible per task.}",
         r"\label{tab:pertask}",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"task & shipped thr & $n$ & $n_{gt}$ & time-F1 & joint-F1 \\", r"\midrule"]
    for t in W.TASKS:
        b = pt.get(t)
        if not b:
            continue
        L.append(f"{SHORT[t]} & {SHIPPED[t]:.3f} & {b['n_samples']} & "
                 f"{b['n_gt']} & {b['time_f1']:.4f} & {fmt(b['joint_f1'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def table4(abl):
    """Ablation of the fit. Mirrors METHODOLOGY_FOR_PAPER's 0.190/0.254/0.316.

    This table reports DIFFERENCES WITH INTERVALS, not four bare point estimates.
    RUNBOOK sec.2.6-2.7 measured that the per-task fit does not beat a single
    global threshold outside noise; a table of four numbers with the fitted arm
    highest would invite precisely the reading that finding rules out. The arms
    run on the same videos, so every delta is a PAIRED bootstrap against the best
    single global -- the arm per-task fitting has to beat to justify itself.

    Off-grid arms are marked, never silently rounded: `shipped_per_task` uses
    values (0.992, 0.985, ...) no cell was ever run at, and 0.5 is not a grid
    point either. The caption says which rows are approximations and by how much.
    """
    if not abl:
        return stub("Table 4", label="tab:ablation", why="Requires the ablation run (ABLATION.json):"
                               " global 0.5 / best single global / shipped per-task"
                               " / fitted per-task.")
    arms = abl.get("arms", {})
    ROWS = (("global_0.5", "global $0.5$"),
            ("best_single_global", "best single global"),
            ("shipped_per_task", "shipped per-task"),
            ("fitted_per_task", "fitted per-task"))
    bsg = abl.get("best_single_global_thr")
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{%s}" % _t4caption(abl),
         r"\label{tab:ablation}",
         r"\begin{tabular}{lrrrl}", r"\toprule",
         r"gate & time-P & time-R & time-F1 & $\Delta$ vs best global [95\% CI] \\",
         r"\midrule"]
    for k, lab in ROWS:
        r_ = arms.get(k)
        if not r_:
            continue
        if r_.get("off_grid"):
            lab += r"$^{\dagger}$"
        if k == "best_single_global" and bsg is not None:
            lab += f" ({bsg:g})"
        d = r_.get("delta_vs_best_single_global")
        if d:
            lo, hi = d["ci95"]
            # an interval containing zero is the finding, so say so in the cell
            # rather than leaving the reader to compare the signs themselves.
            mark = "" if (lo > 0 or hi < 0) else r"\,(n.s.)"
            dcell = f"${d['point']:+.3f}$ $[{lo:+.3f},{hi:+.3f}]${mark}"
        else:
            dcell = r"---"
        f1 = fmt(r_.get("time_f1"))
        if k == "best_single_global":
            f1 = r"\textbf{%s}" % f1
        L.append(f"{lab} & {fmt(r_.get('time_p'))} & {fmt(r_.get('time_r'))} "
                 f"& {f1} & {dcell} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def _t4caption(abl):
    """Everything a reader needs to not over-read this table, in the caption."""
    n = abl.get("n_videos_total")
    pass_ = abl.get("pass", "?")
    bits = [f"Ablation of the fit, pooled micro-averaged over {n} videos "
            f"({'fit subset' if pass_ in ('p1', 'p2') else pass_}). "
            r"\emph{global $0.5$} is the shipped gate's effective behaviour "
            r"(\texttt{hysteresis} tested $p_{\mathrm{hit}}\!\ge\!0.5$ and "
            r"ignored the per-task table); \emph{shipped per-task} is the "
            r"vision-only fit; \emph{fitted per-task} is this work."]
    off = {k: v.get("off_grid", {}) for k, v in abl.get("arms", {}).items()}
    off = {k: v for k, v in off.items() if v}
    if off:
        worst = max((d["abs_error"] for v in off.values() for d in v.values()))
        bits.append(r"$^{\dagger}$Threshold not on the run grid; served by the "
                    r"nearest cell actually evaluated (max.\ error %.3f). "
                    r"These rows are approximations." % worst)
    r = abl.get("bsg_reselect_rate")
    if r is not None:
        bits.append(r"The single global optimum is stable: re-selected in "
                    r"%.0f\%% of %d paired bootstrap resamples of the videos "
                    r"(chance %.0f\%%). The per-task fits are not "
                    r"(25--61\%%; see text)." %
                    (100 * r, 2000, 100 * abl.get("bsg_chance_rate", 0)))
    bits.append(r"joint-F1 is WITHHELD: the judge was unreachable for this run "
                r"and content verdicts are never guessed.")
    return " ".join(bits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    picks = (jload(os.path.join(ROOT, "P2_PICKS.json"))
             or jload(os.path.join(ROOT, "P1_PICKS.json")))
    cells = (jload(os.path.join(ROOT, "results", "p2", "CELLS.json")) or []) + \
            (jload(os.path.join(ROOT, "results", "p1", "CELLS.json")) or [])
    overall = jload(os.path.join(ROOT, "OVERALL.json"))
    abl = jload(os.path.join(ROOT, "ABLATION.json"))
    for name, body in (("table1", table1(picks, cells)),
                       ("table2", table2(overall)),
                       ("table3", table3(overall)),
                       ("table4", table4(abl))):
        fp = os.path.join(a.out, f"{name}.tex")
        with open(fp, "w") as f:
            f.write(body)
        print(f"  {name}: {'STUB' if 'NOT BUILT' in body else 'built'} -> {fp}")


if __name__ == "__main__":
    main()
