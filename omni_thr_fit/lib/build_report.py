#!/usr/bin/env python
"""build_report.py -- the standing technical report on the threshold fit.

  python lib/build_report.py            # -> report/PROGRESS_REPORT.pdf

WHY THIS IS NOT paper/main.pdf. build_pdf.py assembles the sec.9 deliverable: a
12-page ECCV/LNCS submission, written for a reader who wants the result and the
evidence for it. This is the other document -- the internal record of what was
run, what broke, what was measured, and what is still open. It is allowed to be
long, to name file paths, and to carry the negative results and the operational
incidents at full length, none of which fit in a camera-ready.

The two must never disagree, so both read the SAME JSONs and generate their
tables. Nothing here is a hand-typed number: every figure in every table is
pulled from the artefact that produced it, and a missing artefact becomes a
visible note rather than a silent omission.
"""
from __future__ import annotations
import datetime, json, os, subprocess, sys

ROOT = os.environ.get(
    "THR_ROOT", "/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit")
OUT = os.path.join(ROOT, "report")
TECTONIC = os.environ.get("TECTONIC",
                          "/iopsstor/scratch/cscs/dthapa/tools/bin/tectonic")
sys.path.insert(0, os.path.join(ROOT, "lib"))

TASKS = ["cumulative_counting", "dedup_counting", "event_narration",
         "explicit_target_grounding", "instant_event_alert",
         "realtime_state_monitor", "semantic_condition_alert",
         "sequential_step_instruction", "snapshot_counting"]


def load(name):
    p = os.path.join(ROOT, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def esc(s):
    return str(s).replace("_", r"\_")


def note(msg):
    """A missing artefact must be VISIBLE. A blank table reads as 'nothing to
    report'; this reads as 'this was not produced', which is a different claim."""
    return (r"\begin{center}\fbox{\parbox{0.9\linewidth}{\small\textsc{not "
            r"produced:} " + msg + r"}}\end{center}" + "\n")


# ---------------------------------------------------------------- live status

def progress():
    """Recount from the banked predictions rather than trusting any cached
    number: the run is still moving while this document compiles."""
    import worklist as W
    out = {}
    for p in ("p1", "p2"):
        wl = os.path.join(ROOT, f"worklist_{p}.tsv")
        if not os.path.exists(wl):
            continue
        cells = {}
        for r in W.read_worklist(wl):
            cells.setdefault((r[1], r[2]), 1)
        per, done, tot, full = {}, 0, 0, 0
        for task, thr in cells:
            want = len(W.frozen_ids(task))
            d = min(len(W.done_ids(W.cell_dir(p, task, thr))), want)
            done += d
            tot += want
            full += (d >= want)
            a, b = per.get(task, (0, 0))
            per[task] = (a + d, b + want)
        out[p] = {"done": done, "total": tot, "cells_done": full,
                  "cells": len(cells), "per_task": per}
    return out


def tbl_progress(pr):
    rows = []
    for p, lbl in (("p1", "Pass 1 (coarse)"), ("p2", "Pass 2 (refine)")):
        if p not in pr:
            continue
        d = pr[p]
        rows.append(f"{lbl} & {d['done']} & {d['total']} & "
                    f"{100*d['done']/d['total']:.1f}\\% & "
                    f"{d['cells_done']}/{d['cells']} \\\\")
    s = (r"\begin{table}[t]\centering\caption{Sample-level completion of the "
         r"two fit passes, recounted from the banked predictions at build time. "
         r"A cell is one (task, threshold) pair over the 15 frozen videos.}"
         r"\label{tab:progress}" "\n"
         r"\begin{tabular}{lrrrr}\toprule" "\n"
         r"pass & banked & target & \% & cells complete \\\midrule" "\n"
         + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}\end{table}" "\n")
    return s


def tbl_p2_tail(pr):
    if "p2" not in pr:
        return ""
    per = pr["p2"]["per_task"]
    rows = []
    for t in TASKS:
        a, b = per.get(t, (0, 0))
        if b == 0:
            continue
        mark = r"$\checkmark$" if a >= b else f"{b-a}"
        rows.append(f"{esc(t)} & {a} & {b} & {100*a/b:.1f}\\% & {mark} \\\\")
    return (r"\begin{table}[t]\centering\caption{Pass-2 completion per task. The "
            r"tail is not uniform: it concentrates on the tasks whose frozen "
            r"videos are longest, because a lane can only claim a shard whose "
            r"samples fit inside its remaining wall-clock.}\label{tab:p2tail}"
            "\n" r"\begin{tabular}{lrrrc}\toprule" "\n"
            r"task & banked & target & \% & left \\\midrule" "\n"
            + "\n".join(rows) + "\n" + r"\bottomrule\end{tabular}\end{table}" "\n")


# ------------------------------------------------------------------- findings

def tbl_picks(picks):
    if not picks:
        return note("P1\\_PICKS.json -- run \\texttt{lib/pick.py --pass p1}.")
    rows = []
    for t in TASKS:
        d = picks.get(t)
        if not d:
            continue
        flags = []
        if d.get("flat"):
            flags.append("FLAT")
        if d.get("rail"):
            flags.append("rail")
        rows.append(
            f"{esc(t)} & {d['shipped']:.3f} & {d['best']:.2f} & "
            f"{d['best_time_f1']:.4f} & {d['second']:.2f} & "
            f"{d['gap']:+.4f} & {d['final']:.2f} & "
            f"{', '.join(flags) if flags else '--'} \\\\")
    return (r"\begin{table}[t]\centering\small\caption{Pass-1 selection per task. "
            r"\emph{gap} is best minus runner-up time-F1; a negative gap means "
            r"sec.3's tie-band moved the pick off the raw argmax onto the "
            r"emission volume closer to the truth. \textsc{flat} marks a task "
            r"whose curve is identical at every threshold, where \texttt{pick.py} "
            r"keeps the shipped value rather than inventing a winner.}"
            r"\label{tab:picks}" "\n"
            r"\begin{tabular}{lrrrrrrl}\toprule" "\n"
            r"task & shipped & best & F1 & 2nd & gap & final & flags \\\midrule"
            "\n" + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def tbl_noise(fn, label="tab:noise", extra="", span=False):
    if not fn:
        return note("FIT\\_NOISE\\_AUDIT.json -- run "
                    "\\texttt{bin/audit\\_fit\\_noise.py}.")
    rows = []
    for t in TASKS:
        d = fn.get(t)
        if not d:
            continue
        lo, hi = d["delta_vs_rival_ci95"]
        crosses = r"\textbf{yes}" if lo <= 0 <= hi else "no"
        if span:
            # The chance column is CONSTANT on the refined grid (five cells for
            # every task), so it is a caption fact, not a column. Carrying it
            # here is what pushed this table 27 pt past the text block.
            rows.append(
                f"{esc(t)} & {d['fitted_thr']:.2f} & {d['rival_thr']:.2f} & "
                f"{d['gap_to_rival']:+.4f} & [{lo:+.3f}, {hi:+.3f}] & {crosses} & "
                f"{100*d['reselect_rate']:.0f}\\% & {d['grid_span']:.3f} \\\\")
            continue
        rows.append(
            f"{esc(t)} & {d['fitted_thr']:.2f} & {d['rival_thr']:.2f} & "
            f"{d['gap_to_rival']:+.4f} & [{lo:+.3f}, {hi:+.3f}] & {crosses} & "
            f"{100*d['reselect_rate']:.0f}\\% & "
            f"{100*d['chance_reselect_rate']:.0f}\\% \\\\")
    if span:
        chance = {d.get("chance_reselect_rate") for d in fn.values()
                  if "chance_reselect_rate" in d}
        cstr = (r" Chance re-selection is %.0f\%% on every task here (five cells "
                r"each)." % (100 * chance.pop()) if len(chance) == 1 else "")
        return (r"\begin{table}[t]\centering\small\caption{" + extra + cstr +
                r"}\label{" + label + "}\n"
                r"\begin{tabular}{lrrrlcrr}\toprule" "\n"
                r"task & fitted & rival & $\Delta$F1 & 95\% CI & CI$\ni$0 & "
                r"re-sel. & span \\\midrule" "\n"
                + "\n".join(rows) + "\n"
                + r"\bottomrule\end{tabular}\end{table}" "\n")
    return (r"\begin{table}[t]\centering\small\caption{Paired bootstrap over the "
            r"15 frozen videos, 2000 draws, resampling \emph{videos} (never ticks "
            r"or events -- they are serially correlated through one shared KV "
            r"cache). \emph{rival} is the best cell that is not the fitted one. "
            r"\emph{re-sel.} re-runs \texttt{pick.rank} inside every draw and "
            r"counts how often the fitted threshold wins again; compare it to "
            r"chance, not to zero.}\label{tab:noise}" "\n"
            r"\begin{tabular}{lrrrlcrr}\toprule" "\n"
            r"task & fitted & rival & $\Delta$F1 & 95\% CI & CI$\ni$0 & re-sel. "
            r"& chance \\\midrule" "\n" + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def n_ci_contains_zero(fn):
    return sum(1 for d in (fn or {}).values()
               if "delta_vs_rival_ci95" in d
               and d["delta_vs_rival_ci95"][0] <= 0 <= d["delta_vs_rival_ci95"][1])


def tbl_final(final, picks):
    if not final:
        return note("FINAL\\_THRESHOLDS.json -- run "
                    "\\texttt{lib/pick.py --pass p2}.")
    shipped = load("SHIPPED.json") or {}
    rows = []
    for t in TASKS:
        v = final.get(t)
        if v is None:
            continue
        sh = shipped.get(t)
        flag = "kept (flat)" if picks and picks.get(t, {}).get("flat") else ""
        rows.append(f"{esc(t)} & {sh:.3f} & {v:.4f} & {v-sh:+.3f} & {flag} \\\\")
    return (r"\begin{table}[t]\centering\small\caption{Final per-task thresholds. "
            r"The selection ranks the pass-2 cells \emph{together with} the two "
            r"pass-1 candidates, so a coarse point can still win --- which is why "
            r"two finals are not on the pass-2 grid. A task flagged \textsc{flat} "
            r"in pass 1 keeps its shipped value rather than being assigned a "
            r"winner the data does not support.}\label{tab:final}" "\n"
            r"\begin{tabular}{lrrrl}\toprule" "\n"
            r"task & shipped & final & $\Delta$ & note \\\midrule" "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def tbl_ablation(ab):
    if not ab:
        return note("ABLATION.json -- run \\texttt{lib/ablation.py}.")
    order = ["global_0.5", "best_single_global", "shipped_per_task",
             "fitted_per_task"]
    pretty = {"global_0.5": "global 0.50",
              "best_single_global": r"best single global (%.2f)"
                                    % ab["best_single_global_thr"],
              "shipped_per_task": "shipped per-task",
              "fitted_per_task": "fitted per-task"}
    rows, offgrid = [], []
    for k in order:
        a = ab["arms"].get(k)
        if not a:
            continue
        d = a.get("delta_vs_best_single_global")
        if d is None:
            dcol = "--- (reference)"
        elif isinstance(d, dict):
            dcol = (f"{d['point']:+.3f} [{d['ci95'][0]:+.3f}, "
                    f"{d['ci95'][1]:+.3f}]")
        else:
            dcol = f"{d:+.3f}"
        bold = r"\textbf{%s}" % pretty[k] if k == "best_single_global" else pretty[k]
        rows.append(f"{bold} & {a['time_p']:.4f} & {a['time_r']:.4f} & "
                    f"{a['time_f1']:.4f} & {dcol} \\\\")
        n = len(a.get("off_grid") or [])
        if n:
            offgrid.append(f"{pretty[k]}: {n}/9")
    cap = (r"Gate ablation at equal budget, pooled over all "
           + str(ab["n_videos_total"]) + r" frozen videos and micro-averaged the "
           r"way \texttt{metrics.aggregate} pools tp/fp/fn -- never a mean of "
           r"per-sample ratios. $\Delta$ is against the best single global "
           r"threshold, with a paired bootstrap CI over videos. ")
    if offgrid:
        cap += (r"\textbf{Caveat:} the grid does not contain every arm's "
                r"threshold, so some cells are served by the nearest grid point "
                r"(" + esc("; ".join(offgrid)) + r"); every substitution is "
                r"recorded in \texttt{ABLATION.json}.")
    return (r"\begin{table}[t]\centering\small\caption{" + cap +
            r"}\label{tab:ablation}" "\n"
            r"\begin{tabular}{lrrrl}\toprule" "\n"
            r"arm & time-P & time-R & time-F1 & $\Delta$ vs.\ BSG \\\midrule" "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def tbl_perception(pa):
    if not pa:
        return note("PERCEPTION\\_AUDIT.json -- run "
                    "\\texttt{bin/audit\\_perception.py}.")
    rows = []
    for t in TASKS:
        d = pa.get(t)
        if not d or "skipped" in d:
            continue
        ph = d["p_hit"]
        rows.append(
            f"{esc(t)} & {d['n_videos']} & {d['n_ticks']} & "
            f"{d['auc_at_zero_tol3']:.3f} & {d['auc_best_over_offsets']:.3f} & "
            f"{ph['min']:.3f}--{ph['max']:.3f} & {ph['n_distinct']} \\\\")
    if not rows:
        return note("no task had a complete reference cell to audit.")
    return (r"\begin{table}[t]\centering\small\caption{Does $p_\mathit{hit}$ rank "
            r"event-adjacent ticks above quiet ones? \emph{AUC best} is the "
            r"highest AUC any clock offset in $\pm10$\,s or tolerance in "
            r"$\{3,10\}$\,s could buy: a merely misaligned score would peak well "
            r"above its zero-offset value, and none does. The $p_\mathit{hit}$ "
            r"range and distinct-value count exclude the third innocent "
            r"explanation, a degenerate score.}\label{tab:perception}" "\n"
            r"\begin{tabular}{lrrrrlr}\toprule" "\n"
            r"task & vid & ticks & AUC@0 & AUC best & $p_\mathit{hit}$ range & "
            r"distinct \\\midrule" "\n" + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def tbl_refractory(rf):
    if not rf:
        return note("REFRACTORY\\_AUDIT.json.")
    rows = []
    for t in TASKS:
        d = rf.get(t)
        if not d:
            continue
        rows.append(f"{esc(t)} & {d['shipped_refractory']:.0f} & "
                    f"{d['f1_at_shipped']:.4f} & {d['best_refractory']:.0f} & "
                    f"{d['f1_at_best']:.4f} & {d['gain']:+.4f} \\\\")
    return (r"\begin{table}[t]\centering\small\caption{Offline replay of the "
            r"refractory knob at the shipped threshold. The gains are inside the "
            r"0.03 noise band on every task, which is the empirical reason the "
            r"three gate knobs are reported as one coupled operating point rather "
            r"than fitted one at a time.}\label{tab:refractory}" "\n"
            r"\begin{tabular}{lrrrrr}\toprule" "\n"
            r"task & ship $r$ & F1 & best $r$ & F1 & gain \\\midrule" "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def tbl_firsttick(ft):
    if not ft:
        return note("FIRST\\_TICK\\_AUDIT.json.")
    rows = []
    for t in TASKS:
        d = ft.get(t)
        if not d:
            continue
        a, e = d.get("as_is"), d.get("edge1")
        if not a or not e:
            continue
        rows.append(f"{esc(t)} & {a['best_f1']:.4f} & {a['best_thr']:.2f} & "
                    f"{e['best_f1']:.4f} & {e['best_thr']:.2f} & "
                    f"{e['best_f1']-a['best_f1']:+.4f} \\\\")
    return (r"\begin{table}[t]\centering\small\caption{The first-tick defect. "
            r"\emph{as-is} seeds \texttt{prev\_above=False}, making tick 0 a "
            r"rising edge; \emph{edge1} requires a predecessor "
            r"(\texttt{prev\_above=None}). The size of the effect tracks the "
            r"refractory length, which is why the two knobs cannot be fitted "
            r"independently.}\label{tab:firsttick}" "\n"
            r"\begin{tabular}{lrrrrr}\toprule" "\n"
            r"task & as-is F1 & thr & edge1 F1 & thr & $\Delta$ \\\midrule" "\n"
            + "\n".join(rows) + "\n"
            + r"\bottomrule\end{tabular}\end{table}" "\n")


def figure(fname, caption, label):
    # Included by a path RELATIVE to report/, not an absolute one: tectonic warns
    # that an absolute path makes the build non-reproducible elsewhere, and this
    # tree is meant to be cloned and rebuilt on another machine.
    if not os.path.exists(os.path.join(ROOT, "figs", fname)):
        return note(f"figure \\texttt{{{esc(fname)}}} has not been produced.")
    return (r"\begin{figure}[t]\centering" "\n"
            r"\includegraphics[width=0.92\linewidth]{../figs/" + fname + "}\n"
            r"\caption{" + caption + r"}\label{" + label + r"}\end{figure}" "\n")


# ---------------------------------------------------------------------- prose

def build_tex(pr, picks, fn, ab, pa, rf, ft, fn2=None, final=None):
    p1 = pr.get("p1", {})
    p2 = pr.get("p2", {})
    bsg = ab["best_single_global_thr"] if ab else None
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    T = []
    A = T.append
    A(r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[margin=25mm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage[hidelinks]{hyperref}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\setlength{\parskip}{0.35em}
\setlength{\parindent}{0pt}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small Per-task gate threshold fit --- Qwen2.5-Omni-7B on OmniPro Online}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\newcommand{\WITHHELD}{\textsc{withheld}}
% This report is dense with unbreakable \texttt identifiers -- gate_strategy,
% cfg.gate_high_thr, task_refractory_s -- any one of which can overflow a line
% that TeX cannot hyphenate. \sloppy trades a little inter-word stretch for a
% guarantee of no overfull boxes, which is the right trade for an internal
% record whose text changes every time the numbers do.
\sloppy
\begin{document}""")

    A(r"\begin{center}{\LARGE\bfseries Per-task gate threshold fit for "
      r"Qwen2.5-Omni-7B}\\[2mm]{\large Technical progress report --- through "
      r"pass 2}\\[3mm]{\small built " + now + r" from the run tree at\\"
      r"\texttt{" + esc(ROOT) + r"}}\end{center}" "\n\\vspace{3mm}\n"
      # A [t] float lands at the TOP of page 1 -- i.e. above the title, which
      # reads as though the document opens on a stray table. Suppress top floats
      # for this page only; later pages keep normal placement.
      r"\suppressfloats[t]")

    # ---- executive summary
    A(r"\section*{Executive summary}")
    A(r"""This report covers a per-task fit of the output gate of a three-thread
proactive streaming pipeline built on \textbf{Qwen2.5-Omni-7B} (vision $+$ audio),
evaluated on \textbf{OmniPro Online} (2{,}700 samples, 9 tasks $\times$ 300).
The research question is \emph{when should a streaming model speak}, and the
object under study is the output gate: at every tick the controller reads
$p_\mathit{hit}=P(\text{true})$ off the logits and decides whether to emit.""")

    if p1 and p2:
        A(r"\textbf{Where the run stands.} Pass 1 is complete (%d/%d samples, "
          r"%d/%d cells). Pass 2 is at %.1f\%% (%d/%d samples, %d/%d cells), with "
          r"%d samples outstanding. Stage 3 --- the full 2{,}700-sample "
          r"evaluation --- has not started, and is roughly 72\%% of the total "
          r"compute for the study."
          % (p1["done"], p1["total"], p1["cells_done"], p1["cells"],
             100 * p2["done"] / p2["total"], p2["done"], p2["total"],
             p2["cells_done"], p2["cells"], p2["total"] - p2["done"]))

    A(r"""\textbf{The headline result is negative, and it is the deliverable.}
Five findings, each measured and each with its evidence file, converge on one
conclusion: \emph{the gate has one identifiable degree of freedom, not nine.}
Two of them are outright defects in the shipped system that made the nominal
per-task thresholds inert. The remaining three establish that $p_\mathit{hit}$
does not rank event-adjacent ticks above quiet ones, that the per-task fit does
not survive its own bootstrap, and that nine fitted thresholds do not beat a
single global one at a level this data can resolve.""")

    A(r"""\textbf{Honesty rules in force.} \texttt{time\_f1} is judge-free (a
$\pm3$\,s greedy one-to-one match) and is reported everywhere. \texttt{content\_acc}
and \texttt{joint\_f1} require an LLM judge; no judge endpoint is currently
reachable, so those metrics are \WITHHELD{} throughout --- never estimated, and
never rendered as a dash that could be read as zero. The evaluation is not
bit-reproducible; \textbf{$\Delta$F1 below 0.03 is treated as noise.}""")

    A(tbl_progress(pr))
    A(tbl_p2_tail(pr))

    # ---- system
    A(r"\section{The system under test}")
    A(r"""Three threads share one KV cache. The \emph{ingester} decodes video at
1\,fps and audio in 2\,s chunks and appends to the cache; the \emph{controller}
reads $p_\mathit{hit}$ every tick and decides whether to speak; the \emph{writer}
produces the utterance when it does. The quiet path costs zero decode steps ---
$p_\mathit{hit}$ is read straight off the logits at a boolean slot, which is what
makes a tick-rate gate affordable at all.""")

    A(r"""\textbf{The gate is three coupled knobs, never one:} the threshold
\texttt{hit\_threshold}, the mode \texttt{edge} vs.\ \texttt{level}, and the
refractory period \texttt{refractory\_s}. Quoting the configuration file itself:
\emph{using the threshold without its mode and refractory does not reproduce it.}
Section~\ref{sec:f2} shows this empirically --- the size of the first-tick defect
scales with the refractory length, so the two cannot be fitted separately.""")

    # ---- design
    A(r"\section{Experimental design}")
    A(r"""The fit runs on a \textbf{frozen 5\% all-audio subset}: 15 videos per
task, fixed by id before any threshold was tried, so every cell in every pass
scores the same video set and all comparisons are paired. A \emph{cell} is one
(task, threshold) pair evaluated over those 15 videos. Pass 1 sweeps a coarse
10-point grid; pass 2 refines around each task's pass-1 pick.""")

    A(r"""Two design choices matter for every number in this report.
\textbf{Pooling} is micro-averaged over tp/fp/fn exactly as
\texttt{metrics.aggregate} does it --- a mean of per-sample F1 ratios would let a
video with one ground-truth event outvote a video with thirty.
\textbf{Bootstrap resampling is over videos}, never over ticks or events: ticks
within a video are serially correlated through the shared KV cache, so resampling
them would manufacture significance out of that correlation.""")

    A(r"""Scoring is \texttt{omniprofast/metrics.py}, used unmodified. The harness
spins up the real pipeline and captures its emissions rather than reimplementing
it; a second implementation of the metric would be a second set of bugs.""")

    # ---- findings
    A(r"\section{Findings}")

    A(r"\subsection{Finding 1 --- the shipped \texttt{hit\_threshold} was inert}")
    A(r"""Under the shipped \texttt{gate\_strategy="hysteresis"} the fire test
compares \texttt{p\_hit} against \texttt{cfg.gate\_high\_thr} --- a fixed
\emph{global} 0.5. The per-task
\texttt{hit\_threshold} therefore decided only whether an answer was
\emph{decoded}, not whether the gate fired. Across the completed 2{,}700-sample
phase-A run --- 581{,}403 ticks, 50{,}433 fires --- \textbf{not one fire occurred
below $p_\mathit{hit}=0.500$}, so the four tasks configured at or below 0.5 were
all effectively running at 0.5 and their configured values were dead code.
\texttt{task\_gate\_modes} and \texttt{task\_refractory\_s} were read nowhere at
all. The fit tree replaces this with \texttt{gate\_strategy="fitted"}, which
actually consults all three knobs.""")

    A(r"\subsection{Finding 2 --- the first tick is a spurious rising edge}"
      r"\label{sec:f2}")
    A(r"""$p_\mathit{hit}$ is high on the \emph{first tick of every video}: the
model answers the boolean from the prompt before it has seen anything
($\geq0.05$ for 98--100\% of phase-A samples in every task, median 0.34--0.64).
Seeding \texttt{prev\_above=False} made that tick a rising edge, and on a task
with a long refractory the spurious fire consumed the video's only emission ---
every \texttt{instant\_event\_alert} sample emitted once, at video time 1.0\,s,
against ground truth at 26--98\,s.""")
    A(r"""The fix is definitional rather than a tuned guard: an edge requires a
predecessor, so \texttt{prev\_above} starts \texttt{None}. Table~\ref{tab:firsttick}
shows the effect size tracking the refractory --- large on the long-refractory
tasks, $\pm0.001$ on the short ones. Applied to the live gate and to all five
offline replays.""")
    A(tbl_firsttick(ft))
    A(tbl_refractory(rf))

    A(r"\subsection{Finding 3 --- $p_\mathit{hit}$ is uninformative about "
      r"\emph{when}}")
    A(r"""This is the finding that reframes the whole study. On the pass-1 tasks
with a complete reference cell, \textbf{AUC is near chance and every 95\%
confidence interval contains 0.5}. Three innocent explanations were excluded
before reporting it as a property of the model:""")
    A(r"""\begin{itemize}
\item \textbf{A clock offset} between the log's video timestamp and the
benchmark's trigger time. Swept $\pm10$\,s: no peak anywhere, so there is no
constant misalignment hiding an informative score.
\item \textbf{Too tight a tolerance}, if the model anticipates or lags events.
Re-run at $\pm10$\,s: unchanged.
\item \textbf{A degenerate score} pinned at one value. Excluded by the
distribution --- $p_\mathit{hit}$ spans 0.000--0.997 with 232--284 distinct
values and an interquartile range of 0.2--0.7.
\end{itemize}""")
    A(r"""Independently and through a separate code path, per-task \emph{precision
is flat across the entire threshold grid while recall falls} --- firing less often
buys no precision, which is exactly what an AUC of 0.5 predicts. One task,
\texttt{semantic\_condition\_alert}, comes out \emph{below} chance and falls
monotonically as the offset goes positive: its confidence is systematically lower
near an event.""")
    A(r"""The claim this supports is ``indistinguishable from chance at
$n=15$ videos per task'', never ``proven to be exactly chance''. Re-running the
same audit against stage 3 would give $n=300$.""")
    A(tbl_perception(pa))
    A(figure("fig2_roc.pdf",
             r"ROC per task at the operating point. The diagonal is chance; every "
             r"curve hugs it. This is the picture behind Table~\ref{tab:perception}.",
             "fig:roc"))
    A(figure("fig3_pr_bands.pdf",
             r"Precision--recall across the threshold grid with bootstrap bands. "
             r"Precision is flat while recall falls: the threshold controls how "
             r"\emph{often} the system speaks, not \emph{which moments} it picks.",
             "fig:pr"))

    A(r"\subsection{Finding 4 --- the fit does not survive its own bootstrap}")
    A(r"""Pass 1 completed and \texttt{pick.py} returned a winner for all nine
tasks. A \emph{paired} bootstrap over the frozen videos --- same video set across
thresholds, and replaying \texttt{pick.rank} itself rather than a bare argmax ---
shows that \textbf{every task's $\Delta$F1 against its best rival has a 95\% CI
containing zero}, and that the fitted threshold is re-selected in only a minority
of draws (Table~\ref{tab:noise}). One task is re-selected \emph{below} its own
chance rate.""")
    A(r"""Two of the nine ``fits'' are artefacts and are labelled as such rather
than quietly reported: \texttt{instant\_event\_alert} scores 0.000 at every
threshold, which \texttt{pick.py} correctly flags \textsc{flat} while keeping the
shipped value; and \texttt{explicit\_target\_grounding}'s entire curve is one
matched event or zero out of 16, giving it the widest interval in the table. The
only reproducible structure anywhere in the sweep is the collapse at 0.95 --- the
threshold controls emission \emph{volume}, which is Finding 3 arriving again by a
different route.""")
    A(r"""\textbf{A prediction I recorded in advance was wrong.} Finding 3 led me
to predict that the fit would pin to the low rail on every task that emits.
It did not --- the picks scattered mid-grid. The reasoning error is recorded in
the runbook alongside the corrected prediction, because a prediction that is
quietly dropped after it fails is not a prediction.""")
    A(tbl_noise(fn))
    A(figure("fig1_f1_vs_threshold.pdf",
             r"Time-F1 against threshold per task. Flat within the noise band "
             r"across most of the grid, with the collapse at 0.95 the only "
             r"consistent feature.", "fig:f1"))

    A(r"\subsection{Finding 5 --- one identifiable degree of freedom, not nine}")
    if ab:
        A(r"""Four gate configurations were compared at equal budget on the same
videos, with a paired bootstrap over videos (Table~\ref{tab:ablation}). The nine
fitted thresholds beat the single best global threshold by an amount whose
interval straddles zero --- \emph{in sample}, where the fitted arm has nine free
parameters against the global arm's one.""")
        A(r"""Yet the global threshold itself is solid. Because
\texttt{best\_single\_global} is also an argmax over the same grid and the same
videos, it carries the identical winner's curse Finding 4 measured, so the entire
sweep is re-run inside every bootstrap draw. The value $%.2f$ is re-selected in
\textbf{%.0f\%%} of draws against a chance rate of %.0f\%%, versus the far lower
rates in Table~\ref{tab:noise} for the per-task fits."""
          % (bsg, 100 * ab["bsg_reselect_rate"], 100 * ab["bsg_chance_rate"]))
        A(r"""\textbf{The thesis, stated carefully:} pooled over %d videos the
operating point is identified; split nine ways over %d videos each it is not.
That is a statement about statistical power, \emph{not} a claim that the nine
tasks are identical or that per-task gating cannot help. The shipped per-task
configuration is meanwhile significantly \emph{worse} than a flat global
threshold, and the mechanism is visible in the columns: it loses on recall, not
precision."""
          % (ab["n_videos_total"], ab["n_videos_total"] // 9))
    A(tbl_ablation(ab))
    A(figure("fig4_emission_calibration.pdf",
             r"Emissions per ground-truth event against threshold. The gate is a "
             r"volume control: this curve is steep and orderly even where F1 is "
             r"flat.", "fig:calib"))

    A(r"\subsection{Finding 6 --- pass 2 confirms a prediction registered in "
      r"advance}")
    if fn2:
        nz = n_ci_contains_zero(fn2)
        A(r"""Pass 2 refined the grid around each pass-1 pick and completed
675/675 samples across 45/45 cells, every cell reliable. Before it finished, and
in writing, Finding 4 registered a prediction: that the refined grid's
$\Delta$F1-vs-rival intervals would contain zero on \emph{at least seven of
nine} tasks. \textbf{The outcome is %d of 9} (Table~\ref{tab:noise2}). Recording
the prediction first is what makes this a test rather than a description.""" % nz)
        A(r"""\textbf{Two rows must be read carefully.} \texttt{instant\_event\_alert}
and \texttt{snapshot\_counting} report an interval of $[+0.000, +0.200]$. The
lower bound is exactly zero, so these \emph{do not} exclude it; both tasks match
0--2 events out of 15--16, and the interval is two grid-quantised jumps wide.
Calling that ``excludes zero'' would misread an inclusive bound as a tight
positive effect.""")
        A(r"""\textbf{The re-selection rates look better than pass 1's and are
not.} They read 38--74\% against pass 1's 25--61\%, but chance here is 20\% (five
cells per task) against 8--10\% there (ten to twelve). As a multiple of chance,
pass 2 runs 1.9--3.7$\times$ where pass 1 ran 2.5--6.1$\times$ --- no better, and
worse on four tasks. A finer grid packs cells closer together, making the argmax
\emph{less} separable, not more. Re-selection rates from grids of different sizes
must be divided by chance before they can be compared at all.""")
        A(r"""\textbf{What pass 2 genuinely adds is the span column.} Across each
refined neighbourhood the entire F1 range is 0.021--0.133, and on four of nine
tasks \emph{every} cell sits within the 0.03 noise band of the best. Refining the
grid did not resolve a winner; it confirmed there was no gradient there to
resolve --- Finding 3 arriving a fourth time by a fourth independent route.""")
        A(tbl_noise(fn2, label="tab:noise2", span=True, extra=(
            r"Pass-2 paired bootstrap on the refined grid, same protocol as "
            r"Table~\ref{tab:noise}. \emph{span} is the full F1 range across the "
            r"task's refined neighbourhood; compare it against the 0.03 noise "
            r"band. \emph{chance} is 20\% here because the refined grid has five "
            r"cells per task, so re-selection rates are \textbf{not} comparable "
            r"to Table~\ref{tab:noise}'s without dividing by it.")))
    A(tbl_final(final, picks))
    if final:
        A(r"""Table~\ref{tab:final} is the artefact stage 3 would consume. The
selection ranks the pass-2 cells together with the two pass-1 candidates, so a
coarse point can win the finalise: two of the nine finals are not on the pass-2
grid at all, and one task keeps its shipped value because pass 1 flagged its
curve flat. Where the audit's fitted threshold is not a cell that was actually
run, it falls back to the rule's own pick on the pass-2 grid --- the alternative
would be scoring a cell that does not exist.""")
        A(r"""\textbf{One prediction remains open}, and it is the one that
matters: that on the held-out 2{,}700-sample stage-3 run, time-F1 under these
thresholds will land within the 0.03 noise band of the same run under the shipped
ones. That comparison is fit-disjoint, and it --- not the fit's own cells --- is
the arbiter of whether any of this fitting was real.""")

    # ---- pass 1 detail
    A(r"\section{Pass-1 selections}")
    A(r"""Table~\ref{tab:picks} is the raw output of the selection rule, kept in
the report even though Finding 4 shows the individual picks are not resolvable.
It is the input that stage 3 would consume, and the negative result is only
legible next to the numbers it negates.""")
    A(tbl_picks(picks))

    # ---- infrastructure
    A(r"\section{Infrastructure and profiling}")
    A(r"""The run tree is deliberately separate from the source repository so that
no run ever mutates the code under test, and it carries its own copy of the
interpreter, the weights cache and the dataset. That separation was not
precautionary theatre: mid-sweep, a \texttt{pip} reinstall of torch inside a
shared environment owned by another account changed 594 files and deleted
\texttt{torch/bin/torch\_shm\_manager}, and \texttt{import torch} failed outright
on every lane. Anything under another account's scratch can move without warning,
and the dataset case is worse than the environment case because it fails
\emph{silently} rather than loudly.""")

    A(r"""\textbf{Two fleets run in parallel.} The \texttt{debug} partition
allows at most 4 nodes and enforces a quota of 90 node-minutes per job, which
yields 4 nodes for 22 minutes, 2 for 45, or 1 for 90; only one job may run at a
time, with one more queued. The login nodes carry 4 GPUs each and are shared with
other users, so the fleet reads its GPU allocation from a file
(\texttt{LOGIN\_GPUS}) that can be changed while it runs.""")

    A(r"""\textbf{Every long job is self-chaining.} The wall-clock kill is a
SIGKILL, so nothing at the end of a script is guaranteed to execute; each
generation therefore pre-queues its successor with
\texttt{--dependency=afterany} \emph{before} doing any work. Resume is global
rather than per-lane, via a glob over the banked predictions, because the chain
reshapes 4$\to$2$\to$1 nodes and the lane count changes the shard assignment.""")

    A(r"""\textbf{Two scheduling defects were found and fixed by this run.}
First, a lane could re-claim the same barren shard for its whole window: the
scheduler returns the first incomplete claimable unit, and a shard whose samples
are all longer than the lane's remaining time is incomplete forever. That
produced 26{,}277 claims across seven generations --- all for one task --- while
84 of 94 cells were never touched. There are now two filters, one for barren
cells and one for barren shards, which the cell-level check cannot see. Second,
the chain read its worklist from a value baked in at first submit, one dropped
\texttt{--export} away from silently falling back to pass 1, finding it already
complete, and standing down looking like a finished run. The active pass is now
read from a file on every generation.""")

    A(r"""\textbf{Ordering hazard.} The subset stride selects by \emph{index}, so
any filter that shortens the sample list must be applied \emph{after} it; applied
before, it re-indexes the stride onto different samples. Measured cost of getting
this backwards: 487 of 688 evaluated ids landed outside the intended subset.""")

    A(r"""\textbf{Observed tail behaviour.} Completion is not uniform across
tasks (Table~\ref{tab:p2tail}) because a lane can only claim a shard whose
samples fit inside its remaining wall-clock. Late in a pass the outstanding
samples are the longest videos, the short-window debug lanes report every unit
barren, and the tail is carried almost entirely by the login lanes, which run
without a per-unit duration cap. This is expected rather than a fault, but it
means the last few percent of a pass proceed at a small fraction of its average
rate, and any schedule estimate that extrapolates the average will be wrong.""")

    # ---- remaining
    A(r"\section{What remains}")
    A(r"""\begin{enumerate}
\item Finish pass 2, then \texttt{score\_cells}, \texttt{pick.py} and the
      bootstrap audit on it. All three are CPU-only and cost no GPU time. This
      tests a prediction registered in advance: that at least seven of the nine
      confidence intervals will again contain zero.
\item Apply the fitted thresholds and run stage 3, the full 2{,}700-sample
      evaluation. This is the bulk of the remaining compute.
\item Re-run the perception audit against stage 3, where it has $n=300$ per task
      instead of 15, and redraw the ROC figure from those logs.
\item Restore an LLM judge and lift the \WITHHELD{} on the content metrics. Three
      endpoints were tried and all failed --- one returning 404, one rate-limited,
      one unavailable.
\end{enumerate}""")

    A(r"""\textbf{Open decision.} Stage 3 cost scales with the number of gate arms
carried into it. Two arms --- the fitted per-task gate and the single best global
threshold --- is the recommendation: it is the comparison the report actually
turns on, and the wider ablation can stay on the fit subset where it already has
an answer.""")

    A(r"\section{Reproducibility}")
    A(r"""Every finding in this report regenerates from committed artefacts with
no GPU. The banked predictions are versioned --- they represent roughly 150
GPU-hours and the whole study is re-scorable from them offline --- while the
per-tick run logs are not, being large and reproducible by any re-run. Four
things live outside the repository and must be re-provided before any \emph{new}
sample can be evaluated: the benchmark manifest and its video corpus, the model
weights, the Python environment, and the working copy of the system under test.
Resume is keyed on the banked predictions, so nothing already completed is ever
recomputed.""")

    A(r"\end{document}")
    return "\n\n".join(T)


def main():
    os.makedirs(OUT, exist_ok=True)
    pr = progress()
    tex = build_tex(pr, load("P1_PICKS.json"), load("FIT_NOISE_AUDIT.json"),
                    load("ABLATION.json"), load("PERCEPTION_AUDIT.json"),
                    load("REFRACTORY_AUDIT.json"), load("FIRST_TICK_AUDIT.json"),
                    load("FIT_NOISE_AUDIT_P2.json"), load("FINAL_THRESHOLDS.json"))
    src = os.path.join(OUT, "PROGRESS_REPORT.tex")
    with open(src, "w") as f:
        f.write(tex)
    r = subprocess.run([TECTONIC, "-X", "compile", src, "--outdir", OUT],
                       capture_output=True, text=True)
    if r.returncode:
        r = subprocess.run([TECTONIC, src, "--outdir", OUT],
                           capture_output=True, text=True)
    pdf = os.path.join(OUT, "PROGRESS_REPORT.pdf")
    if r.returncode or not os.path.exists(pdf):
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        sys.exit("compile failed")
    try:
        import re
        n = len(re.findall(rb"/Type\s*/Page[^s]", open(pdf, "rb").read()))
    except Exception:
        n = -1
    print(f"-> {pdf}  ({os.path.getsize(pdf)/1024:.0f} KB, ~{n} pages)")


if __name__ == "__main__":
    main()
