#!/usr/bin/env python
"""build_pdf.py -- assemble and compile the sec.9 deliverable.

  python lib/build_pdf.py            # -> paper/paper.pdf, reports the page count

WHY THIS IS NOT MODELLED ON system_3/build_pdf.py, despite sec.9 saying so.
That script is a markdown -> PDF renderer (markdown_pdf/Story + PyMuPDF) that
rasterises every equation to PNG. It emits no LaTeX at all, and neither of its
two core dependencies is installed here. sec.9's OUTPUT requirements -- Springer
`llncs.cls`, ECCV camera-ready, 12 pages, "vector PDF from matplotlib (never PNG
-- LNCS is print)" -- cannot be met by it, and its PNG-rasterising approach is
the very thing sec.9.4 forbids. The output spec wins over the implementation
reference; see THRESHOLD_FIT_RUNBOOK sec.7.1.

STRUCTURE. Prose lives in paper/sections/*.tex and is authored by hand. Tables
and figure includes are GENERATED here, so a table can never drift from the JSON
it came from, and a figure that was not produced cannot be silently omitted --
it becomes a visible framed note instead.
"""
from __future__ import annotations
import argparse, glob, os, re, subprocess, sys

ROOT = os.environ["THR_ROOT"]
PAPER = os.path.join(ROOT, "paper")
TECTONIC = os.environ.get("TECTONIC",
                          "/iopsstor/scratch/cscs/dthapa/tools/bin/tectonic")

SECTIONS = ["intro", "related", "method", "protocol", "results",
            "discussion", "limitations", "conclusion"]

PREAMBLE = r"""\documentclass[runningheads]{llncs}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage[hidelinks]{hyperref}
% WITHHELD is a first-class value in this paper, not a missing one: with no
% reachable judge, content metrics are withheld rather than guessed, and a dash
% would read as zero. See EVAL_PROTOCOL.md.
\newcommand{\WITHHELD}{\textsc{withheld}}
\newcommand{\PENDING}{\textsc{[pending run]}}
\begin{document}
\title{When Should a Streaming Model Speak?\\
Per-Task Gate Thresholds for Qwen2.5-Omni on OmniPro Online}
\author{}
\institute{}
\maketitle
"""


def fig(name, caption, label, width=r"\linewidth"):
    """Include a figure, or say plainly that it was not produced."""
    fp = os.path.join(ROOT, "figs", name)
    if not os.path.exists(fp):
        # Plain concatenation, not an f-string: LaTeX brace nesting plus f-string
        # brace doubling is how the first version emitted an unbalanced \fbox and
        # took the whole document down with "File ended while scanning use of".
        return (r"\begin{center}\fbox{\parbox{0.9\linewidth}{\textbf{"
                + label + r" not yet produced}}}\end{center}" + "\n")
    return "\n".join([r"\begin{figure}[t]", r"\centering",
                      rf"\includegraphics[width={width}]{{{fp}}}",
                      rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                      r"\end{figure}", ""])


def section(name):
    fp = os.path.join(PAPER, "sections", f"{name}.tex")
    if os.path.exists(fp):
        return f"\\input{{{fp}}}\n"
    return (f"% section {name} not written yet\n"
            f"\\section{{{name.title()}}}\n"
            f"\\textit{{[{name} not yet written]}}\n")


def assemble():
    T = os.path.join(PAPER, "tables")
    def tbl(n):
        fp = os.path.join(T, f"{n}.tex")
        return f"\\input{{{fp}}}\n" if os.path.exists(fp) else ""
    parts = [PREAMBLE,
             r"\begin{abstract}", section("abstract").replace("\\section{Abstract}", ""),
             r"\end{abstract}", "",
             fig("fig0_teaser.pdf",
                 "Shipped versus fitted per-task gate threshold, sorted by "
                 "$|\\Delta|$.", "fig:teaser"),
             section("intro"), section("related"), section("method"),
             section("protocol"),
             tbl("table1"),
             fig("fig1_f1_vs_threshold.pdf",
                 "F1 against gate threshold, one panel per task. Nine tasks are "
                 "nine panels, not nine colours on one axis. Panel titles carry "
                 "$n$ and $n_{gt}$; all panels share one $y$ scale.",
                 "fig:f1"),
             fig("fig2_roc.pdf",
                 "ROC of per-tick $p_{hit}$ against the $\\pm 3$\\,s positive "
                 "label, with AUC and bootstrap 95\\% CI. AUC is threshold-free "
                 "and therefore unaffected by this fitting: it measures the "
                 "backbone, while F1 measures the gate.", "fig:roc"),
             fig("fig3_pr_bands.pdf",
                 "Per-sample precision and recall against threshold. The band is "
                 "$\\pm 1$ s.d. clipped to $[0,1]$ and is a dispersion "
                 "statistic, not a confidence interval: per-sample precision and "
                 "recall are frequently exactly 0 or 1, so the distribution is "
                 "not Normal. Actual per-sample values are overlaid.",
                 "fig:bands"),
             section("results"), tbl("table2"), tbl("table3"),
             fig("fig4_emission_calibration.pdf",
                 "Emission calibration: emissions per ground-truth event against "
                 "threshold, log axis, reference line at 1.", "fig:calib"),
             fig("fig5_refinement.pdf",
                 "Two-pass refinement: pass-1 grid (open), the refined interval "
                 "(shaded), pass-2 points (filled) and the final choice.",
                 "fig:refine"),
             tbl("table4"),
             section("discussion"), section("limitations"), section("conclusion"),
             r"\end{document}", ""]
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=PAPER)
    ap.add_argument("--pages", type=int, default=12)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tex = os.path.join(a.out, "main.tex")
    with open(tex, "w") as f:
        f.write(assemble())
    print(f"  assembled {tex}")

    if not os.path.exists(TECTONIC):
        sys.exit(f"tectonic not found at {TECTONIC}; see RUNBOOK sec.7.1")
    # --keep-logs: the page count comes from XeTeX's own log, not from guessing at
    # the PDF's internals. A first attempt counted /Type /Page objects with a
    # regex and always reported "?", because XeTeX writes compressed object
    # streams and the markers are not in the byte stream at all.
    r = subprocess.run([TECTONIC, "--keep-logs", "-o", a.out, tex],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        sys.exit("tectonic failed")
    pdf = os.path.join(a.out, "main.pdf")
    n = None
    try:
        with open(os.path.join(a.out, "main.log"), errors="replace") as f:
            m = re.search(r"Output written on .*?\((\d+) pages?", f.read())
        n = int(m.group(1)) if m else None
    except OSError:
        pass
    print(f"  compiled {pdf}  ({os.path.getsize(pdf)//1024} KiB, "
          f"{n if n else '?'} pages)")
    if n and n != a.pages:
        print(f"  NOTE: {n} pages, sec.9.1 budgets {a.pages} excluding references.")


if __name__ == "__main__":
    main()
