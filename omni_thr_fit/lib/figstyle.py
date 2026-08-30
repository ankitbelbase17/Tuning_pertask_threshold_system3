#!/usr/bin/env python
"""figstyle.py -- the sec.9.4 chart specification, in one place.

Every figure imports this. The point is that the ECCV-grade rules (palette,
rcParams, redundant encoding, column widths) are stated ONCE and cannot drift
between Fig 0 and Fig 5.

THE PALETTE IS THE SPEC'S, NOT AN EYEBALLED ONE. sec.9.4 fixes three roles and
records why: the luminance ladder 0.073 / 0.187 / 0.278 is monotone, so the
figures survive grayscale printing, and all three pairs clear the CVD dE>=8
target. The slot-3 aqua that a default palette would give was rejected there for
sitting at contrast 2.82 on white. Do not substitute colours.
"""
from __future__ import annotations
import matplotlib as mpl

# role -> (hex, linestyle, marker); identity is NEVER colour alone (sec.9.4)
VIOLET = "#4a3aa7"   # series 1 -- time-F1 / precision
BLUE   = "#2a78d6"   # series 2 -- joint-F1 / recall
ORANGE = "#eb6834"   # series 3 -- third series where one is needed
INK    = "#0b0b0b"   # primary text
INK2   = "#52514e"   # secondary text / axis
GRID   = "#e8e8e6"
MUTED  = "#9a9a97"   # chance diagonal, shipped rules -- recessive by design

S1 = dict(color=VIOLET, linestyle="-",  marker="o")
S2 = dict(color=BLUE,   linestyle="--", marker="s")
S3 = dict(color=ORANGE, linestyle=":",  marker="^")

COL_SINGLE = 3.3     # inches, LNCS single column
COL_FULL   = 6.9     # inches, LNCS full width

RC = {
    # Type-3 fonts fail many camera-ready checks; 42 embeds TrueType.
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK2, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
}


def use():
    mpl.rcParams.update(RC)


def panel_grid(fig, n=9):
    """3x3 small multiples. sec.9.3: nine tasks means nine PANELS, not nine
    colours on one axis -- a 9-series categorical palette is not resolvable."""
    return fig.subplots(3, 3, sharex=True)


def tidy(ax):
    ax.tick_params(length=2, width=0.5)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
