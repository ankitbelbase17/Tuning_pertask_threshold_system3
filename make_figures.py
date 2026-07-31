#!/usr/bin/env python3
"""Generate architecture + analysis figures for SYSTEM3_TECHNICAL_ARCHITECTURE.
All numbers are taken directly from the markdown doc's tables/prose."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import os

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ---- palette (theme-neutral, print-friendly) ----
C = dict(
    enc="#4C72B0", ing="#DD8452", ctrl="#55A868", cache="#8172B3",
    accent="#C44E52", grey="#7f7f7f", light="#eaeaf2", ink="#222222",
)
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#444", "axes.linewidth": 0.8,
    "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
})


def box(ax, xy, w, h, text, fc, ec=None, fs=10, tc="white", bold=True):
    ec = ec or fc
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight="bold" if bold else "normal", zorder=3)


def arrow(ax, p0, p1, color=C["ink"], style="-|>", lw=1.8, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=16,
                                 lw=lw, color=color, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))


# =====================================================================
# FIG 1 — System architecture / three-thread data-flow
# =====================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 8.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")

    box(ax, (0.4, 9.4), 3.0, 1.0, "video file\n(PyAV decode)", C["grey"], fs=10)
    box(ax, (4.2, 9.2), 7.2, 1.4,
        "ENCODER THREAD  (vision_stream.py)\ndecode frame -> ViT + merger (embed_frame)\n-> [1, N, H]   N ~ 180-196   (paced to wall clock)",
        C["enc"], fs=9.5)
    box(ax, (4.2, 6.9), 7.2, 1.5,
        "INGESTER THREAD  (input_ingester.py) -- SOLE WRITER\n[+ \"time Xs\"] -> mgr.ingest(embeds)  (want_logits=False)\nmgr.evict() past kv_budget (StreamingLLM, pinned sink)\nclock.set(vt)  -> publishes video time",
        C["ing"], fs=9.5)
    box(ax, (2.0, 4.35), 9.4, 1.65,
        "SHARED KV CACHE  (manager.py : KVCacheManager, DynamicCache)\n"
        "- ONE contiguous linear tensor per layer (NOT block-paged)\n"
        "- two clocks: next_pos (logical RoPE)  vs  phys len (write index)\n"
        "- MVCC: snapshot_clone() = deep copy -> readers hold NO lock",
        C["cache"], fs=9.5)
    box(ax, (2.0, 1.5), 9.4, 2.15,
        "CONTROLLER THREAD  (controller.py) -- the \"System 2\" loop\n"
        "assemble prompt = ICL(task) + deferred-Q + history + cue + \"{\"\n"
        "PREFILL prompt -> DECODE control JSON (greedy, mask EOS, stop on \"}\")\n"
        "apply diff:  fps (input gate) | next_check_s (self-schedule)\n"
        "have_enough_info -> rising-edge gate -> FIRE | answer + event_time_s -> reported[]",
        C["ctrl"], fs=9.3)

    arrow(ax, (1.9, 9.9), (4.2, 9.9))
    arrow(ax, (7.8, 9.2), (7.8, 8.4), color=C["ink"])
    ax.text(8.0, 8.78, "vis_q  (bounded; blocking in det. mode)", fontsize=8.3, color=C["ink"])
    arrow(ax, (6.8, 6.9), (6.8, 6.0), color=C["ing"])
    ax.text(7.0, 6.42, "writes", fontsize=8.3, color=C["ing"])
    arrow(ax, (6.0, 4.35), (6.0, 3.65), color=C["cache"])
    ax.text(6.2, 3.95, "snapshot_clone()  (deepcopy K/V + pos + phys)", fontsize=8.3, color=C["cache"])

    # feedback: controller fps -> encoder (INPUT GATE)
    arrow(ax, (11.4, 2.57), (11.75, 2.57), color=C["accent"], lw=2.0)
    arrow(ax, (11.75, 2.57), (11.75, 9.9), color=C["accent"], lw=2.0)
    arrow(ax, (11.75, 9.9), (11.4, 9.9), color=C["accent"], lw=2.0)
    ax.text(11.9, 6.2, "set_fps()  (INPUT GATE)", fontsize=8.6, color=C["accent"],
            rotation=90, va="center", weight="bold")

    # FIRE output
    arrow(ax, (2.0, 2.1), (0.6, 2.1), color=C["ctrl"], lw=2.0)
    ax.text(0.5, 2.4, "answer\n(OUTPUT\nto user)", fontsize=8.6, color=C["ctrl"],
            ha="left", weight="bold")

    ax.text(6, 10.75, "Figure 1  Three-thread streaming pipeline over one shared KV cache",
            ha="center", fontsize=12.5, weight="bold", color=C["ink"])
    ax.text(6, 0.55,
            'Only coupling = the shared cache. Encoder never touches it; ingester is sole writer; controller reads MVCC snapshots.\n'
            '"Async state" here = (a) MVCC deepcopy snapshot + (b) in-memory reported[] history.  NO GPU->CPU offload, NO disk/LMCache/3FS.',
            ha="center", fontsize=8.2, color=C["grey"], style="italic")
    fig.savefig(f"{OUT}/fig1_architecture.png"); plt.close(fig)


# =====================================================================
# FIG 2 — Per-component latency & GPU overhead (from the §4.3 table)
# =====================================================================
def fig_latency():
    comps = ["Config assembly\n(string build)", "Frame ingest\n(want_logits=F)",
             "KV snapshot\n(deepcopy @24K)", "Config prefill\n(~1037 ICL tok)",
             "Decode / token\n(in-pipeline)"]
    lat = [0.5, 4, 12, 215, 112]          # ms, midpoints from the doc
    kind = ["CPU", "compute", "memory", "compute", "memory/launch"]
    kcol = {"CPU": C["grey"], "compute": C["enc"], "memory": C["cache"],
            "memory/launch": C["accent"]}
    colors = [kcol[k] for k in kind]

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    y = np.arange(len(comps))
    ax.barh(y, lat, color=colors, edgecolor="#333", height=0.62, zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y); ax.set_yticklabels(comps, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("latency  (ms, log scale)")
    ax.set_xlim(0.3, 600)
    labels = ["< 1 ms", "~few ms/frame", "5-18 ms", "~200-230 ms/tick", "~105-120 ms/tok"]
    for yi, v, lab in zip(y, lat, labels):
        ax.text(v * 1.15, yi, lab, va="center", fontsize=9, color="#111")
    ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
    handles = [Line2D([0], [0], marker="s", ls="", mec="#333", mfc=kcol[k], ms=10, label=k)
               for k in kcol]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5, title="bound by",
              title_fontsize=8.5, framealpha=0.95)
    ax.set_title("Figure 2  Per-stage latency  —  decode dominates, config assembly is free",
                 fontsize=12, weight="bold", pad=10)
    fig.savefig(f"{OUT}/fig2_latency.png"); plt.close(fig)


# =====================================================================
# FIG 3 — KV cache memory: GQA correction + growth vs budget
# =====================================================================
def fig_kv_memory():
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.4),
                                 gridspec_kw={"width_ratios": [1, 1.4]})

    # (a) generic MHA (32 heads) vs GQA (8 kv heads) at full budget
    per_tok_mha = 2 * 36 * 32 * 128 * 2 / 1024        # KiB
    per_tok_gqa = 2 * 36 * 8 * 128 * 2 / 1024
    bars = a0.bar(["generic H=32\n(overestimate)", "GQA H_kv=8\n(faithful)"],
                  [per_tok_mha, per_tok_gqa],
                  color=[C["grey"], C["ctrl"]], edgecolor="#333", zorder=3)
    a0.set_ylabel("KV memory  (KiB / token)")
    a0.bar_label(bars, labels=[f"{per_tok_mha:.0f}", f"{per_tok_gqa:.0f}"],
                 padding=3, fontsize=10, weight="bold")
    a0.annotate("4x", xy=(0.5, per_tok_gqa + 60), fontsize=15, color=C["accent"],
                weight="bold", ha="center")
    a0.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    a0.set_title("(a) per-token footprint", fontsize=10.5, weight="bold")
    a0.set_ylim(0, per_tok_mha * 1.25)

    # (b) memory vs sequence length
    budget = 262144
    S = np.linspace(0, budget, 400)
    gib = per_tok_gqa * 1024 * S / (1024 ** 3)
    a1.fill_between(S / 1000, gib, color=C["cache"], alpha=0.25, zorder=1)
    a1.plot(S / 1000, gib, color=C["cache"], lw=2.2, zorder=2)
    a1.axhline(36, ls="--", color=C["accent"], lw=1.3)
    a1.text(5, 36.8, "36 GiB @ 262K budget", color=C["accent"], fontsize=9)
    typ = 0.22 * budget
    a1.axvline(typ / 1000, ls=":", color=C["ing"], lw=1.6)
    a1.scatter([typ / 1000], [per_tok_gqa * 1024 * typ / (1024 ** 3)], color=C["ing"],
               zorder=5, s=45)
    a1.text(typ / 1000 + 4, 6.5,
            "typical OmniPro clip\n~22% budget (300s @1fps)\n=> eviction rarely fires",
            color=C["ing"], fontsize=8.6)
    a1.set_xlabel("sequence length  (thousands of tokens)")
    a1.set_ylabel("KV cache size  (GiB, bf16)")
    a1.set_xlim(0, budget / 1000); a1.set_ylim(0, 40)
    a1.grid(ls=":", alpha=0.5)
    a1.set_title("(b) growth vs. kv_budget", fontsize=10.5, weight="bold")

    fig.suptitle("Figure 3  KV-cache memory model  (Qwen3-VL-8B text decoder: L=36, H_kv=8, D=128, bf16)",
                 fontsize=12, weight="bold", y=1.02)
    fig.savefig(f"{OUT}/fig3_kv_memory.png"); plt.close(fig)


# =====================================================================
# FIG 4 — ICL prompt token cost per task (§4.2)
# =====================================================================
def fig_prompt_tokens():
    labels = ["system_prompt\n(seed, once)", "writer_prompt\n(probe arm)",
              "generic\ncontroller", "ETG ICL\n(grounding)", "IEA ICL\n(instant)",
              "SCA ICL\n(semantic)"]
    toks = [31, 42, 463, 707, 970, 1037]
    colors = [C["grey"], C["grey"], C["enc"], C["ctrl"], C["ctrl"], C["accent"]]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    bars = ax.bar(labels, toks, color=colors, edgecolor="#333", zorder=3)
    ax.bar_label(bars, padding=3, fontsize=9.5, weight="bold")
    ax.set_ylabel("approx. tokens  (re-prefilled every tick)")
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    ax.set_ylim(0, 1180)
    ax.set_title("Figure 4  Per-tick ICL prompt token cost  —  SCA task is the heaviest (~1,037 tok)",
                 fontsize=12, weight="bold", pad=10)
    ax.text(5, 1090, "re-prefill waste\n(unexploited KV reuse, §7.5)", fontsize=8.3,
            color=C["accent"], ha="center", style="italic")
    fig.savefig(f"{OUT}/fig4_prompt_tokens.png"); plt.close(fig)


# =====================================================================
# FIG 5 — Prompt-version F1 progression + emit count (§5.2)
# =====================================================================
def fig_versions():
    vers = ["v0\nbaseline", "v1\nedge", "v2\nevidence", "v3\n+example", "v2best\nfiner grid"]
    f1 = [0.115, 0.000, 0.255, 0.150, 0.051]
    emits = [20, 3, 15, 8, 7]
    x = np.arange(len(vers))
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    bars = ax.bar(x, f1, width=0.55, color=[C["grey"], C["grey"], C["ctrl"], C["grey"], C["grey"]],
                  edgecolor="#333", zorder=3)
    ax.bar_label(bars, labels=[f"{v:.3f}" for v in f1], padding=3, fontsize=9.5, weight="bold")
    ax.set_ylabel("time_F1", color=C["ctrl"])
    ax.set_ylim(0, 0.31)
    ax.set_xticks(x); ax.set_xticklabels(vers, fontsize=9)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)

    ax2 = ax.twinx()
    ax2.plot(x, emits, "-o", color=C["accent"], lw=1.8, ms=7, zorder=4)
    for xi, e in zip(x, emits):
        ax2.text(xi, e + 0.6, str(e), color=C["accent"], fontsize=8.5, ha="center")
    ax2.set_ylabel("emits", color=C["accent"])
    ax2.set_ylim(0, 24)

    ax.annotate("best (~2.2x baseline)", xy=(2, 0.255), xytext=(2.4, 0.285),
                fontsize=9, color=C["ctrl"], weight="bold",
                arrowprops=dict(arrowstyle="->", color=C["ctrl"]))
    ax.set_title("Figure 5  Prompt-version lineage (SCA, frozen model)  —  F1 (bars) vs emit count (line)",
                 fontsize=11.5, weight="bold", pad=10)
    ax.text(0.02, -0.24, "Caveat (§6.2): these F1 deltas fall within run-to-run noise; the durable "
            "wins are edge-in-code + look-before-judge, not the exact ranking.",
            transform=ax.transAxes, fontsize=8, color=C["grey"], style="italic")
    fig.savefig(f"{OUT}/fig5_versions.png"); plt.close(fig)


# =====================================================================
# FIG 6 — Head-to-head precision/recall operating points (§6.1)
# =====================================================================
def fig_headtohead():
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    # iso-F1 contours
    P = np.linspace(0.01, 1, 300); R = np.linspace(0.01, 1, 300)
    PP, RR = np.meshgrid(P, R)
    F1 = 2 * PP * RR / (PP + RR)
    cs = ax.contour(PP, RR, F1, levels=[0.1, 0.2, 0.29, 0.4, 0.6],
                    colors="#bbb", linewidths=0.9, zorder=1)
    ax.clabel(cs, fmt="F1=%.2f", fontsize=7.5)

    ax.scatter([0.300], [0.281], s=280, color=C["ctrl"], edgecolor="#222", zorder=5,
               label="ICL Controller (30 emits)")
    ax.scatter([0.181], [0.844], s=280, color=C["accent"], edgecolor="#222", zorder=5,
               marker="D", label="Probe-Gate (149 emits)")
    ax.annotate("precise & restrained\ntime_F1 0.290 | joint_F1 0.185", (0.300, 0.281),
                xytext=(0.34, 0.20), fontsize=8.5, color=C["ctrl"])
    ax.annotate("high recall, low precision\ntime_F1 0.298 | joint_F1 0.181", (0.181, 0.844),
                xytext=(0.24, 0.90), fontsize=8.5, color=C["accent"])

    ax.set_xlabel("time precision"); ax.set_ylabel("time recall")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(ls=":", alpha=0.4)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title("Figure 6  Head-to-head (11 SCA videos)  —  tied on F1,\nopposite operating points",
                 fontsize=12, weight="bold", pad=10)
    fig.savefig(f"{OUT}/fig6_headtohead.png"); plt.close(fig)


# =====================================================================
# FIG 7 — Decode latency: floor, wins banked, remaining lever (§7.1)
# =====================================================================
def fig_decode():
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    stages = ["mem-bound\nfloor", "GPU-resident\nsampling (orig)",
              "GPU-resident\nsampling (fix)", "isolated\nlang-only",
              "in-pipeline\n(contention)", "target:\nStaticCache+\nCUDA graphs"]
    ms = [5, 149, 94, 45, 112, 13]
    colors = [C["grey"], C["accent"], C["ing"], C["enc"], C["accent"], C["ctrl"]]
    bars = ax.bar(stages, ms, color=colors, edgecolor="#333", zorder=3)
    ax.bar_label(bars, labels=[f"{v}" for v in ms], padding=3, fontsize=9.5, weight="bold")
    ax.set_ylabel("ms / token")
    ax.set_ylim(0, 170)
    ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
    # win annotations
    ax.annotate("", xy=(2, 100), xytext=(1, 155),
                arrowprops=dict(arrowstyle="->", color=C["ctrl"], lw=1.8))
    ax.text(1.5, 158, "1.6x banked", color=C["ctrl"], fontsize=9, weight="bold", ha="center")
    ax.annotate("", xy=(5, 20), xytext=(3, 52),
                arrowprops=dict(arrowstyle="->", color=C["ctrl"], lw=1.8, ls="--"))
    ax.text(4.3, 55, "3-4x remaining\n(blocked: DynamicCache\nvs CUDA graphs)",
            color=C["ctrl"], fontsize=8.3, ha="center")
    ax.set_title("Figure 7  Decode latency  —  the true bottleneck (§7.1)",
                 fontsize=12, weight="bold", pad=10)
    fig.savefig(f"{OUT}/fig7_decode.png"); plt.close(fig)


for f in (fig_architecture, fig_latency, fig_kv_memory, fig_prompt_tokens,
          fig_versions, fig_headtohead, fig_decode):
    f()
    print("ok", f.__name__)
print("DONE")
