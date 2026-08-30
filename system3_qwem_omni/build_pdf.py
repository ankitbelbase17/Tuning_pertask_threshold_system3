#!/usr/bin/env python3
"""Render the architecture doc to PDF: rasterise LaTeX math (markdown-pdf/Story
has no math engine) and embed the figures, then lay out with PyMuPDF."""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import mathtext
from markdown_pdf import MarkdownPdf, Section
from markdown_it import MarkdownIt

_MATH_PARSER = mathtext.MathTextParser("agg")

ROOT = os.path.abspath(".")
md = open("SYSTEM3_TECHNICAL_ARCHITECTURE.md", encoding="utf-8").read()

# ---------------------------------------------------------------------------
# 1. Rasterise math.  Story renders HTML with no JS/MathJax, so every $..$ and
#    $$..$$ is pre-rendered to a transparent PNG via matplotlib mathtext and
#    embedded with an explicit per-equation width (Story ignores intrinsic size).
# ---------------------------------------------------------------------------
EQ_DIR = "figures/eq"
os.makedirs(EQ_DIR, exist_ok=True)
eq_css = []
_n = [0]


def _prep(latex):
    # Collapse newlines/whitespace: mathtext cannot span a newline (a multi-line
    # $$..$$ block otherwise fails to parse and matplotlib silently draws it raw).
    s = " ".join(latex.split())
    # mathtext lacks \text / \texttt (supports \mathrm / \mathtt); preserve the
    # spaces inside a text run with \  so words don't get glued together.
    def _txt(m, cmd):
        return cmd + "{" + m.group(1).replace(" ", r"\ ") + "}"
    s = re.sub(r"\\text\{([^}]*)\}", lambda m: _txt(m, r"\mathrm"), s)
    s = re.sub(r"\\texttt\{([^}]*)\}", lambda m: _txt(m, r"\mathtt"), s)
    return s


def render_eq(latex, display):
    n = _n[0]; _n[0] += 1
    tex = _prep(latex)
    fs = 22 if display else 17
    try:
        # matplotlib swallows mathtext ParseExceptions at draw time and renders the
        # raw string, so validate explicitly and fall back to legible code on failure.
        _MATH_PARSER.parse(f"${tex}$")
        fig = plt.figure()
        fig.text(0.5, 0.5, f"${tex}$", fontsize=fs, ha="center", va="center")
        path = f"{EQ_DIR}/q{n}.png"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)
    except Exception as e:
        print(f"  [math] FALLBACK on eq {n}: {str(e)[:80]}")
        return f"`{' '.join(latex.split())}`"  # keep as code so it is at least legible
    plt.close(fig)
    w, h = Image.open(path).size
    if display:
        W = min(515.0, round(w * 0.26, 1))          # cap to content width
        cls = f"qd{n}"
        eq_css.append(f".{cls}{{width:{W}pt; display:inline; margin:0;}}")
        return (f'\n\n<div style="text-align:center; margin:12px 0;">'
                f'<img class="{cls}" src="{path}"/></div>\n\n')
    else:
        W = min(320.0, round(w * 0.19, 1))           # inline: ~body height
        cls = f"qi{n}"
        eq_css.append(f".{cls}{{width:{W}pt; display:inline; margin:0 1pt; "
                      f"vertical-align:-1.5pt;}}")
        return f'<img class="{cls}" src="{path}"/>'


def _sub_math(seg):
    seg = re.sub(r"\$\$(.+?)\$\$", lambda m: render_eq(m.group(1), True),
                 seg, flags=re.DOTALL)
    seg = re.sub(r"\$(.+?)\$", lambda m: render_eq(m.group(1), False), seg)
    return seg


# process math only OUTSIDE fenced code blocks and inline `code` spans
out = []
for part in re.split(r"(```.*?```)", md, flags=re.DOTALL):
    if part.startswith("```"):
        out.append(part)
        continue
    for sp in re.split(r"(`[^`]*`)", part):
        out.append(sp if sp.startswith("`") and sp.endswith("`") else _sub_math(sp))
md = "".join(out)
print(f"  [math] rendered {_n[0]} equations")

# ---------------------------------------------------------------------------
# 2. Figures.
# ---------------------------------------------------------------------------
def img(fname, caption):
    # page-break div guarantees the figure gets full page height (Story otherwise
    # shrinks an image to whatever vertical space is left at the bottom of a page).
    p = f"figures/{fname}"
    return (f'\n\n<div class="pb"></div>\n\n![{caption}]({p})\n\n'
            f'*{caption}*\n\n')


insertions = [
    ("### 2.3 ASCII data-flow diagram",
     "### 2.3 Data-flow diagram" +
     img("fig1_architecture.png",
         "Figure 1. Three-thread streaming pipeline over one shared KV cache. "
         "The only coupling between threads is the cache; the red loop is the fps input gate.") +
     "The rendered schematic above is followed by the literal ASCII diagram from the source:\n"),
    ("### 3.2 KV cache size and memory footprint",
     "### 3.2 KV cache size and memory footprint" +
     img("fig3_kv_memory.png",
         "Figure 3. KV-cache memory model. (a) GQA (8 KV heads) uses 4x less than the "
         "generic 32-head estimate; (b) cache grows linearly to 36 GiB at the 262K budget, "
         "but a typical clip sits near 22%.")),
    ("### 4.2 Generation frequency, latency, and token cost",
     "### 4.2 Generation frequency, latency, and token cost" +
     img("fig4_prompt_tokens.png",
         "Figure 4. Per-tick ICL prompt token cost by task. The SCA task's ~1,037-token "
         "block is re-prefilled every tick (the re-prefill waste of §7.5).")),
    ("### 4.3 Profiling & latency breakdown table",
     "### 4.3 Profiling & latency breakdown table" +
     img("fig2_latency.png",
         "Figure 2. Per-stage latency (log scale). Decode per token dominates; config "
         "assembly is effectively free and config prefill is a modest ~0.2 s/tick.")),
    ("### 5.2 Prompt versioning and system evolution",
     "### 5.2 Prompt versioning and system evolution" +
     img("fig5_versions.png",
         "Figure 5. Prompt-version lineage on SCA (frozen model): time_F1 (bars) against "
         "emit count (line). v2-evidence is best; deltas are within run-to-run noise.")),
    ("### 6.1 Online-mode evaluation",
     "### 6.1 Online-mode evaluation" +
     img("fig6_headtohead.png",
         "Figure 6. Controller vs. probe-gate on 11 SCA videos: statistically tied on F1 "
         "(iso-F1 contours) but at opposite precision/recall operating points.")),
    ("### 7.1 Decode latency is the true bottleneck (not the cache, not config)",
     "### 7.1 Decode latency is the true bottleneck (not the cache, not config)" +
     img("fig7_decode.png",
         "Figure 7. Decode latency: the memory-bound floor, the 1.6x sampling win already "
         "banked, in-pipeline contention, and the 3-4x StaticCache+CUDA-graph lever still blocked.")),
]

for anchor, replacement in insertions:
    assert md.count(anchor) == 1, f"anchor not unique/found: {anchor!r} ({md.count(anchor)})"
    md = md.replace(anchor, replacement, 1)

md = md.replace(
    "## 1. Introduction",
    "## List of Figures\n\n"
    "1. Three-thread streaming pipeline over one shared KV cache (§2.3)\n"
    "2. Per-stage latency breakdown (§4.3)\n"
    "3. KV-cache memory model: GQA correction and growth vs. budget (§3.2)\n"
    "4. Per-tick ICL prompt token cost by task (§4.2)\n"
    "5. Prompt-version F1 progression vs. emit count (§5.2)\n"
    "6. Controller vs. probe-gate operating points (§6.1)\n"
    "7. Decode-latency bottleneck and optimisation levers (§7.1)\n\n"
    "---\n\n## 1. Introduction", 1)

# ---------------------------------------------------------------------------
# 3. Render.
# ---------------------------------------------------------------------------
css = """
.pb { break-before: page; page-break-before: always; }
body { font-family: -apple-system, 'Helvetica Neue', Arial, sans-serif; line-height: 1.5; }
h1 { font-size: 20pt; } h2 { font-size: 15pt; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 12.5pt; } h4 { font-size: 11pt; }
img { width: 100%; display: block; margin: 10px auto; }
em { color: #555; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 4px 7px; }
th { background: #f0f0f4; }
code { background: #f4f4f4; padding: 1px 3px; border-radius: 3px; font-size: 9pt; }
pre { background: #f7f7f7; padding: 8px; border-radius: 4px; font-size: 8pt; overflow-x: auto; }
blockquote { border-left: 3px solid #bbb; margin-left: 0; padding-left: 12px; color: #555; }
""" + "\n".join(eq_css) + "\n"

pdf = MarkdownPdf(toc_level=3)
pdf.m_d = MarkdownIt("commonmark", {"html": True}).enable("table")  # allow raw <img>/<div>
pdf.add_section(Section(md, root=ROOT), user_css=css)
pdf.meta["title"] = "System 3 Technical Architecture"
pdf.meta["author"] = "System_3 (branch icl_ingester_writer)"
out = "SYSTEM3_TECHNICAL_ARCHITECTURE.pdf"
pdf.save(out)
print("saved", out, os.path.getsize(out), "bytes")
