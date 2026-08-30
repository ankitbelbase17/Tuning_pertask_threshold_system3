# ashok_figures — working instructions

Standing instructions for the paper figures. Read this first every session.

## Ground rules (from Dipan, 2026-08-03)

- **All figure work happens in this directory** (`ashok_figures/`). Nothing else.
- **Discuss before drawing.** We talk through the design first. Only when Dipan says
  **"go"** do I produce **one** iteration. Then he edits it and hands it back. Repeat.
- **SVG is the format.** Dipan can open and edit it in Figma and return it to me.
- **Forget `figures/` and `make_figures.py`.** Those were made by a friend, they are not
  good, and we are explicitly not repeating those mistakes. Do not reuse their style,
  layout, or palette.
- **Cluster environment (CSCS Slurm): never run `sudo`.** Installing user-level tools
  (conda/pip/npm --prefix) is fine and allowed. Adding skills for this task is allowed.

## What the figure has to be

One high-level **circuit-style** diagram of a **general architecture for building a
streaming video–language model / omni model**. Intended to be *the go-to recipe*, not a
description of one system.

Requirements:

1. **Surprising at a glance.** Should look striking/dense the way the Sakana CTM figure
   does — colorful, yellow accents, non-generic. Not another gray box-and-arrow chart.
2. **The method must be delivered at a glance.** Someone who reads only this figure
   should get the idea.
3. **Inputs and outputs must be unambiguous.**
4. **The self-reprogrammable-by-an-LLM part must be the visual centerpiece.**
   Components inside the box are reprogrammed by **configs**, and those configs depend on
   the **task** and the **input video at hand**.

## Inspirations (`inspirations.png`)

- **Left — Sakana CTM Fig. 3:** the visual energy target. Dense, hand-drawn feel, grouped
  labeled panels, numbered circles ①–⑩ keyed to the caption, saturated accent colors
  (yellow, cyan, green, magenta) against grayscale texture.
- **Right — VISPROG-style flow:** the legibility target. Clean vertical program-generation
  loop: Program Generator → high-level program (dashed) → Program Interpreter → outputs,
  with inputs and in-context instruction–program pairs entering at the bottom.

We want the **energy of the left** with the **readability of the right**.

## Source sketch (`circuit_0.png`)

Dipan's Excalidraw notes. Two drafts:

- **Top (detailed, older):** streaming video → Vision Encoder → **LLM1** (input ingest,
  has KV cache, *not configurable*) → **LLM2** (thinking, emits Config) → **Config**
  (input density / zoom in–out / input proactivity control, output proactivity, response)
  → output to user. An **Interpreter** sits under the config path and *configures FPS from
  the input, zooms, crops*. Text inputs (question, system prompt, in-context examples)
  feed in at the top. Labeled "Streaming VLM System at t" vs "System at t+1".
- **Bottom (simplified, agreed in the meeting — this is the direction):** one box at
  time *t* containing encoders + LLM, fed by **Video** and **Audio**, with a **config file**
  written into it; "reconfig depending on inputs"; the box is **self-configurable**; the
  same box at *t+1* labeled **Evolution / Self-Programming**. LLM1 is not configurable.

The bottom sketch is the agreed layout; the top sketch is the vocabulary of components.

## 🚫 HOW TO COLLABORATE (Dipan, 2026-08-03) — read before proposing anything

**Build on what we already thought about and drew. Do not invent new schemes.**

The sketch in `circuit_0.png` (bottom draft) is the agreed starting point. My job is to make
*that* clearer, cleaner and more surprising — not to arrive with a fresh concept each round.

- **Do not introduce vocabulary or structure Dipan never said.** ("fast path / slow path",
  "nested rings", ROS-node taxonomies — all invented by me, all rejected. Don't revive them.)
- **Do not dump 5 options.** Give one strong direction, at most one alternative, and say why.
- **Act like a creative collaborator, not an eager AI that thinks everyone is dumb and
  restarts from scratch every turn.**
- Improve by *refining and sharpening* what exists. Earn any new element.

## 🔢 ITERATIONS — keep every one

There will be many iterations and **each one is kept**. Never overwrite a previous one.

```
ashok_figures/
  iterations/iterNN.svg    one file per iteration, numbered, immutable once rendered
  out/iterNN.png           render, ONLY so the figure can be looked at
  build.sh                 bash build.sh          -> all
                           bash build.sh iter03   -> just one
```

**SVG only.** No PDFs, no exports, no paper-ready artifacts — we are nowhere near putting
this in a paper. Keep the directory empty of anything that isn't earning its place.

New iteration = **copy the latest to the next number, then edit the copy.** The highest
number is always the current one. Dipan's own edits (Figma exports, markups) come back in
as the next number too.

| # | what it was | verdict |
|---|---|---|
| iter01 | one box, config card, aperture, pruned frames | rejected — childish, placeholder-line boxes, no ingest LLM, no KV cache |
| iter02 | real text, ingest LLM + KV cache outside the box, mono config | half-empty box, amber knot, cache exiled to corner, **no future on the page** |

## ✅ SETTLED DECISIONS

- **One box, drawn once.** Do not draw the system twice at t and t+1 — it is misleading.
  Time is shown by the stream flowing *through* the single box.
- **Pick a style and stick to it** across all three figures. No restyling per figure.
- Creativity is wanted — but it must come from *sharpening and completing* the existing
  sketch, not from replacing it.

## 📄 THE PAPER

**Title:** *Foresight: Planning Future Perception in Streaming Vision-Language Models*

**Contributions:**
1. **Input pruning + planning** — deciding what to perceive next, and perceiving less of it.
2. **A good config file** — the control artifact the system writes for itself.
3. KV-cache management (compact, summarize) — *explicitly not our problem; keep it minor.*
4. **Training-free future forecasting — and the config is *why* it is possible.**

The through-line is **foresight**: the system plans its own future perception. The figure
must make that the visible idea. (Simultaneity/async is background from MISSION.md — it is
**not** the headline of this paper.)

## ⭐ FRAMING — read this before anything else

**This is a conceptual figure for a grand claim, not a diagram of our code.**

The contribution is the **idea**: a general recipe for giving *any* frozen model the human
property of doing everything at once. The repo is one special-case implementation — an
existence proof that it works. It is **not** the subject of the figure.

Therefore: **no** model names, no file names, no field names, no measured numbers, no
implementation caveats anywhere in this figure. Generic role names only. Keep it **light
and clean** — take *energy and colour* from the CTM inspiration, not its clutter.

The vision, from `MISSION.md` §1:

> Humans **see, listen, think, and speak all at once.** None of those faculties waits for
> another to finish. Every current streaming model breaks this — they alternate, and the
> response blocks perception, so they miss what happens while they are talking.
> We give a frozen model that simultaneity through **architecture, not training**:
> independent, concurrently-running processes that share memory and never block one another.
> The design metaphor is **ROS** — independent nodes, async message passing, a shared
> blackboard, no node waiting on another.

**Consequences for the figure:**

- It must **look concurrent**, not sequential. A left-to-right pipeline is the wrong shape
  and is exactly what every other paper draws.
- **Two speeds must be visible:** a fast reflex that never decodes (*is anything worth
  saying?*) and a slow deliberative process (*what to say, and how to re-aim perception*).
- **The backward edge is the centerpiece:** thinking rewrites perception. Including
  *deferred reasoning* — writing a question to ask oneself later.
- **Modality-agnostic by construction:** any modality is just another encoder node feeding
  the same shared memory. The recipe must show that slot.
- Roles, not products: Encoder · Ingester · Shared Memory · Trigger · Controller ·
  Memory Log · Writer.

The honesty concerns below belong in the **paper text**, not in this figure. Author's call,
already made — do not relitigate.

## Ground truth — what the code actually does (verified 2026-08-03)

Do not draw the sketch literally. Verified against source:

- **There is no LLM1/LLM2.** One frozen checkpoint (`Qwen3-VL-8B-Instruct`), two *threads*
  over **one shared KV cache**. The ingester prefills and never runs `lm_head`; the
  controller takes an MVCC `deepcopy` snapshot, decodes on the throwaway clone, and never
  writes back. Single-writer invariant.
- **Frozen, training-free, forever.** No gradients (MISSION invariant 4). The control
  language is taught by static hand-written ICL, seeded once into a **pinned, un-evictable
  sink** together with the system prompt.
- **The controller is a polling loop**, waking on `next_check_s` clamped to [0.2, 1.5] s of
  video time — discrete ticks, not continuous thought.
- **Decode is a schema walk:** code force-feeds every JSON key and delimiter; the model
  only fills value slots. Booleans are **logit reads** (`P(true)` over `{true,false}`) at a
  forced slot — zero decode steps. 100% valid JSON, 647/647 argmax agreement.
- **No audio.** Video frames only, fixed 1.0 fps in eval.
- **No Interpreter component.** It is `_clamp` + `set_fps` + `set_next_check`, inline.
- **No zoom/crop.** Deliberately deferred.

### The self-reprogramming loop — partly live (corrected)

An earlier note here said the loop never fires. **That was stale** (it came from an audit
predating the current decoder). Measured over 200,415 ticks on the current full run:

- **Self-pacing is live.** The controller sets its own next wake time and it changes the
  real cadence.
- **Deferred reasoning is live.** It writes questions to ask itself later, and they are fed
  back into the next tick's prompt. This is pillar 4 — the capability the rival lacks.
- **Speaking decisions** come from a logit read at a forced position, zero decode steps.
- Frame-rate steering is emitted but the encoder ignores it under deterministic eval; the
  running accumulators are emitted but not yet fed back; the memory-note field is untried.

Enough of the loop is real to call the architecture demonstrated. The rest is future work,
and the figure draws the architecture as proposed regardless — author's call, settled.

## Working loop

```
ashok_figures/
  INSTRUCTIONS.md    this file
  inspirations.png   CTM + VISPROG references
  circuit_0.png      Dipan's excalidraw notes
  <figure>.svg       the source I edit
  build.sh           svg -> png (for review) + pdf (for the paper)
  out/               rendered artifacts
```

- I edit the SVG, render it to PNG, **look at the PNG myself** before showing it.
- Dipan edits in Figma (or marks up the PNG, or says what to change in words) and returns it.
- `rsvg-convert` is available for SVG→PNG and SVG→PDF. No LaTeX on this machine.
