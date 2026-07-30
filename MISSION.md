# MISSION — System 3: a training-free cognitive architecture for proactive streaming video

**Target venue:** ICLR 2027 · **Deadline: 18 Sep 2026 (AoE)** · ~7 weeks from 2026-07-30
**Benchmark:** OmniPro ([arXiv 2605.18577](https://arxiv.org/html/2605.18577v1))
**Rival baseline:** MiniCPM-o 4.5 / Omni-Flow ([arXiv 2604.27393](https://arxiv.org/html/2604.27393))
**Scope for v1:** vision-only, `audio_dependency=none`, frozen Qwen3-VL-8B, no training.

> This document is the north star. `ICL_DIFF_CONTROLLER.md` holds the mechanism design,
> `EXPERIENCE.md` the two proactivity systems, `EXPERIMENTS.md` the run log.

---

## 1. The vision

Humans **see, listen, think, and speak all at once.** None of those faculties waits for
another to finish. You do not stop seeing while you form a sentence; you do not stop
hearing while you decide whether to interrupt.

Every current streaming video model breaks this. They alternate: perceive → then respond →
then perceive again. The response blocks perception. That is why they miss what happens
while they are talking.

**We want to give an AI that simultaneity.** Not by training a new model — by giving a
*frozen* VLM a **cognitive architecture**: a set of independent, concurrently-running
processes that share memory and never block one another.

The design metaphor is **ROS (Robot Operating System)**: independent nodes, asynchronous
message passing, a shared blackboard, no node waiting on another to make progress.

```
          UNBOUNDED STREAM  (video now; omni-modal later)
                   │
                   ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │                    INDEPENDENT ASYNC PROCESSES                    │
   │                                                                   │
   │  [ENCODER]      sees.     fps + attention steered by control      │
   │      │                                                            │
   │      ▼                                                            │
   │  [INGESTER]     remembers. sole writer of the shared KV cache     │
   │      │                                                            │
   │      ├──────────────────┬──────────────────┐                      │
   │      ▼                  ▼                  ▼                      │
   │  [TRIGGER]          [CONTROLLER]       [MEMORY]                   │
   │  is anything         thinks. writes     what happened,            │
   │  worth saying?       the config diff    what was said             │
   │  ~50-100 ms          + the answer                                 │
   │  NEVER decodes       ~1 s, generative                             │
   │      │                    │                                       │
   │      └────── fires ──────▶│                                       │
   │                           ▼                                       │
   │                     [WRITER] speaks. right content, right time    │
   └───────────────────────────────────────────────────────────────────┘

  End state: streaming input, streaming output, proactive control of BOTH.
```

**The single hard requirement:** the right content, at the right time, within **±3 s**.

---

## 1a. NON-NEGOTIABLE INVARIANTS

These bind every design decision, every experiment, and **every tool we build**.

### ⛔ INVARIANT 1 — The system under test must never observe the future.

**The rule is about what the system can SEE, not about what the harness pre-computed.**

**Forbidden:** any component — encoder, ingester, controller, trigger, writer — observing
information from beyond the current video time. No look-ahead. No rewinding. No component
may require the clip to exist in full, or behave differently because it does.

**Allowed:** the harness pre-computing a fixture, *provided it is revealed strictly
causally* — one tick at a time, in order, with the future never visible.

> **The test:** if you cannot tell, from inside the controller, whether you are on a live
> stream or a fixture, the fixture is legal.

This is the distinction that matters, and it makes disk pre-caching a legitimate and useful
prompt-tuning tool rather than a violation. Pre-compute the perception once; feed it to the
system as a stream.

**Two conditions on any pre-computed fixture:**

1. **Cache what is upstream of the decision, not the decision's context.** Store the
   **vision embeddings** (post-encoder, pre-decoder), *not* the KV cache.
   Measured for Qwen3-VL-8B: KV is **144 KB/token** (36 layers × 8 KV heads × 128 dim ×
   2 × bf16) → **8.2 GB per 300 s video**. Vision embeddings are 4096 × bf16 = **8 KB/token**
   → **455 MB** for the same video. **18× smaller**, and — more importantly — a frozen KV
   cache bakes in every ingestion-side hyperparameter (kv_budget, eviction policy, timestamp
   format, sink size), so it cannot be used to test any change to them. Embeddings can.
2. **Assert the eviction boundary.** A cached prefix stops being a valid prefix once
   eviction fires. At 185 tok/frame the budget of 262 144 holds ~1400 frames ≈ **23 min at
   1 fps** — fine for OmniPro clips (`max_seconds=300` → 55.5 k tokens), but the harness must
   **assert** it rather than assume it, and long-horizon experiments must rebuild rather
   than slice.

**Preferred sweep design — one pass, many variants:** stream the video forward exactly once
and, at each tick, evaluate *all* variants against the cache state at that instant. Every
variant sees byte-identical context, so the A/B removes the cross-run variance that produced
F1 0.255 vs 0.051 on one config. Valid under eviction, valid on an unbounded stream.

Threshold and gate tuning stays free: log per-tick scores during a legal streaming run, then
sweep thresholds offline over the **saved numbers** — analysing outputs, not re-reading video.

### ⛔ INVARIANT 2 — Streaming output, not just streaming input
The system must be able to speak *while still perceiving*. Response generation must never
suspend perception. (Lockstep mode is the one sanctioned exception, for benchmark
reproducibility only — see §6.)

### ⛔ INVARIANT 3 — Single writer, snapshot readers
The ingester is the sole writer of the primary KV cache. Readers take MVCC snapshots and
hold no lock. Never relax this to "just let the controller write a bit."

### ⛔ INVARIANT 4 — Frozen model
No gradients, ever. Adaptation happens through in-context learning and architecture. The
moment we fine-tune, we are competing with MiniCPM-o 4.5 on its own terms with less data,
less compute and less time — and the thesis is gone.

---

## 2. The thesis — and why it is defensible against the rival

MiniCPM-o 4.5 is the strongest online-mode baseline on OmniPro (**20.9% F1**). Read its
method carefully and one detail matters more than all the rest:

> "the model first predicts a **binary listen/speak control token** before content
> generation" — the *Listen-Speak (LS) formulation*, which "separates the control decision
> from content generation for stability."

**They had to train that control token into the model.** Full pipeline: speech pretraining,
joint omni pretraining, SFT, then GRPO + RLAIF-V. ~9B parameters, end-to-end.

**Our claim: you do not need to train it. The decision is already in a frozen model's
logits — you just have to ask, and read.**

That is the paper. One sentence:

> **A frozen VLM already knows when to speak. Proactivity is an architecture problem,
> not a training problem.**

Supporting evidence we already hold: the probe-gate arm reached **joint_f1 0.206** on our
27-video 9-task eval — *approximately* MiniCPM-o-4.5's league — with **zero training**.

⚠️ **That comparison is not yet valid and must not be published as-is.** Different subset
(27 videos vs 2,700 samples), different audio conditions, older scoring. It is a promising
signal, not a result. Making it a real, defensible number is the work.

### Where the rival is weak — our three openings

| MiniCPM-o 4.5's own stated limitation | Our attack |
|---|---|
| *"Proactive behavior is still relatively simple"* | The controller **self-schedules, asks itself questions, and steers its own perception**. Proactivity as a control loop, not a token. |
| OmniPro: long-term trigger retention averages **37%** of early-segment performance | **Unbounded stream is our thesis.** Shared KV cache + StreamingLLM eviction + an explicit memory log that *survives* eviction. |
| Requires full multi-stage training to change behaviour | **In-context learning only.** New task = new ICL block. No gradient ever computed. |

---

## 3. The benchmark, honestly

OmniPro, **Online mode**: frames arrive one by one, the model decides autonomously when to
respond, response timestamp = video-second of emission, ±3 s greedy temporal match,
content scored by exact-match (structured) or Gemini judge ≥3 (open-ended), metric = F1.

### Sample counts — measured from `omnipro_data/benchmark.json` (2,700 samples)

**We have been building for the three smallest task subsets in the benchmark.**

| Level | Task | **none** | helpful | required | ICL status |
|---|---|---:|---:|---:|---|
| Reasoning | `dedup_counting` | **121** | 162 | 17 | ❌ needs `count` + identity memory |
| Perception | `realtime_state_monitor` | **98** | 44 | 158 | ❌ needs `phase` |
| Comprehension | `cumulative_counting` | **73** | 29 | 198 | ❌ needs `count` |
| Reasoning | `sequential_step_instruction` | **66** | 82 | 152 | ❌ |
| Perception | `snapshot_counting` | **35** | 38 | 227 | ❌ needs `count` |
| Perception | `instant_event_alert` | 17 | 21 | 262 | ✅ |
| Comprehension | `semantic_condition_alert` | 11 | 27 | 262 | ✅ |
| Perception | `explicit_target_grounding` | 6 | 6 | 288 | ✅ |
| Comprehension | `event_narration` | 5 | 91 | 204 | ❌ |
| | **TOTAL** | **432** | 500 | 1768 | |

`audio=none` is **16.0%** of OmniPro; `none+helpful` is **34.5%**.

### ⚠️ The strategic error this exposes

**The 3 tasks we built ICL for hold 34 vision-only samples. The 6 we skipped hold 398.**

This single fact explains the measurement crisis in `EXPERIMENTS.md`. The 11-video SCA
subset is not a sample — **it is the entire `audio=none` population for that task (n=11).**
No amount of prompt engineering produces a stable F1 on n=11. We were not unlucky; we were
measuring a task that cannot support the claim.

Two consequences, both acted on in the roadmap (§9):

1. **`dedup_counting` is the task OmniPro hands to a vision-only system.** Only 17 of 300
   samples require audio — 283 are attemptable. It is also the hardest cognitive level
   (Reasoning), so success there is the strongest possible result. It needs `count` +
   identity memory, i.e. **pillar 7c**.
2. **`explicit_target_grounding` (n=6) and `event_narration` (n=5) cannot carry a claim.**
   Keep ETG as a qualitative demo; deprioritise narration entirely.

**Re-prioritised target set:** `dedup_counting` (121) + `realtime_state_monitor` (98) +
`cumulative_counting` (73) + `sequential_step_instruction` (66) = **358 samples**, ~10× the
current footing. Four of those five need the memory work (`count` / `phase`) — so **pillar
7c is now the critical path, not a nice-to-have.**

### The audio problem — state it before a reviewer finds it

**84% of OmniPro samples depend on audio.** We ingest none. The `audio_dependency=none`
subset is therefore a **small slice** of the benchmark.

Non-negotiable framing rules:
- We report on the vision-only subset and **say so in the abstract**.
- We **never** claim overall OmniPro SOTA.
- The comparison against MiniCPM-o 4.5 must be **re-run by us on the same vision-only
  subset**, or it is not a comparison.
- The contribution is *architecture*, and it is modality-agnostic by construction —
  audio becomes another encoder node feeding the same ingester. That is the honest pitch.

---

## 4. The eight pillars

### 1. Independent, proactively-controlled vision encoder
Runs as its own process. The control loop steers **fps** within `[idle, focus]` bounds —
sample sparsely when nothing is happening, densely when something is developing.
*Deferred (deliberately):* token pruning — keep only tokens containing objects of interest
named by the control state, drop the rest. **Not now. Keep v1 simple.**

### 2. Ingester — the sole writer
Sees everything, always. Prefills every frame into the shared KV cache. Does not think,
plan, or decide. Bounds memory via StreamingLLM eviction with the system prompt pinned as
an un-evictable sink. Publishes video time on a shared clock.

### 3. Controller — always thinking
Reads the cache, reasons about what is happening, and when something is worth reporting,
produces the answer. This is where task semantics live.

### 4. Config JSON + diff — the control signal
A persistent control state, updated each tick by a **diff**, never rewritten. Carries:
where to look, how much attention to give, **and what question to ask itself later.**

> *Worked example (the user's):* task is "tell me when a player scores." The ball moves
> toward the penalty area. The controller does not answer — it writes
> `{"question_for_next":"did a goal happen?","next_check_s":1,"fps":3}`.
> One second later it checks that specific question against fresh frames.

**This is deferred reasoning, and it is a genuine capability the rival does not have.**

### 5. In-context learning per task
Each of the 9 tasks gets an ICL block teaching what counts as an event and how to phrase
the answer. **Task semantics only** — output format belongs in the decoder (§6).

### 6. Shared KV cache
One cache. Ingester is the sole writer (no write-write conflict on a linear sequence).
Readers take MVCC snapshots and generate on private clones, holding no lock. Two clocks
(logical RoPE position vs physical length) survive eviction.

### 7. Agent memory — the writer's own trace
**Memory is the append-only record of the writer's cognition:** the *essence of what it
thought* at each moment, and *what it actually told the user*, timestamped and accumulated
over the life of the stream.

It exists to make three things possible:

| purpose | why memory is required |
|---|---|
| **Deduplication** | "Have I already reported this?" cannot be answered from the current frame. It needs the record of what was said before. Proven: the model re-fired one identical alert **7×** when asked to track this in its head. |
| **Accumulation** | "How many so far?" / "which distinct ones have I seen?" is a running total over the whole stream. This is what `cumulative_counting` and `dedup_counting` *are*. |
| **Survival** | The KV cache forgets: at 262 144 tokens / ~185 tok per frame ≈ **1400 frames ≈ 23 min @ 1 fps**, eviction begins. Memory is the only thing that outlives it — **load-bearing for the unbounded-stream claim.** |

Structure — one append-only, timestamped log with two entry kinds, plus derived accumulators:

- **`thought`** — the essence of the writer's reasoning at that tick, in its own words.
  Model-written, appended, never rewritten.
- **`said`** — the exact text delivered to the user. **Code-owned and model-unwritable:** if
  the model could write this it could hallucinate having already answered, and dedup would
  silently fail open.
- **`count` / `phase`** — accumulators maintained across ticks from the log.

**Append-only, bounded ring — never rewritten.** A memory the model rewrites each tick costs
tokens proportional to its length, so tick latency would grow with watch time — the system
would get *slower the longer it watches*, exactly backwards for an unbounded stream.
**Memory is a log, not a document.**

### 8. Fast trigger — a separate process that never decodes ⭐
An independent process continuously watching the cache to answer one question: *is
something worth reporting happening right now?* — **without** running the slow
autoregressive loop. Target: a decision every **1–2 s or faster**.

**This is the most important pillar and it is currently mis-architected.** See §5.

---

## 5. The central architectural decision

### The problem

Today the controller does everything on one path: wake up → prefill ~900 ICL tokens →
decode ~35 tokens → decide → answer. Measured: **3.2–8.8 s per tick.**

Decode is ~100 ms/token and **that floor cannot be bought away** — the GPU sits ~90% idle
waiting on kernel launches, so more GPUs do not help (measured: multi-GPU made it *worse*,
107–407 ms/tok, because the cross-GPU cache copy costs more than the contention it saves).

So the timing decision is trapped behind a slow generative process. **The system is
slowest at exactly the moment timing matters.**

### The fix — split *when* from *what*

```
  TRIGGER  (fast, always on, never decodes)      ~50-100 ms
     splice a question onto the cache, ONE forward pass,
     read the logits for "yes"/"no" -> a continuous score in [0,1]
     apply the tuned Schmitt gate (hyst2b: fire >0.5, re-arm <0.40)
                      │
                      │ fires
                      ▼
  CONTROLLER (slow, generative, only when needed)  ~1 s
     schema-walked decode -> config diff + the answer
```

Why this is right:

- **It is the rival's own mechanism, training-free.** MiniCPM-o trains a binary
  listen/speak token. We read the same binary decision from frozen logits. Same idea, no
  gradients. That parallel is the paper's strongest argument.
- **Timing precision stops depending on decode speed.** Trigger latency ~100 ms, so ±3 s is
  bounded by frame rate, not token throughput.
- **It ends the false head-to-head.** `probe` and `controller` are currently framed as
  *rival systems*. They are not — they are the **two halves of one architecture**. The
  probe is the trigger; the controller is the writer. `EXPERIENCE.md`'s head-to-head should
  become an *ablation* ("trigger alone" vs "controller alone" vs "both").
- **It gives a continuous confidence** where the controller only had true/false — which is
  what makes hysteresis, calibration, and dense evaluation possible at all.

**Cost:** the trigger question is one static sentence, so it lacks the per-tick perception
grounding (`seen`) that was our single biggest accuracy lever (F1 0.0 → 0.255). Mitigation:
the controller still runs on a slow background schedule to refresh `seen`, deferred
questions, and memory — the trigger only handles the *interrupt*.

---

## 6. Non-blocking vs. reproducibility — the honest resolution

Pillar: *"no process should wait or block anything."* The code today does the opposite —
`deterministic=True` makes the ingester **block** until the controller finishes
(`input_ingester.py:102-108`).

Both are correct, for different purposes. **Keep both, and never mix them in one number.**

| mode | behaviour | what it is for |
|---|---|---|
| `deterministic=True` **(lockstep)** | ingester waits for every due tick; frame-indexed walk; bit-reproducible | **All benchmark numbers.** Matches OmniPro's "frames one by one". Without it, run-to-run variance (F1 0.255 vs 0.051 on one config) exceeds every effect we measure. |
| `deterministic=False` **(free-running)** | nothing waits; wall-clock paced; frames may be missed while thinking | **The real-time claim.** Latency, dropped-frame rate, end-to-end responsiveness. |

Two different claims, both honest, **stated separately**:
- ✅ *"identifies the correct moment within ±3 s under OmniPro's online streaming protocol"* — lockstep
- ✅ *"sustains N fps with a p95 speak-latency of X ms on a live stream"* — free-running
- ❌ *never* a single number implying both

**Paper plan:** headline table in lockstep; a dedicated real-time section reporting
free-running latency + frame retention. The gap between them **is a finding**, not an
embarrassment — no prior work reports it at all.

### Honest note on "independent processes"

These are Python **threads** sharing one CUDA context, not OS processes. On one GPU with
one model, forward passes serialize on the SMs regardless. What the architecture buys is
the removal of **logical** blocking — no component waits on another's *completion* to make
progress. Physical serialization is managed by making the hot path (the trigger) cheap.
True multi-process with GPU IPC is future work and should not be claimed.

---

## 7. Does the code match the intention?

| # | Pillar | Code today | Verdict |
|---|---|---|---|
| 1 | Controlled encoder | `vision_stream.py`, `ctrl.set_fps()`, fps ∈ [1,3] | ✅ **matches** (pruning deferred by choice) |
| 2 | Ingester, sole writer | `input_ingester.py` + `manager.py` | ✅ **matches** |
| 3 | Controller always thinking | `controller.py` | ⚠️ **polls on a self-set timer**, not continuous |
| 4 | Config JSON + diff | state dict + `_extract_json` merge | ❌ **the diff does not work** — 0/15 tick compliance |
| 4b | Deferred question | `question_for_next` + `next_check_s` | ✅ **exists** — this is the football example, already built |
| 5 | Per-task ICL | 3 of 9 tasks | ⚠️ **partial**, and doing a job it can't do (format) |
| 6 | Shared KV cache | `manager.py`, MVCC snapshot | ✅ **matches** |
| 7a | Memory: what was *said* | `reported` list → prompt history | ✅ **matches** |
| 7b | Memory: writer's *thoughts* | `seen` exists but is **discarded every tick** | ❌ **missing** — the trace is thrown away |
| 7c | Memory: accumulators (`count`/`phase`) | — | ❌ **missing** (blocks 4 tasks) |
| 8 | Fast non-decoding trigger | exists as `gate_mode="probe"` | ❌ **built as a RIVAL, not a component** |
| — | Non-blocking | `deterministic=True` blocks by design | ⚠️ **intentional** — see §6 |

**Summary: the substrate matches the vision well. The proactivity layer does not.**

Pillars 1, 2, 6, 7a are built and sound — that is the hard infrastructure and it is done.
The three real gaps: **the diff doesn't work (4), memory of events is missing (7b/7c), and
the fast trigger is mis-framed as a competitor instead of a component (8).**

Two corrections to how the work has been framed:

1. **The head-to-head is the wrong experiment.** `probe` vs `controller` are not
   alternatives — probe is *when*, controller is *what*. Reframe as an ablation.
2. **The ICL prompts are doing two jobs and failing one.** ~400 of ~900 tokens teach output
   *format*; measured compliance is 0/15, and the model doesn't even copy the examples'
   spacing. Format must move to the decoder. This also *shrinks* the prompt — and our own
   finding is **"more prompt = worse."** Latency and accuracy point the same way here.

---

## 8. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Vision-only subset too small to be publishable | **high — CONFIRMED REAL** | Measured: our 3 built tasks hold **34** `none` samples; the 6 unbuilt hold **398**. Pivot to `dedup_counting`/`state_monitor`/`counting` (§3). Report `none` as headline + `none+helpful` (932 samples) as a second subset. |
| Measurement noisier than the effects | **high** | Dense per-tick AUC for iteration (~3,000 labelled decisions vs ~33 events); F1 only for final numbers. |
| 8B comprehension ceiling on abstract conditions | medium | Known: some videos emit 0 across every variant. Report as a limitation; do not prompt-engineer against it. |
| 6 of 9 tasks unbuilt with 7 weeks left | medium | Memory design unlocks 4 of them at once. Build in mechanism-reuse order. |
| Rival comparison not apples-to-apples | **high** | Re-run MiniCPM-o 4.5 ourselves on the identical subset, or drop the claim. |
| Env instability (scratch loses stdlib) | low | `recreate_env.sh`, known and handled. |

---

## 9. Roadmap to 18 Sep 2026

**Phase 1 — make measurement trustworthy (weeks 1–2). Nothing else matters until this is done.**
1. Schema-walked decoder: format guaranteed in code, prompts shrink ~40%, ticks ~3× cheaper.
2. One-pass multi-variant sweep + dense AUC metric — **forward-only, never buffer a clip**
   (INVARIANT 1). All variants scored against identical context in a single stream.
3. Re-run every existing variant on it. Find out which past "regressions" were real.

**Phase 2 — the architecture (weeks 3–4).**
4. Split trigger from writer (§5). This is the paper's core claim; build it properly.
5. **Memory: note log + `count` + `phase`. Now critical path** — it unlocks the four tasks
   holding 358 of our 432 vision-only samples (§3).
6. Reframe the head-to-head as an ablation: trigger-only / controller-only / both.

**Phase 3 — coverage (weeks 5–6). Re-ordered by sample count, not by mechanism ease.**
7. `realtime_state_monitor` (n=98, `phase`) → `cumulative_counting` (n=73, `count`) →
   `dedup_counting` (n=121, `count` + identity memory — **the flagship result**) →
   `snapshot_counting` (n=35) → `sequential_step_instruction` (n=66).
   Keep ETG (n=6) as a qualitative demo; **drop `event_narration` (n=5)**.
8. Free-running real-time measurement: latency, frame retention, p95 speak-latency.

**Phase 4 — the paper (week 7).**
9. Ablations, the honest subset table, the rival re-run, limitations.

**Standing rule: one variable per experiment.** V3 changed three things at once
(2nd example + calibration + gate 0.5→0.35), regressed, and taught us nothing about which.
That single discipline failure has cost more time than any technical problem in this project.
