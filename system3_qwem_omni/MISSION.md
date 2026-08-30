# MISSION — System 3: a training-free cognitive architecture for proactive streaming video
## Name of paper: Foresight: Planning Future Perception in Streaming Vision-Language Model
- What are our contributions
    - input pruning, planning
    - good config file
    - kv cache mgmt (compact, summarize) [not our problem]e
     - without training, future forecast (why?) because of config.
        ours does not work in non streaming  becuase no reconfig

We can self configure streamming vlm. needs 2 llms, config file (str of file)
over the time see the diagrams
benchmark , results .-> rest details, 

1. benchmark perf 2. circuit diagram 3. state diagram 


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

<!-- ### ⚠️ The strategic error this exposes

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
 -->
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
| `deterministic=True` **(lockstep)** | ingester waits for every due tick; frame-indexed walk; ~~bit-reproducible~~ **see below** | **All benchmark numbers.** Matches OmniPro's "frames one by one". Without it, run-to-run variance (F1 0.255 vs 0.051 on one config) exceeds every effect we measure. |

> ### ⚠️ "bit-reproducible" is FALSE — measured 2026-08-10
> A null control (`omniprofast/ab_inplace.sh`: the *same* config run twice, lockstep,
> same seed, one 25 s video) diverged: **17 of 25 `ctrl.raw` ticks differed**, along
> with 16 `ctrl.gate` and all 3 emissions. Example at t=10.0 s —
> run A `{"seen":"soccer players on field, one about to kick ball",…}`,
> run B `{"seen":"soccer game, players in action",…}`.
> Headline metrics happened to match on this clip (`time_f1=0.5` both), so the
> divergence is in generated text and `p_hit`, not necessarily in the score — but
> lockstep does **not** deliver bit-identical runs, and any A/B that assumes it does
> cannot attribute a difference to its own variable. Root cause not yet found;
> `seed_everything(cfg.seed, cfg.deterministic)` runs per sample, so the suspects are
> CUDA kernel non-determinism and allocator/batch-shape effects on bf16 reductions.
> **Consequence:** every past head-to-head run as two separate processes needs a null
> control before its delta is believed. This is currently **blocking** the removal of the
> per-tick KV-cache copy — see §11. Leading hypothesis, from that section: seeded
> `torch.multinomial` sampling (`writer_greedy=False`) amplifying bf16 kernel jitter into
> a different token, and from there a different stream. Untested.
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

---

## 10. Field ablation — the minimal controller output, per task

**Started 2026-08-10.** Pillar 4 says the config JSON is the control signal. It does not say
every field earns its place, and it does not say the same fields earn their place on every
task. This section finds the **minimal output schema per task**: the smallest set of fields
that holds the benchmark number, with everything else deleted.

Why it matters beyond tidiness: **every field the model has to decode is a field the model
has to wait for.** `have_enough_info` costs zero decode steps (it is a logit read on a forced
key). `seen` and `answer` are free decodes and are therefore the entire tick latency. A field
that is not read by the scorer and not load-bearing for the gate is pure delay on the one
path where delay is the metric.

### The field inventory, with its true cost

Emitted by the schema walk in `controller.py:_schema_tick`. "Cost" is decode steps —
forced tokens prefill in one forward and are ~free; sampled tokens are ~100 ms each.

| field | how produced | cost | when |
|---|---|---|---|
| `seen` | free decode, cap 12 tok | **~12 decodes — the whole quiet-tick cost** | every tick (`seen_mode=before`) |
| `have_enough_info` | logit read `P(true)` | **0 decodes** | every tick |
| `event_time_s` | free decode, cap 4 tok | ~4 decodes | hit ticks only |
| `answer` | free decode, cap 32 tok | ~32 decodes | hit ticks only |
| `more` | logit read `P(true)` | **0 decodes** | every tick |
| `fps`, `next_check_s`, `note`, `count`, `phase` | free decode of the tail, cap 60 tok | ~60 decodes | only when `p_more ≥ 0.5` |

### Three separate questions per field — do not conflate them

1. **Is it evaluated?** Does `metrics.py:TASK_CONTENT_KIND` read it for this task at all?
2. **Is it load-bearing?** Does removing it change the number, through some path *other*
   than being scored?
3. **What does it cost?** Decode steps, and therefore tick latency and timing error.

A field can be (1)-no and still (2)-yes. **`answer` on the alert tasks is exactly this case
and it is the trap in this whole analysis:** `instant_event_alert` and
`semantic_condition_alert` are `time_only` — the scorer never looks at the text, so any
time-matched emit is correct regardless of what it says. But in `controller.py` the fire
condition is `fire = bool(answer) and armed and p_hit >= high`. **An empty `answer` cannot
fire.** `answer` also feeds `reported`, which feeds the "what you have already told the user"
prompt block and the `_word_sim` dedup check. So deleting it deletes the gate and the dedup
memory, not just 32 decode steps.

The honest form of the hypothesis for those two tasks is therefore not "drop `answer`" but:
**replace the answer-as-gate with something that costs nothing, and emit a fixed string.**

### Corpus

`omniprofast/output_full9/` — 1,196 sample runs, ~200k logged ticks, all 9 tasks:

| task | sample runs |
|---|---:|
| `dedup_counting` | 370 |
| `realtime_state_monitor` | 187 |
| `sequential_step_instruction` | 169 |
| `cumulative_counting` | 143 |
| `event_narration` | 126 |
| `snapshot_counting` | 91 |
| `instant_event_alert` | 52 |
| `semantic_condition_alert` | 46 |
| `explicit_target_grounding` | 12 |

Every tick logs `ctrl.raw` (the emitted JSON) and `ctrl.gate` (`p_hit`, `p_more`, `fire`,
`gen`, `ntok`, `fps`, `next`, `q`, `count`, `notes`). Task is recovered by joining on the
`===== [i/n] task::video::id` sample header. **This is offline analysis of saved numbers —
legal under INVARIANT 1, no video is re-read.**

### Method — in order, one at a time

1. **Parse.** One tidy per-tick table: `run, task, video, vt, field values, p_hit, p_more,
   fire, gen_s, ntok`. Nothing interpreted yet.
2. **Occupancy.** Per task: how often is each field non-empty / non-default? A field the
   model almost never fills is dead by observation, before any ablation.
3. **Cost attribution.** Regress `gen_s` on which fields were emitted. Turn "field X" into
   "field X costs Y ms/tick and Z seconds of timing error per video."
4. **Predictive value.** Does the field's content correlate with a correct emit, or only
   with an emit? A field that fires as often on false positives as on true positives is
   noise the model is paying for.
5. **Ablate.** Only then, and one field per run, on the tasks where the parse says it matters.

### Step 1-2 result (2026-08-10) — `omniprofast/fields.py output_full9`

199,909 ticks, 400 run logs, 9 tasks. **Occupancy only — no ablation has been run yet,
so nothing below is yet a causal claim.**

First, a logging bug found by the parse: `controller.py:513` truncates `ctrl.raw` at 240
chars. 16,995 ticks (8.5%) are cut, **100% of them exactly at the cap** — no model failure,
but it is *selective*: it eats the longest emissions, i.e. exactly the ticks that used the
`more` tail. `fields.py` salvages keys before the cut. **Raise the cap before the next run**
or every future tail measurement is biased low. Worst hit: `event_narration`, 57% truncated.

How often each key is emitted at all, pooled across all 199,909 ticks:

| field | emitted | share |
|---|---:|---:|
| `seen` | 199,909 | **100.00%** |
| `have_enough_info` | 199,909 | **100.00%** |
| `answer` | 126,763 | 63.41% |
| `event_time_s` | 126,242 | 63.15% |
| `count` | 32,309 | 16.16% |
| `question_for_next` | 9,690 | 4.85% |
| `next_check_s` | 6,262 | 3.13% |
| `fps` | 4,155 | 2.08% |
| `phase` | 1,074 | 0.54% |
| `note` | **0** | **0.00%** |

Four findings, in order of how much they cost us:

1. **The quiet path is not quiet.** `_schema_tick`'s docstring assumes "the ~99% quiet
   path". Measured hot-path rate (`have_enough_info=true`, which decodes `event_time_s` +
   `answer`, ~36 tokens): `realtime_state_monitor` **89.9%**, `cumulative_counting` 70.0%,
   `event_narration` 68.9%, `snapshot_counting` 67.2%, `explicit_target_grounding` 64.8%,
   `sequential_step_instruction` 56.0%, `dedup_counting` 51.4%, `instant_event_alert` 48.2%,
   `semantic_condition_alert` 39.2%. **The design was costed against a 1% hot path and is
   running a 40–90% one.** Median tick: 2.4–4.2 s. This, not any individual field, is where
   the latency went.
2. **`note` has never been emitted. Not once in 199,909 ticks.** Pillar 7b — "the essence of
   what it thought at each moment", the append-only trace — is present in the schema, wired
   into a bounded ring in `controller.py`, and *completely unused by the model*. §7 marks 7b
   "❌ missing — the trace is thrown away". It is worse than that: the trace is never written.
3. **The perception-control signal is unused on the tasks we care about.** `fps` and
   `next_check_s` on the four flagship tasks: `dedup_counting` 0.1%/0.1%, `realtime_state_
   monitor` 0.4%/0.3%, `cumulative_counting` 0.6%/0.6%, `snapshot_counting` 0.3%/0.3%. They
   are only touched on the alert tasks (`semantic_condition_alert` 14.4%/17.4%,
   `instant_event_alert` 8.1%/15.2%). Pillar 1's steered encoder is, on the flagship set,
   a constant.
4. **`question_for_next` — the deferred-question mechanism, §4's football example and the
   thing "the rival does not have" — fires on 4.85% of ticks**, and on the flagship tasks
   `dedup` 5.9%, `cumulative` 3.8%, `snapshot` 0.0%, `realtime_state_monitor` 0.3%.

And one thing working as designed: **`count` appears only on the three counting tasks**
(`cumulative` 30.4%, `dedup` 29.9%, `snapshot` 20.0%) and nowhere else, and **`phase` only
on `realtime_state_monitor`** (3.2%) — the one task scored on `state`. The per-task ICL *is*
gating which fields the model reaches for. It just reaches too rarely on `phase`.

### Step 3 result (2026-08-10) — who actually CONSUMES each field

Occupancy says what the model writes. This says what the code reads. Traced every
field from the merge in `controller.py` to its consumer. **Three fields have no
consumer at all**, which no amount of prompt work would have revealed.

| field | consumer in the code | verdict |
|---|---|---|
| `seen` | **none** — appended to `seen_trace`, and `seen_trace` is only read behind `cfg.seen_trace_in_prompt`, which defaults **False** | value discarded, but see below |
| `have_enough_info` | `level` → the fire gate | live, and free (logit read) |
| `event_time_s` | **the emitted timestamp** (see below) | live, scored on all 9 tasks |
| `answer` | gates firing (`fire = bool(answer) and ...`), becomes the emission text, appended to `reported` which feeds the dedup prompt | live; 3 jobs at once |
| `fps` | `ctrl.set_fps()` → the encoder's sample rate | live, barely used |
| `next_check_s` | `next_check_vt` → when the next tick happens | live, barely used |
| `question_for_next` | spliced into the next tick's prompt | live, barely used |
| `note` | `notes` ring → the "NOTES YOU KEPT" prompt block | **consumer exists, producer never fires** |
| `count` | line 311 (init), line 566 (printed in the log). **Nothing else.** | **WRITE-ONLY** |
| `phase` | line 311 (init). **Nothing else — not even logged.** | **WRITE-ONLY** |

**`event_time_s` is the highest-value field per token in the schema.** It does not
annotate the emission, it *is* the emission's timestamp:

```python
t_rec = vt
ev = state.get("event_time_s")
if ev is not None:
    t_rec = min(vt, max(vt - 10.0, float(ev)))   # back-date, capped at 10 s
reported.append((t_rec, answer)); evaluator.record_trigger(t_rec, 1.0)
```

OmniPro matches within ±3 s of `t_rec`. So four decode tokens move the number F1 is
computed from, on **every** task including the two `time_only` ones. Whether the
back-dating helps or hurts has never been measured, and can be measured offline for
free (§Todo 2).

**`seen` is a scratchpad, not memory — and that distinction is the whole point.**
Its value is thrown away, but it is decoded *before* the `have_enough_info` logit
read, so its tokens are in the cache when `p_hit` is read. That is the F1 0.0 →
0.255 lever. The 12 tokens buy "look before you judge". Calling it memory is wrong;
deleting it because "the value is unused" would be very wrong.

**`count` and `phase` are produced into a void.** The comment at `controller.py:305`
records that the merge used to discard them and was fixed. The merge was fixed; **no
consumer was ever written.** Pillar 7c is, today, a field in a dict.

### Deliverable — the per-task minimal schema

`scored` = the metric reads it. `structural` = the code needs it whatever the task.
`fill%` = measured non-default rate (`output_full9`). **Ablations have not been run
yet**, so "drop" is a hypothesis with a measured cost, not a proven result.

| task | content kind | scored content field | structural (all tasks) | task field | fill% | drop candidates |
|---|---|---|---|---|---|---|
| `instant_event_alert` | `time_only` | **none** | `have_enough_info`, `event_time_s`, `answer`† | — | ans 48.5 | `answer` text†, `fps` 8.0, `next_check_s` 15.0, `q` 1.5 |
| `semantic_condition_alert` | `time_only` | **none** | as above | — | ans 39.3 | `answer` text†, `fps` 14.1, `next_check_s` 17.3, `q` 4.7 |
| `explicit_target_grounding` | `position` | `answer` (9-cell) | as above | — | ans 64.3 | `fps` 4.8, `next_check_s` 5.7, `q` 1.7 |
| `snapshot_counting` | `count` | `answer` (integer) | as above | `count` | 14.8 | `count` (write-only), `q` 0.0, `fps` 0.3 |
| `cumulative_counting` | `count` | `answer` (integer) | as above | `count` | 25.8 | `count` (write-only), `fps` 0.5, `next_check_s` 0.6 |
| `dedup_counting` | `count` | `answer` (integer) | as above | `count` | 29.6 | `count` (write-only), `fps` 0.1, `next_check_s` 0.1 |
| `realtime_state_monitor` | `state` | `answer` (state name) | as above | `phase` | **2.3** | `phase` (write-only AND barely emitted), `fps` 0.0, `q` 0.0 |
| `event_narration` | `gpt_judge` | `answer` (free text) | as above | — | ans 27.5‡ | — (the one task that uses cadence: `next_check_s` 31.8, `q` 44.6) |
| `sequential_step_instruction` | `gpt_judge` | `answer` (free text) | as above | — | ans 45.2 | `fps` 3.4, `q` 4.2 |

† **The alert-task trap.** The scorer never reads `answer` on these two, but
`fire = bool(answer) and ...` means an empty `answer` *cannot fire*, and `answer`
feeds `reported` (the dedup prompt + `_word_sim`). The hypothesis is not "drop
`answer`" — it is **"replace the answer-as-gate with something free and emit a fixed
string"**, saving ~32 decode tokens on 39-49% of ticks.

‡ `event_narration` is 57% log-truncated, so its fill rates are lower bounds.

**Dead on every task, all 199,909 ticks:** `note` (never emitted), `count`/`phase`
(never read). **Near-dead on the four flagship tasks:** `fps`, `next_check_s`,
`question_for_next` — all under 6%.

### Todo — in order, one variable each

1. **Raise the `ctrl.raw` log cap** (`controller.py:513`, 240 chars). Costs nothing,
   and until it is raised every tail-field measurement is biased low.
2. **Measure `event_time_s` offline. NOW THE MOST URGENT ITEM — see below.**
   Replay saved logs scoring emissions at `t_rec` (back-dated) vs at `vt` (tick
   time). No GPU, no video — the logs hold both.

### ⚠️ `event_time_s` back-dating — a protocol risk, measured 2026-08-10

OmniPro online mode defines the **response timestamp as the video-second of
emission** (§3). We do not report that. `controller.py` overwrites it with the
model's own claim about when the event happened, clamped to 10 s in the past:

```python
t_rec = min(vt, max(vt - 10.0, float(ev)))     # ev = model-authored event_time_s
```

`metrics.py` then matches on exactly that: `emit_times = [float(e["t_sec"]) …]` →
`match_emits_to_gt(…, tolerance=3.0)`. So **`event_time_s` moves time-F1 (and
joint-F1 through the content gate) and cannot touch `content_acc`.**

Measured over the 30,405 fired ticks in `output_full9`:

| | |
|---|---:|
| emissions back-dated at all | **80%** |
| back-dated by **more than the entire ±3 s tolerance** | **75%** |
| median shift | **10.0 s** — the clamp ceiling, on all 9 tasks |

Per task, the >3 s shift rate: `instant_event_alert` 94%, `cumulative_counting` 93%,
`snapshot_counting` 92%, `realtime_state_monitor` 87%, `semantic_condition_alert`
84%, `dedup_counting` 81%, `explicit_target_grounding` 79%, `sequential_step_
instruction` 51%, `event_narration` 14%.

The model is **not** emitting noise — `corr(vt, event_time_s) = +0.726`, and the
median tracks video time by segment (4 s when `vt<20`, 137 s when `vt∈[120,300)`).
It is reading real in-context timestamps. It simply reports events as having
happened well in the past, and the code then moves the emission there.

**Provenance: there is none.** `git log -S "vt - 10.0"` returns exactly one commit,
`2b0fc3c` (14 Jul 2026), whose subject is *"V1 edge: level->edge firing in code +
greedy control + level ICL"*. The clamp arrives as a side clause — *"Optional
event_time_s recorded (clamped vt-10..vt)"* — with no derivation, no justification
for 10 over 3 or 5, and no experiment. It is not OmniPro's rule and it is not in any
design doc. It has silently shaped every eval number since 14 July.

### Re-scored (`omniprofast/retime.py`, 2026-08-10) — the clamp COSTS us

**I predicted this inflated our numbers. It does the opposite. Recorded because the
prediction was wrong, not because it was right.** 30,405 fired ticks, 886 samples,
re-scored offline under four timestamp policies with `metrics.py`'s own scorer:

| policy | meaning | macro time-F1 | Δ vs `vt` | micro time-F1 |
|---|---|---:|---:|---:|
| `vt` | when we actually spoke — **protocol-faithful** | **0.1659** | — | 0.175 |
| `clamp3` | back-date, capped at the tolerance | 0.1646 | −0.0013 | 0.173 |
| `clamp10` | **today's behaviour** | 0.1558 | **−0.0101** | 0.165 |
| `ev` | model's raw claim, unclamped | 0.1356 | −0.0303 | 0.142 |

Worst-hit tasks: `sequential_step_instruction` −0.037, `semantic_condition_alert`
−0.020, `realtime_state_monitor` −0.013.

**Deleting the clamp is a free win of +0.0101 macro time-F1**, and it makes us
protocol-faithful at the same time. There is no trade-off to weigh.

**Why it hurts, and why that is not a contradiction.** Per-emission, the model's raw
`event_time_s` really is closer to ground truth (41.7% within ±3 s vs 13.8% for
`vt`). But OmniPro matches **greedily, 1-to-1**: a ground-truth event can be claimed
by only one emit. Back-dating makes emissions *pile onto the same few moments*,
where the extras are wasted; `vt` *spreads them across the video*, so more distinct
GT events get covered. At recall 0.818 the binding constraint is coverage, not
per-stamp accuracy. **Good localisation and good coverage are different objectives,
and the metric rewards the second.**

**What this actually exposes is far larger than the clamp: precision 0.098.**
29,706 emissions against 3,562 ground-truth events — we fire roughly **8× too
often**, and ~90% of emissions are false positives. Every timestamp policy is a
rounding error next to that. The gate, not the clock, is where the number is.

`EVAL_PROTOCOL.md` records three rules where our scorer disagreed with upstream, all
since fixed. This is a fourth, on the **prediction** side rather than the scoring
side, which is why that audit did not catch it.

**Actions:** delete the clamp (stamp `vt`); keep event localisation as its own
result if it is worth reporting — "we report *when*, not just *that*" — in its own
table, never folded into time-F1. The §2 signal is not inflated by this; if
anything it was understated.

### ⭐ `event_time_s` is an IDENTIFIER, not a timestamp (Dipan, 2026-08-10)

The right use of the field, and it inverts the conclusion above. `event_time_s` is a
bad *timestamp* (back-dating loses 0.010) and an excellent *identity key*: **if the
model reports an event time it has already spoken about, it is the same occurrence —
stay quiet.** And because the schema walk emits `event_time_s` *before* `answer`, the
decision is made **before** the expensive decode: a suppressed tick costs 0 answer
tokens instead of 32.

Replayed offline over 29,706 emissions (`retime.py --dedup`), emission time always
`vt`:

| rule | emits | time-P | time-R | **macro time-F1** |
|---|---:|---:|---:|---:|
| none (today) | 29,706 | 0.098 | 0.818 | 0.1659 |
| `gap10` — plain refractory timer, **the control** | 12,260 | 0.149 | 0.513 | 0.2105 |
| `ev0` — suppress exact repeats, **no tuning** | 14,292 | 0.142 | 0.568 | **0.2191** |
| `ev10` — suppress within ±10 s | 9,828 | 0.163 | 0.448 | **0.2445** |

**The control is the point.** `ev3` emits *more* than `gap10` (12,808 vs 12,260) and
still scores higher (0.2325 vs 0.2105). At equal or larger emission budgets the
identity key beats the timer, so `event_time_s` is carrying real information — this
is not "any dedup helps".

Per task at `ev10`, **8 of 9 improve**: `snapshot_counting` +0.193,
`instant_event_alert` +0.152, `explicit_target_grounding` +0.125,
`dedup_counting` +0.079, `cumulative_counting` +0.078, `semantic_condition_alert`
+0.058, `realtime_state_monitor` +0.031, `event_narration` +0.006. Only
`sequential_step_instruction` regresses (−0.015) — plausibly real, since its events
are genuinely repetitive steps.

Side effect: **636,096 answer tokens never decoded** (~17.7 GPU-hours at 100 ms/tok).
Accuracy and latency move the same direction, which is rare here.

**Four caveats, none of which are optional:**

1. **This is a replay screen, not a result.** Suppressing an emission would have
   changed `reported`, hence the next prompt, hence later ticks. A real run must
   confirm it. Same standing as any offline gate sweep (§1a).
2. **Time-F1 only.** Content is unjudged here. Dedup can suppress the emission that
   had the *correct* content and keep a worse one, so joint-F1 may move differently.
3. **`ev10`'s window was chosen on the data it is scored on** — selection on test.
   Mitigated by the plateau (0.2408–0.2445 across ev5–ev25, so it is not a knife
   edge), but the window must be fixed a priori or fitted on a held-out split before
   publication. This project has already paid for that mistake once.
4. **Prefer `ev0` as the default.** It is exactly the stated idea — *same timestamp
   twice = duplicate, don't decode, don't emit* — it has **no free parameter**, and
   it already delivers +0.053 macro. `ev10` is the tuned variant and must be
   defended as one.

**This reframes pillar 7a/7c.** Dedup was assumed to need the `reported` text history
and word-overlap similarity (`_word_sim`). It does not: a 4-token integer the model
already emits does it better, earlier, and for free.
3. **Decide `count`/`phase`: wire a consumer, or delete the fields.** Today they are
   the worst of both — paid for in decode tokens, worth nothing. This is pillar 7c and
   it is the critical path for 4 tasks (§9), so "delete" is almost certainly wrong;
   the point is that *writing them was never the missing piece*.
4. **`note`: delete it, or find why the model never writes one.** A field with a
   0/199,909 fill rate is either badly prompted or genuinely unwanted. Do not leave a
   third state.
5. **Alert tasks: kill the answer-decode.** Gate on `p_hit` alone and emit a constant.
   ~32 tokens off 39-49% of ticks on IEA + SCA, and by construction it cannot change
   the score — `time_only` means the text is never read.
6. **Then ablate**, one field per run, on the tasks where 1-5 leave real questions.

**Do not skip to 6.** Steps 2-5 are free or nearly free and each one either removes a
field or promotes it to "must keep" without spending a GPU-hour.

---

## 11. Killing the per-tick KV-cache copy — attempted 2026-08-10, **BLOCKED**

**The cost.** INVARIANT 3 makes the controller an MVCC *reader*: every tick it calls
`mgr.snapshot_clone()`, a full deep copy of the shared KV cache, generates on the private
clone, and throws it away. At ~144 KB/token that is a second copy of the whole cache once
per second of video — on a 300 s clip, ~8 GB of GPU memory allocated and freed 300 times.
The same prefix is paid for twice: once resident for the ingester, once copied for the
controller.

**The proposed fix — borrow, don't copy.** In lockstep (`deterministic=True`, which is
every benchmark number, §6) there is *no concurrent writer*: the ingester is parked in
`while clock.get_next_check() <= vt: sleep(0.002)` for the whole tick. The copy protects
against nothing. So: generate directly on the primary at the same `(pos_start,
phys_start)`, then truncate the appended tokens away. Same K/V prefix + same positions ⇒
identical logits, zero copy.

**What was built** (all still uncommitted as of 2026-08-12):
- `async_omni_v2/manager.py` — `borrow_begin()/borrow_end()`, with `_assert_not_borrowed()`
  guarding `ingest`/`evict`/`probe` and refusing a second concurrent borrow. A borrow is
  *declared*, not locked: a violation raises loudly rather than silently corrupting.
- `async_omni_v2/config.py` — `controller_cache_mode: "snapshot" | "inplace"`, default
  `snapshot`. `controller.py` refuses `inplace` **loudly** and falls back unless
  `deterministic=True` and the controller shares the manager's GPU. It never silently
  downgrades — a silent fallback would make a memory/latency result unattributable.
- `omniprofast/test_inplace.py` — the *mechanism* proof: T1 bit-identical restore, T2
  `torch.equal` logits vs a snapshot forward, T3 a 16-token greedy walk matching token for
  token, T4 isolation under exceptions and concurrent writers, T5 memory/time cost.
- `omniprofast/ab_inplace.sh` + `ab_inplace_diff.py` — the *system* proof: the real
  pipeline over real video, three arms (`snapshot`, `snapshot2`, `inplace`), diffing
  `ctrl.raw` / `ctrl.gate` / emissions / `online_pred` with timing fields stripped.

### 🚧 THE BLOCKER: the null control failed, so the A/B cannot decide anything

Run 2026-08-10 09:10-09:12, one 25 s clip (`instant_event_alert::Ull7qP303ds`), 25 ticks.

| comparison | ctrl.raw | ctrl.gate | emissions |
|---|---|---|---|
| **snapshot vs snapshot2** (null control — *same mode twice*) | 17/25 differ | 16/25 differ | 3/3 differ |
| **snapshot vs inplace** (the actual test) | 17/25 differ | 15/25 differ | 3/3 differ |

**The noise floor is as large as the effect.** `inplace` is statistically
indistinguishable from re-running `snapshot` — which means this harness can neither
confirm nor refute equivalence. The A/B is not a negative result about `inplace`; it is a
*non-result*, and the cause is the reproducibility failure recorded in §6.

Headline metrics were identical across all three arms (`time_f1=0.5`, `joint_f1=0.5`,
`content_acc=1.0`, `n_emits=3`, `ctrl_p_hit` mean 0.58), so nothing observed suggests
`inplace` is wrong — there is simply no evidence either way.

**Cost, for the record** (from the profile summaries; this clip only, ~4.6k-token cache):

| | snapshot | inplace |
|---|---|---|
| cache op | `snapshot_clone` ×25 — 0.21 s total, 8.4 ms mean, 21.4 ms max | `ctrl.borrow.begin/end` ×25 — **0.01 s total**, 0.5 ms max |
| controller gen | mean 1.76 s/tick | mean 1.99 s/tick |
| total GPU | 5.88 s | 5.72 s |

The copy op itself is ~20x cheaper to eliminate, but at 25 s it is only 8 ms/tick — below
the generation noise, which is why `inplace` measures *slower* here. **This clip is too
short to show the win.** The copy scales with cache length: ~4.6k tokens here vs ~55k on a
300 s clip, so the number that matters has not been measured yet.

**Also missing:** `test_inplace.py` output was never captured to disk. The mechanism proof
(T1-T5) — which runs both modes *inside one process*, where the model is fixed and the
reproducibility problem does not apply — is the one piece of evidence that would still be
valid, and there is no record of it.

### Leading hypothesis for the blocker

`writer_greedy = False` (config.py:249, changed 2026-07-30 to fix frozen perception). The
sampler's generator *is* seeded (`writer_seed=3407`), so the uniform draws are
reproducible — but `torch.multinomial` over bf16 probabilities **amplifies** kernel-level
non-determinism: one near-tie resolves differently, one token changes, and the rest of the
stream diverges from there. Under greedy argmax the same jitter is almost always absorbed.
This connects §6's open "root cause not yet found" to a specific, testable mechanism.

### Next step (cheap — 3 arms x ~1 min on one clip)

1. **Re-run the null control with `writer_greedy=true`** on the same split.
   - Clean ⇒ the non-determinism is sampling amplification. Greedy then gives a valid A/B
     harness for `inplace` **and for every other two-process comparison in this project**,
     which is worth far more than the cache change.
   - Still dirty ⇒ the jitter is in the kernels, and no two-process A/B here is
     trustworthy. Fall back to in-process comparison for everything.
2. **Run `test_inplace.py` and save the output**, at `--tokens 2000` for T1-T4 and
   `--only-cost --tokens 20000` for a realistic T5. This is valid regardless of step 1.
3. Only then re-run `ab_inplace.sh`, and **on a long clip** — the copy cost is invisible at
   25 s.

**Status: parked.** `controller_cache_mode` defaults to `snapshot`, so nothing in the
benchmark path is affected. Do not switch the default until step 1 or step 2 passes.
