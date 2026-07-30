# ROADMAP — System 3 → ICLR 2027

**Today:** 2026-07-30 · **Deadline:** 2026-09-18 AoE · **50 days / 7 weeks**
See `MISSION.md` for the vision, `ICL_DIFF_CONTROLLER.md` for mechanism design.

---

## The organising principle

> **The current idea is half-baked. That is the reason to build the measurement rig
> first — not a reason to postpone it.**

Right now one experiment costs ~1 hour and returns a number noisier than the effect
(`EXPERIMENTS.md`: F1 0.255 vs 0.051 on one config). Under those conditions brainstorming
is *worthless* — you cannot tell a good idea from a bad one, so more ideas just add noise.

After Phase 1, one experiment costs ~5 minutes and returns a trustworthy number. **Then**
you can try thirty ideas in a week. Phase 1 does not delay the research; it is what makes
research possible. Week 3 is a dedicated idea sprint that only exists because of it.

**Standing rule: one variable per experiment.** V3 changed three things at once, regressed,
and taught us nothing. That discipline failure has cost more time than any technical problem.

---

## PHASE 1 — Measurement rig (Week 1: Jul 30 – Aug 6)

*Goal: any variant testable in < 5 min with error bars you trust.*

### 1.1 Schema-walk decoder — `controller.py`
Replace the free-running generate loop with a forced skeleton; model fills only value slots.

- Forced literals prefilled as multi-token embeds (one forward each, not one per token)
- `hit` read as a logit comparison at the forced position → **continuous score**, no decode
- Early exit: `t` / `answer` slots only entered when `hit` fires
- `more` escape hatch: one forced bool; if true, free-decode the tail (fps / next_check_s /
  note / count / phase). Full expressive power, zero cost on quiet ticks.

**What changes:** quiet tick 35 decodes → ~10 forwards (~3.2 s → ~1.1 s). Malformed JSON
becomes impossible. `fps` spam becomes impossible.
**Verify:** log both the decoded `hit` and the logit-read `hit` for one full run; confirm
they agree. *If they disagree, stop and understand why before building on it.*

### 1.2 Strip format from the ICL prompts — `config.py`
Delete the `Fields:` schema block and JSON syntax from all 3 ICL blocks. Keep only task
semantics: what counts as the event, how to phrase the answer.

**What changes:** ~900 → ~500 tokens. Per our own finding ("more prompt = worse"), this
should *help* accuracy, not just latency. **Test it as its own variable.**

### 1.3 One-pass multi-variant sweep — new `omniprofast/sweep.py`
**See MISSION.md INVARIANT 1: the system must never observe the future. A pre-computed
fixture is legal iff it is revealed strictly causally.**

Stream each video forward exactly **once**. At every tick, evaluate **all N variants**
against the cache state at that instant. The controller never rewinds, never looks ahead,
never holds the clip. Persist every variant's per-tick score.

**Optional embedding cache (`omniprofast/embcache.py`)** — for iterating on prompts across
days/jobs without re-running the ViT. Store **vision embeddings**, never the KV cache:
455 MB vs 8.2 GB per 300 s video (8 KB/token vs 144 KB/token, measured), and embeddings
don't bake in `kv_budget` / eviction / timestamp format, so ingestion-side changes remain
testable. Feed them to the ingester **one frame at a time, in order** — the controller
cannot tell the difference from live capture. Assert no eviction before treating any prefix
as valid (262 144 budget / 185 tok per frame ≈ 23 min @ 1 fps).

**Honest note:** ingestion is only ~7% of sweep cost; the one-pass design already amortises
it. The embedding cache is a **workflow** win (decouples "get video in" from "try a prompt"),
not a large speed win.

**What changes:**
- N variants compared in one pass, all seeing **byte-identical** context — a cleaner A/B
  than separate runs can ever give (it removes the cross-run variance that made
  F1 0.255 vs 0.051 possible in the first place).
- Ingestion amortised across variants.
- Gate-threshold sweeps become a numpy loop over the **saved scores** — no GPU,
  milliseconds. That is analysing outputs, not replaying video: legal.
- **Valid under eviction and on unbounded streams**, which a cached-prefix replay is not.

**Honest cost note:** decode is ~99% of tick cost and it does *not* amortise — each variant
still decodes at each tick. The speedups come from the schema-walk decoder (~3×) and from
running a coarser tick cadence during development (~3–5×), not from avoiding ingestion.
Realistic target: **~1 h → ~5–10 min per sweep**, plus perfectly matched comparisons.

### 1.4 Dense AUC metric
Label every tick from ground truth; compute AUC / average-precision over ~3,300 decisions.

**What changes:** iteration signal goes from ~33 events to ~3,300 points. Separates
"perception is wrong" from "gate is mistuned" — currently indistinguishable.
**Rule:** AUC for iteration only. Every *reported* number stays OmniPro F1.

### 1.5 ⚠️ RoPE position re-basing — a correctness bug in the core claim
**Found 2026-07-30 while assessing omni backbones; it affects the CURRENT 8B system.**

`manager.py` treats `next_pos` as a monotonic logical clock that "survives eviction"
(`self.next_pos += embeds.shape[1]`, never rebased — lines 52/71). Eviction bounds
*memory* but not *position*. Qwen3-VL-8B's `max_position_embeddings` is 262 144, and at
185 tok/frame @ 1 fps we cross it after **23.6 minutes of video**. Past that the model is
extrapolating RoPE beyond its trained range.

This is precisely the mistake StreamingLLM warns about: positions must be assigned
**within the cache window**, not from the original sequence. So the system that claims
*unbounded* streaming currently degrades at ~23 min — and every eval so far
(`max_seconds=300`) has run comfortably inside that window, which is why it was never seen.

**Fix:** derive `pos_start` from the physical window rather than a monotonic counter. Safe
for this design because the model reads time from the *text timestamp tokens*, not from
RoPE positions. Removes the horizon limit entirely.

**Also blocks the omni option:** Qwen3-Omni's `max_position_embeddings` is 65 536 and
Qwen2.5-Omni's is 32 768 → ~5.9 min and ~3 min respectively. Any omni swap requires this
fix first.

> **Phase 1 gate:** two runs of the same config must give the same AUC to 3 decimals, a
> full sweep must finish in < 10 min, **and a >25-min stream must not degrade**. Do not
> proceed until all three hold.

---

## PHASE 2 — Rebuild history, then the core claim (Week 2: Aug 7 – Aug 13)

### 2.1 Re-run every past variant on the new rig
baseline / V1-edge / V2-evidence / V3 / v2best — same rig, same seed, same judge cache.

**What changes:** you learn which "regressions" were real. Live possibility: **V3 never
regressed and we deleted a good idea based on noise.** Cheap to find out now.

### 2.2 Ablate V2 properly — one variable at a time
`seen` on/off · `event_time_s` on/off · level-edge vs raw level. Three runs, ~15 min total.

**What changes:** the F1 0.0 → 0.255 story becomes attributable instead of folklore. This is
a paper table.

### 2.3 Split trigger from writer — the core claim
- **TRIGGER:** own thread, always on, one forward pass, logit read, Schmitt gate (hyst2b).
  Never decodes. Target < 150 ms.
- **CONTROLLER/WRITER:** runs on trigger fire, plus a slow background tick to refresh
  `seen` / deferred questions / memory.

**What changes:** timing precision stops depending on decode speed. This is the
training-free answer to MiniCPM-o's trained listen/speak token — **the paper's thesis.**

### 2.4 Reframe the head-to-head as an ablation
trigger-only · controller-only · both. They were never rivals.

---

## PHASE 3 — Idea sprint (Week 3: Aug 14 – Aug 20)

*The half-baked part. Now affordable: ~5 min per idea, trustworthy numbers.*

Run wide, kill fast, one variable each. Candidate pool (extend freely):

| # | Idea | Hypothesis |
|---|---|---|
| A | Trigger question phrasing sweep (10–20 variants) | The static probe sentence is unoptimised; cheap upside |
| B | `seen` in the trigger vs not | Does perception grounding help a *logit* read, or only a decode? |
| C | Multi-question trigger (2–3 probes, max-pooled) | One question can't cover an abstract condition |
| D | ICL in the pinned sink vs spliced per tick | Prefill once; tests recency-vs-cost |
| E | Calibrated threshold per task | Gate tuning is now free — why hand-pick 0.5? |
| F | fps steering on/off | Does self-paced perception earn its complexity? |
| G | Deferred `question_for_next` on/off | The football-example mechanism — does it pay? |
| H | Frame token budget: 180 vs 90 vs 360 | Accuracy/latency curve; feeds the pruning story |
| I | Trigger cadence: every frame vs every 0.5 s vs 1 s | Where does ±3 s actually break? |
| J | Backdating window for `event_time_s` | Cheap timing win, already half-built |

**Deliverable:** a ranked table of what moves AUC and what doesn't. Lock the winners.
**Do not** start GEPA until this is done — hand-search first tells you whether the search
space is even worth automating.

---

## PHASE 4 — Memory + the tasks that carry the paper (Week 4: Aug 21 – Aug 27)

### 4.1 Three-tier state + memory
`reported` (code-owned, model-unwritable) · `notes` ring (append-only, survives eviction) ·
`count` / `phase` accumulators.

**Critical path:** unlocks the four tasks holding **358 of our 432** vision-only samples.

### 4.2 Task expansion, ordered by sample count
1. `realtime_state_monitor` (n=98) — needs `phase`
2. `cumulative_counting` (n=73) — needs `count`
3. `dedup_counting` (n=121) — `count` + identity memory. **The flagship**: only 17/300
   samples need audio, and it's the hardest cognitive level (Reasoning).
4. `snapshot_counting` (n=35)
5. `sequential_step_instruction` (n=66)

Keep `explicit_target_grounding` (n=6) as a qualitative demo. **Drop `event_narration` (n=5).**

### 4.3 Eviction stress test
Run a video long enough to force eviction; show the `notes` log preserves what the cache
loses. **This is the evidence for the unbounded-stream claim** — OmniPro reports long-term
trigger retention at only 37% across all models, so this is a differentiator if it holds.

---

## PHASE 5 — GEPA + the rival (Week 5: Aug 28 – Sep 3)

### 5.1 GEPA over ICL semantics
Standalone [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa), **not** DSPy (its adapters
fight the KV-splice pipeline; see the DSPy analysis).
- unit = one video · metric returns `{"score": AUC, "feedback": <failure trace in words>}`
- reflection LM = Gemini (already wired for judging)
- search space = **task semantics only** — format is locked by the decoder

**What changes:** prompt improvement stops depending on guessing what the model
misunderstood. Budget `auto="light"` first; measure cost before scaling.
**Kill criterion:** if GEPA does not beat the Week-3 hand-searched best, drop it and say so.

### 5.2 Re-run MiniCPM-o 4.5 on our exact subset
Same videos, same audio filter, same judge, same ±3 s protocol.
**Non-negotiable:** without this there is no comparison, only two unrelated numbers.

---

## PHASE 6 — Real-time + ablations (Week 6: Sep 4 – Sep 10)

### 6.1 Free-running mode (`deterministic=False`)
Report separately from all lockstep numbers: p50/p95 speak latency, frame retention,
sustained fps, and how accuracy degrades vs lockstep.

**The gap between lockstep and free-running is a finding.** No prior work reports it.

### 6.2 Final ablation table
architecture (trigger / controller / both) · `seen` · memory · fps steering ·
deferred questions · ICL vs none.

### 6.3 Honest limitations
audio (84% of benchmark excluded) · 8B comprehension ceiling · threads-not-processes ·
subset sizes.

---

## PHASE 7 — Paper (Week 7: Sep 11 – Sep 18)

Thesis: **a frozen VLM already knows when to speak; proactivity is an architecture problem,
not a training problem.**

- Abstract states the vision-only subset **in the abstract**, not buried
- Headline table: lockstep F1 vs MiniCPM-o 4.5 on the identical subset
- Second table: real-time latency (nobody else reports this)
- Architecture figure: the ROS-style process graph
- Ablations, limitations, reproducibility (seeds, judge cache, lockstep)
- **Buffer: 3 days.** Do not plan work into them.

---

## Kill criteria — decide early, not in week 7

| If… | Then… |
|---|---|
| Trigger split doesn't beat controller-only on AUC by Week 2 end | Keep the fused controller; re-pitch around memory + unbounded stream |
| `dedup_counting` doesn't work by Week 5 | Report 4 tasks well rather than 9 badly |
| MiniCPM-o re-run is infeasible | Drop the head-to-head; position as training-free-baseline, cite their number without comparing |
| Phase 1 slips past Aug 10 | Cut Phase 3 to 3 days; the rig is worth more than the sprint |

---

## Immediate next actions

1. `controller.py` — schema-walk decoder + logit-read `hit`, **with both values logged for verification**
2. `omniprofast/sweep.py` — one-pass multi-variant harness (**forward-only; no replay**)
3. AUC metric + per-tick score persistence
4. Re-run V2 / V3 / v2best on the rig

Nothing else starts until (1)–(3) pass the Phase 1 gate.
