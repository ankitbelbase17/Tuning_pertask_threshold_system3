# EXPERIENCE.md — the two proactivity systems: where the code is, what we tried, what we observed

The project vision: humans **look, think, and talk all at once**. We build that from a
single frozen VLM (Qwen3-VL-8B) given a cognitive architecture: an encoder thread
(looking), an ingester that is the sole writer of one shared KV cache (remembering),
and a proactivity mechanism that decides when to speak (thinking + talking).
Two competing designs exist for that last part — this file documents both, because
the head-to-head between them is the core architectural question of the research.

---

## System 1 — PROBE-GATE (fixed-cadence yes/no logit gate + separate writer)

**Where the code is:** branch **`main`** (tip `a6abab2`, "Default output gate to tuned
hysteresis (hyst2)"). Being ported to the current branch as `gate_mode="probe"` so both
systems run under the identical bulletproofed eval.
- `async_omni_v2/proactivity.py` — `yes_share(logits, yes_ids, no_ids)`: one forward
  pass after splicing a yes/no question; the signal is the RELATIVE share
  P(yes)/(P(yes)+P(no)) (absolute probs are tiny/meaningless).
- `async_omni_v2/writer.py` — on trigger: MVCC `snapshot_clone()` of the cache, splice
  `writer_prompt + cue`, free-generate ONE line (the user-facing answer).
- `async_omni_v2/manager.py` (`probe()`) — splice question → read logits → ERASE the
  probe (truncate back) so it leaves no trace in the primary cache.
- `async_omni_v2/input_ingester.py` (on main: gate blocks) — every `goal_gate_every`
  frames run the probe; **Schmitt/hysteresis gate** decides firing.

**How it decides:** "Do we have enough info / has the event happened?" is asked as a
LOGIT question every N frames. Edge discipline is code-side hysteresis: fire on a
rising crossing of `gate_high_thr=0.5` while armed; re-arm when the share falls below
`gate_low_thr=0.40` OR `gate_rearm_s=5.0` after a fire ("hyst2b", the tuned best).

**What we tried & observed (27-video, 9-task eval, older scoring):**
- Level-triggered gate (no hysteresis): massive over-firing — every probe above
  threshold re-fires while a condition stays true.
- Hysteresis sweep → hyst2b best: **joint_f1 0.206** (vs 0.149 baseline gate), roughly
  MiniCPM-o-4.5-level on that table. Precision improved without killing recall.
- Strengths observed: cheap (1 forward/frame), timing sharp, simple.
- Weaknesses observed: fixed cadence (no self-pacing), no dedup reasoning (hysteresis
  is signal-shape only, blind to content), the yes/no question is one static sentence
  (no per-tick perception step), writer is fire-and-forget (no memory of what it said).
- **NEVER run on the 11-video SCA subset under the new deterministic/judge≥3 protocol**
  — that's the missing number this head-to-head fills.

## System 2 — ICL CONTROLLER (generative self-scheduling; "icl_ingester_writer")

**Where the code is:** branch **`icl_ingester_writer`** (current; V2-pure merged at
`84a44e6`, lockstep at `56d1944`).
- `async_omni_v2/controller.py` — each self-scheduled tick: snapshot the cache, splice
  the ICL prompt (+ timestamped "Already reported" history), generate one compact JSON
  **diff** of a persistent control state; code applies it.
- `async_omni_v2/config.py` — `task_controller_prompts` (per-task ICL; SCA = V2-pure:
  `seen` + `event_time_s` + level semantics), sampling preset (greedy), check grid
  0.2–1.5s, fps 1–3.
- DSL per tick: `seen` (look before judging), `have_enough_info` (LEVEL: satisfied
  NOW?), `event_time_s` (model reads the clock for the onset), `answer` (what+why,
  <25 words), `fps`, `next_check_s`, `question_for_next`.
- Edge discipline is code-side here too: fire on the rising edge of the level
  (semantic Schmitt gate) or a clearly different answer (word-overlap <0.5) within a
  true stretch.

**What we tried & observed (11-video SCA subset; full log in EXPERIMENTS.md):**
- Model self-dedup (`new_event` decided in the model's head): 7× identical re-fires →
  the model CANNOT track novelty internally. → moved edge detection to code.
- Empty-`{}` quiet diffs: model went passive (0 emits in ~300 ticks) → booleans must be
  re-asserted every tick (forces the judgment).
- **`seen` field = biggest accuracy lever** (V1 without it: silent, F1 0.0; V2 with it:
  best observed F1 0.255). `event_time_s` fixes ±3s near-misses.
- More prompt = worse (V3's extra example + calibration regressed).
- Latency: decode dominates (~94–120 ms/tok after the GPU-decode fix; 1.6× from
  removing the per-token CPU logit round-trip); diff-decoding makes quiet ticks ~14 tok.
- Strengths observed: self-pacing, content-aware dedup, per-tick perception (`seen`),
  one mechanism = gate + writer, task-adaptable via ICL only.
- Weaknesses observed: ~30-token ticks vs 1 forward for the probe (cost), recall
  ceiling on abstract visual conditions, sensitive to prompt phrasing.

## The shared substrate (both systems)

- Single shared KV cache; ingester is the SOLE writer; readers take MVCC snapshot
  clones; StreamingLLM eviction with pinned system-prompt sink; two clocks (logical
  RoPE pos vs physical length) survive eviction.
- **Deterministic lockstep walk** (`56d1944`, `cfg.deterministic=True`): frames feed
  one-by-one; clock published only after a frame is fully in cache; the feed waits for
  every due decision to complete → every decision sees exactly frames [0..vt],
  bit-reproducibly. This matches the OmniPro paper's online protocol ("frames one by
  one"; response timestamp = video-second of emission).
- **Reproducible judging:** Gemini judge pinned with `seed` + verdicts persisted in
  `omniprofast/judge_cache.json`; threshold ≥3 (paper).
- Why all this: we PROVED run-to-run variance (same config: F1 0.255 vs 0.108 vs
  0.051) exceeded every variant difference — async snapshot race + stochastic judge.
  No comparison is meaningful without the lockstep + seeded judge.

## The head-to-head (this experiment)

Same 11 SCA (audio=none) videos, same lockstep determinism, same seed, same judge
cache, ±3s greedy match, judge≥3:
- **probe-hyst2b**: probe every frame @1fps (≈ the controller's ~1s check grid),
  hyst2b gate, writer answers with the same answer-format instruction (what+why,
  <25 words) — same task knowledge, different architecture.
- **controller V2-pure**: as merged (seen + event_time_s + level-edge + greedy,
  grid 0.2–1.5s).

Decision rule afterwards: lock the winner per task in EXPERIMENTS.md (with its
reproducible run artifact), then expand task-by-task with minimal per-task ICL, in
mechanism-reuse order: instant_event_alert → realtime_state_monitor → counting family
(persistent `count` in the diff state) → explicit_target_grounding → narration/step.
