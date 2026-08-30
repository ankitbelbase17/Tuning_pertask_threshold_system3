# icl_ingester_writer — Experiment Log (semantic_condition_alert)

**Method (the research):** a single frozen Qwen3-VL-8B acts as its own perception
scheduler. One in-context-taught control-JSON DSL, emitted over a shared streaming
KV cache, simultaneously steers input sampling (fps), self-schedules its next check
(`next_check_s`), gates output (`have_enough_info`), and writes the alert. No training.

**Eval:** OmniPro online mode (arXiv 2605.18577). 11 `semantic_condition_alert`
videos, `audio_dependency=none`. Frames fed at fixed fps (NOT wall-clock); a
response's timestamp = the video-second it's emitted; ±3s greedy temporal match;
time-F1 = harmonic mean of match precision/recall; joint-F1 = match must ALSO be
content-correct; content judged by Gemini (paper: score ≥3 correct).

---

## ACCURACY — what we tried (SCA, 11 videos)

| # | variant | mechanism | emits | tp | time_f1 | note |
|---|---|---|---|---|---|---|
| 0 | baseline "trust new_event" | model self-dedups in its head | 20 | 3 | **0.115** | Kabul alert fired 7× (identical) |
| 1 | booleans-always diff | reset booleans each tick, emit `{}` when quiet | ~14 | 1 | ~0.06 | some spam, some silent |
| 2 | V1 edge | model reports LEVEL; CODE fires on rising edge; greedy | 3 | 0 | **0.0** | too conservative → silent |
| 3 | **V2 evidence** | + `seen` (look-before-judge) + `event_time_s` | 15 | 5 | **0.255** | **BEST so far (~2.2× baseline)** |
| 4 | V3 | + 2nd (visual) ICL example + calibration + gate 0.5→0.35 | 8 | 3 | 0.15 | regressed |
| 5 | v2best | V3-lineage + finer grid 0.2–1.5s + fps 1–3 + judge≥3 | 7 | 1 | 0.051 | regressed further |

### What WORKED
- **`seen` field (look-before-judge):** biggest single accuracy lever. Forcing the
  model to write what's on screen *before* the boolean grounds the judgment
  (chain-of-thought compressed into one JSON field). Helped both *whether* and *when*.
- **`event_time_s` (model reads the clock):** records the model's stated onset time
  (clamped to [vt−10, vt]) instead of the tick time → fewer ±3s near-misses.
- **level→edge in CODE, not in the model:** model judges "is the condition true NOW",
  the controller fires only on the rising edge (a semantic Schmitt gate). This is the
  right architecture (the model cannot track "already reported" reliably).

### What did NOT work
- **Model self-dedup (`new_event` in its head):** re-fired the same event 7× (identical text).
- **Empty `{}` diff on quiet ticks:** model went passive → 0 emits across ~300 ticks.
- **V1 edge alone:** without `seen`, too conservative → silent on most videos.
- **V3 additions (2nd example + "more-likely-than-not" calibration + tighter gate):**
  regressed; over-instructing a frozen 8B hurts.
- **Recall ceiling:** several videos emit 0 across EVERY variant — the 8B genuinely
  never recognizes those abstract/visual conditions (comprehension limit, not prompt).

### BIGGEST CAVEAT (proven 2026-07-14) — results are NOT trustworthy yet
Run-to-run **variance exceeds the variant differences**: a near-identical config
scored time_f1 **0.255 one run and 0.051 another**. Telemetry showed **zero frames
dropped** (frames_emitted == frames_ingested on all 11 videos), so it is NOT frame
drops — it is the **async thread race**: the controller's snapshot each tick captures
whatever the ingester happened to push by that wall-clock instant, and thread
scheduling varies every run → different cache → different greedy output. **=> every
accuracy comparison above is within noise.** Must make the pipeline deterministic
(lockstep frame-indexed walk, no wall clock) before trusting any F1.

---

## LATENCY — what we tried (writer/controller decode, GH200, 8B bf16, sdpa)

| lever | result | status |
|---|---|---|
| baseline: forward ships 151k-vocab logits to CPU each token | 149 ms/tok | the bug |
| **keep logits on GPU + sample on GPU** | 94 ms/tok (**1.6×**) | ✅ committed |
| + reuse position tensors | 84 ms/tok (1.8×) | not implemented |
| **skip lm_head for ingest** (want_logits=False) | free on ingest | ✅ committed |
| deepcopy KV snapshot | 5–18 ms even at 24k | NOT a bottleneck |
| multi-GPU split (controller own GPU) | 107–407 ms/tok, no better | ✗ cross-GPU cache copy cancels it |
| torch.compile (reduce-overhead) | FAILS on DynamicCache | ✗ needs StaticCache |
| flash-attention-2 | not built for this aarch64 env | ✗ sdpa is best |
| in-pipeline telemetry | decode ~120 ms/tok dominates; prefill ~0.23s | confirmed |

**Latency conclusion (SOLID, no confound):** the CPU logit round-trip was the bug;
GPU-resident decode is a real 1.6×. Decode dominates (~120 ms/tok); ICL prefill is
negligible (~0.23s). Remaining ~45 ms/tok floor is per-call launch overhead →
StaticCache + CUDA graphs is the only remaining lever (future work).

---

## BEST-OF-BEST
- **Accuracy:** V2 "evidence" — **time_f1 0.255 / joint_f1 0.186** (`seen` + `event_time_s`
  + level-edge). ~2.2× the 0.115 baseline. ⚠️ confounded by non-determinism.
- **Latency:** GPU-resident decode — **149→94 ms/tok (1.6×)**, committed & clean.

## NEXT (in order)
1. **Deterministic eval mode** (frame-indexed lockstep walk) — unblocks all accuracy claims; matches the paper's "frames one by one".
2. Re-measure baseline vs V2 deterministically → confirm the `seen` win is real.
3. Ablate V2 (which of `seen` / `event_time_s` / edge drives the gain).
4. Attack the recall ceiling: at t=0 compile the abstract condition into concrete
   visual checks (VISPROG-style), then let `seen` check against them.

## Ops note
Scratch conda env (`prosync_env`) periodically loses its stdlib (`encodings` wiped) →
`init_fs_encoding` LookupError. Fix: `bash recreate_env.sh` (rebuilds from the intact
home miniforge). Login node has 4 GH200s usable via CUDA_VISIBLE_DEVICES; SLURM debug
partition = 1 job/user (an auto-respawning bash can hog it).
