# ICL Diff-Controller — design, measurements, and the plan

**Branch:** `icl_ingester_writer`  ·  **Status:** design locked, implementation pending
**Companion docs:** `EXPERIENCE.md` (the two proactivity systems), `EXPERIMENTS.md` (run log),
`SYSTEM3_TECHNICAL_ARCHITECTURE.md` (the substrate).

---

## 1. The research idea

Humans **look, think, and talk at the same time**. We build that from a single *frozen*
VLM (Qwen3-VL-8B) plus a cognitive architecture — no fine-tuning, no extra heads.

The system is an **asynchronous loop over an unbounded video stream** that proactively
decides *what to say* and, critically, *when to say it*:

```
  config.json  ──seeds──▶  persistent control state
                                  │
   frames ──▶ ENCODER ──▶ INGESTER ──▶ [ shared KV cache ]   (sole writer)
                                              │
                                              │ MVCC snapshot / splice
                                              ▼
                                          WRITER/CONTROLLER
                                              │
                                     emits a JSON **diff**
                                              │
                                    code merges diff into state
                                              │
                            ┌─────────────────┼─────────────────┐
                            ▼                 ▼                 ▼
                      steer encoder    schedule next      emit answer
                      (input gate)      check (pacing)    (output gate)
```

* **Ingester** — does not think. Ingests frames into one shared KV cache, is the *sole
  writer*, bounds memory (StreamingLLM eviction with a pinned system-prompt sink), and
  publishes video time on a shared clock.
* **Writer/Controller** — reads that cache each tick and emits **one compact JSON diff**
  describing what changed and what should happen next. Code applies the diff.
* **Memory** — the diff carries what has happened *and* what the writer has already told
  the user, so the model can avoid re-reporting and can survive cache eviction.

**Hard requirement:** a proactive answer must land within **±3 s** of the ground-truth
moment in the video.

**Benchmark:** OmniPro, **9 tasks**, run at `audio_dependency=none` (the pipeline ingests
no audio, so this is the only honest subset). Each task needs its own ICL prompt.

**Coverage today — 3 of 9 tasks have prompts and results:**

| task | ICL prompt | notes |
|---|---|---|
| `semantic_condition_alert` | ✅ `_SEMANTIC_CONDITION_ALERT` | V2-pure; best observed F1 0.255 |
| `explicit_target_grounding` | ✅ `_EXPLICIT_TARGET_GROUNDING` | 3×3 grid, exact-match scored |
| `instant_event_alert` | ✅ `_INSTANT_EVENT_ALERT` | sharp onset |
| `snapshot_counting` | ❌ | needs persistent `count` |
| `cumulative_counting` | ❌ | needs persistent `count` |
| `dedup_counting` | ❌ | needs persistent `count` + identity memory |
| `realtime_state_monitor` | ❌ | needs persistent `phase` |
| `event_narration` | ❌ | gpt-judged free text |
| `sequential_step_instruction` | ❌ | gpt-judged free text |

---

## 2. Measured problem: the diff is not a diff

Observed controller tick (verbatim from the run log, `instant_event_alert`, video
`26BT7YzpzQs`):

```
{"seen": "dark screen with glowing blue particles", "have_enough_info": false, "fps": 1.0, "next_check_s": 1 }
```

Tokenized with the Qwen3-VL-8B tokenizer — **35 tokens**, matching the log's `ntok=35`:

| segment | tokens | decided by |
|---|---:|---|
| `"seen": "` | 4 | **code** (constant) |
| `dark screen with glowing blue particles` | 6 | model |
| `", "have_enough_info": ` | 8 | **code** (constant) |
| `false` | 1 | model |
| `, "fps": ` | 5 | **code** (constant) |
| `1.0` | 3 | model (`1` would cost 1) |
| `, "next_check_s": ` | 7 | **code** (constant) |
| `1` | 1 | model |
| ` }` | 1 | **code** (constant) |
| **structure** | **25 (71%)** | 100% forceable |
| **values** | **11 (29%)** | the only real decisions |

Three separate defects fall out of this:

**(a) 71% of the decode budget is punctuation.** The model sequentially re-types key
names that are already written in the prompt.

**(b) The diff instruction is ignored, 0/15.** The ICL says emit `fps` / `next_check_s`
*only when they change*. In all 15 logged ticks they are emitted every time, always `1.0`
and `1`. That is **16 of 35 tokens — 46% of the budget — carrying zero bits of
information across the entire run.**

**(c) Floats cost 3× integers.** `1.0` → `[16, 13, 15]`; `1` → `[16]`.

> ### Conclusion 1 — a diff is a decoder constraint, not a prompt instruction.
> This is the same failure mode as `new_event` in `EXPERIENCE.md`: the model could not
> track novelty in its head, so edge detection moved into code and it worked.
> "Omit the field if unchanged" is the identical ask — silent bookkeeping against a
> state the model cannot see — and it failed identically.
> **The model must be *physically unable* to emit the redundant tokens.**

---

## 3. The lever: prefill/decode asymmetry

```
prefill of k tokens = 1 forward pass  (parallel)
decode  of k tokens = k forward passes (sequential)
```

Every token we **force** instead of **sample** is nearly free. The 25 structure tokens are
constant strings → 4 batched prefills instead of 25 sequential decodes.

Further, `true` / `false` are **single tokens** (ids `1866` / `3849`). After prefilling the
literal `","hit":`, the returned logits sit *exactly* at the boolean's position — so the
level is read as a **logit comparison with zero decode steps**, and yields a *continuous
confidence* rather than a hard bit.

> ### Conclusion 2 — the ICL controller absorbs the probe-gate.
> A continuous `P(true)` at a forced position is precisely the probe-gate's `yes_share`.
> The ICL controller therefore inherits the tuned Schmitt/hysteresis gate (`hyst2b`) for
> free. The two arms of the head-to-head stop being alternatives and become one
> mechanism: **generative perception (`seen`) + logit-read level + code-side edge.**

---

## 4. The ±3 s budget, honestly

Real-world lag ≈ `next_check_s` + `gen_s`.

| | now | required |
|---|---|---|
| quiet tick `gen_s` | 3.2 – 8.8 s | — |
| hot tick `gen_s` (+~25 tok for `answer`) | ~6 – 11 s | — |
| total lag | **4 – 11 s** | **≤ 3 s** |

We are **2–4× over budget, and worst precisely at the moment timing matters.**

`deterministic=True` currently hides this: the ingester stops the world while the
controller thinks, so the frame-indexed clock stays honest and ±3 s scoring remains valid.
That is correct for reproducibility, but must be stated plainly:

> **The ±3 s claim today is only defensible because the world stops. It is not yet a
> real-time system.** Closing that gap is the point of this work.

Budget arithmetic for genuine real-time at 1 fps with a 1 s check grid:
one video-second must cover one frame ingest (~185 tok prefill, ~60 ms) plus one
controller tick → **tick budget ≈ 1.5 s → ≈ 16 forward passes at ~90 ms each.**

> ### Conclusion 3 — decouple timing from content.
> Fire on the boolean (cheap, ~1 s), **stamp the timestamp**, *then* generate the answer
> prose. The generation latency of the text no longer contaminates the recorded time.
> This is what the probe-gate's separate writer got right and the fused controller lost.

---

## 5. Memory: three memories, not one

A single `memory` field that the model **rewrites** each tick costs tokens proportional to
memory length, every tick — so tick latency grows with watch time. The system would get
*slower the longer it watches*, which is exactly backwards for an unbounded stream.

> ### Conclusion 4 — memory is a **log**, not a **document**.
> Append-only, bounded ring, **0 tokens on quiet ticks**.

Three distinct things are being conflated under "memory". They must stay separate:

| # | memory | owner | mechanism | why |
|---|---|---|---|---|
| 1 | **what I told the user** | **code** | `reported: [(t, answer)]`, append-only | If the model could write this it could hallucinate having already answered. `EXPERIENCE.md` already proved it cannot track this internally. Rendered into the prompt as timestamped history. |
| 2 | **what happened in the video** | model | optional `"note"` key → appends to a bounded `notes` ring | The KV cache *is* this memory — **until it evicts**. At `kv_budget=262144` / ~185 tok per frame ≈ **1400 frames ≈ 23 min @ 1 fps**. Past that the cache forgets and `notes` is the only survivor. **Load-bearing for the unbounded-stream claim.** |
| 3 | **task accumulators** | model | `count` (int), `phase` (str) | 1 token to update. Unlocks `snapshot_counting`, `cumulative_counting`, `dedup_counting`, `realtime_state_monitor` — 4 of the 6 missing tasks. |

---

## 6. The config JSON

The run-level config that seeds a session. Fields marked **required**.

```jsonc
{
  // ---- identity (required) ----
  "task":        "instant_event_alert",   // selects tick schema + ICL block
  "video_id":    "26BT7YzpzQs",
  "instruction": "Let me know when the audience starts clapping",
  "event":       "audience starts clapping",
  "audio":       "none",

  // ---- what the model is asked to produce (required) ----
  "schema": "onset",        // onset | condition | grounding | count | state | narrate
  "icl":    "instant_event_alert",

  // ---- bounds: code clamps every model-supplied value into these ----
  "bounds": {
    "fps":            [1, 3],
    "next_check_s":   [0.2, 1.5],
    "max_seen_tokens":   12,
    "max_answer_tokens": 28,
    "notes_ring":         8,      // bounded memory -> O(1) tick cost
    "kv_budget":     262144
  },

  // ---- initial persistent control state ----
  "init": {
    "fps": 1,
    "next_check_s": 1,
    "question_for_next": "",
    "count": 0,
    "phase": "",
    "notes": []
  },

  // ---- gate policy (code-owned, never model-visible) ----
  "gate": {
    "edge":             "rising",   // rising | hysteresis
    "high_thr":         0.5,        // on P(true) from the logit read
    "low_thr":          0.40,
    "rearm_s":          5.0,
    "distinct_sim_thr": 0.5,        // word-overlap dedup within a true stretch
    "backdate_max_s":   10.0        // clamp on model-supplied event_time_s
  },

  // ---- determinism ----
  "seed": 0,
  "writer_seed": 3407,
  "greedy": true,
  "deterministic": true
}
```

### The three state tiers

The single most important design rule is **which tier a field lives in**.

**Tier 1 — TRANSIENT.** Reset to default every tick; the model must re-assert. These are
the judgment, and re-asserting is what *forces* the judgment (empty-`{}` quiet diffs made
the model go passive — 0 emits in ~300 ticks).

| field | type | when |
|---|---|---|
| `seen` | str (≤12 tok) | **always, first** — look before judging |
| `hit` | bool | **always** — condition satisfied *now* (a LEVEL, not an edge) |
| `t` | int | only when `hit` — onset second, read off the `time Xs` markers |
| `answer` | str (≤28 tok) | only when `hit` — deferred to the writer pass |

**Tier 2 — PERSISTENT.** Survive across ticks; changed **only** when the model explicitly
emits the key. *This* is the real diff surface.

`fps` · `next_check_s` · `question_for_next` · `count` · `phase` · `note` (appends to `notes`)

**Tier 3 — CODE-OWNED.** Never emitted by the model, never parsed from its output.

`reported` · `prev_hit` · `vt` · `armed` · `last_fire_vt`

---

## 7. How the diff is generated and merged

### Generation — a schema-walked decoder, not a free-form JSON dump

Code walks a fixed spine. Constant literals are **prefilled** (1 forward each, batched);
only value slots are **sampled**.

```
  prefill  {"seen":"              (4 tok, 1 fwd)
  DECODE   <seen text>            (~6 fwd, stop on '"', cap 12)
  prefill  ","hit":               (~6 tok, 1 fwd)  -> logits land ON the bool slot
  READ     P(true) vs P(false)    (0 fwd)          -> continuous confidence
  if hit:
    prefill  ,"t":                (1 fwd)
    DECODE   <int>                (1-2 fwd, stop on non-digit)
    [answer deferred to the writer pass -- see Conclusion 3]
  prefill  ,"more":               (1 fwd) -> logits land on the escape-hatch bool
  READ     P(true) vs P(false)    (0 fwd)
    if more: free-decode the tail (fps/next_check_s/note/count/phase) until '}'
  else:    done
```

**The `more` escape hatch** is what preserves expressive power without paying for it: the
model keeps full ability to change cadence, append memory, or bump a counter, but the
*default* costs **1 forward and 0 decodes**. Pay only when you speak.

Forward-pass count per quiet tick:

| | today | schema-walked |
|---|---:|---:|
| structure | 25 sequential decodes | 4 batched prefills |
| `seen` | 6 decodes | 6 decodes |
| `hit` | 1 decode | **0** (logit read) |
| `fps` + `next_check_s` | 4 decodes | **0** (Tier 2, unchanged) |
| **total forwards** | **35** | **~10** |
| **est. wall @ ~90 ms** | ~3.2 s | **~0.9 s** |

≈ **3.5× fewer forward passes**, landing a quiet tick under the 1.5 s real-time budget.
Hot tick (to *stamp the time*, answer deferred) ≈ 13 forwards ≈ 1.2 s.

### Merge — code-side, deterministic

```python
# 1. reset Tier 1 to defaults (forces re-judgment every tick)
state.update(seen="", hit=False, t=None, answer="")

# 2. apply the emitted keys
for k, v in diff.items():
    if k == "note":          notes.append((vt, v)); notes = notes[-ring:]   # APPEND, never rewrite
    elif k in TIER1|TIER2:   state[k] = v
    else:                    log_violation(k)                               # Tier 3 is unwritable

# 3. clamp every numeric into config bounds
state.fps          = clamp(state.fps, *bounds.fps)
state.next_check_s = clamp(state.next_check_s, *bounds.next_check_s)

# 4. edge detection in CODE (the model only ever reports a LEVEL)
rising   = state.hit and not prev_hit
distinct = state.hit and prev_hit and word_sim(answer, reported[-1]) < distinct_sim_thr
fire     = rising or distinct
```

---

## 8. Open questions — measure before optimizing

1. **Where does `gen_s` actually go?** The log shows `gen=8.8 s` and `gen=5.9 s` at an
   *identical* `ntok=47`. A 60% spread at constant token count means something other than
   decode is varying — GPU contention with the encoder/ingester, or `snapshot_clone`.
   `ctrl_prefill_s`, `ctrl_decode_s` and `snapshot_clone` are **already profiled** — read
   them before touching anything.
2. **`snapshot_clone` cost.** A full `deepcopy` of the KV cache runs *every tick*. At 300 s
   of video that cache is ~55 K tokens (order-GB). In `deterministic` mode the ingester is
   already blocked, so the clone may be droppable entirely in favour of
   splice-then-truncate under the lock (exactly what `manager.probe()` does).
3. **ICL placement.** The per-task ICL is **669–944 tokens re-prefilled every tick**
   (`_SEMANTIC_CONDITION_ALERT` 944, `_EXPLICIT_TARGET_GROUNDING` 669,
   `_INSTANT_EVENT_ALERT` 891). Moving it into the pinned system-prompt **sink** would make
   it prefill *once*. Trade-off: it then sits thousands of tokens back instead of adjacent
   to the decision point, and `EXPERIENCE.md` records real sensitivity to prompt changes
   ("more prompt = worse"). **Clean A/B: ICL-in-sink vs ICL-spliced.**
4. **Does the logit-read `hit` match the decoded `hit`?** Must be validated before it
   replaces free decode — cheap to check by logging both for one run.

---

## 9. Implementation order

1. **Schema-walked decoder** in `controller.py` — forced literals, sampled slots, logit-read
   booleans, `more` escape hatch. *Biggest latency win, no accuracy risk.*
2. **Three-tier state + merge rules** — Tier 3 becomes model-unwritable.
3. **`notes` ring + `count` + `phase`** — the memory design, append-only.
4. **Deferred answer writer** — decouple timing from content.
5. **Re-run the head-to-head** (11 SCA videos, seeded judge, lockstep) — confirm accuracy
   held while latency dropped.
6. **Expand tasks** in mechanism-reuse order:
   `instant_event_alert` → `realtime_state_monitor` (`phase`) → counting family (`count`)
   → `explicit_target_grounding` → narration / step instruction.

Every change is gated on the deterministic lockstep + seeded judge cache: run-to-run
variance (F1 0.255 / 0.108 / 0.051 on one config) once exceeded every variant difference.
**No comparison is meaningful without it.**
