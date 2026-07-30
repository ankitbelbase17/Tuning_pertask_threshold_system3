# PROMPT_AUDIT — do the 9 ICL blocks say what the researcher intends?

**Date:** 2026-07-30 · **Branch:** `icl_ingester_writer` · **Read-only audit, no code changed.**
Audited against `MISSION.md` (4 invariants + 8 pillars), `PROMPT_STRATEGY.md`,
`ICL_DIFF_CONTROLLER.md`, `ROADMAP.md`.

Files audited: `async_omni_v2/prompts.py` (all prompt text), `async_omni_v2/config.py`
(wiring + decoder settings), `async_omni_v2/controller.py` (the schema walk that decides
which prompt fields are physically reachable), `omniprofast/system5_adapter.py` (runtime
selection), `omniprofast/metrics.py` (the scorer), plus the 3,049 controller ticks in
`omniprofast/output_all/g0*/run_20260730_184923.log`.

---

## 0. The two findings that dominate everything else

Before the per-task table, two facts that change how every row should be read.

### 0.1 The `more` escape hatch never opens — so `count`, `phase`, `question_for_next` and `note` are unreachable text

`decode_mode="schema"` is the default (`config.py:95`). The schema walk
(`controller.py:146-252`) force-feeds this spine and nothing else:

```
{"seen":"<decode>","have_enough_info":<logit read>[,"event_time_s":<decode>,"answer":"<decode>"],"more":<logit read>
```

Every other field — `count`, `phase`, `fps`, `next_check_s`, `question_for_next`, `note` —
exists **only** inside the `more` tail, entered when `p_more >= more_threshold` (0.5,
`config.py:127`).

Measured over the newest full run (3,049 ticks, 4 shards):

| quantity | value |
|---|---|
| ticks where `p_more >= 0.5` | **0 / 3049** |
| global **max** `p_more` observed | **0.438** |
| ticks with `count != 0` | **0** |
| ticks with a non-empty `question_for_next` | **0** |
| ticks with a non-empty `notes` ring | **0** |

Per-task `p_more` means: `realtime_state_monitor` 0.247, `cumulative_counting` 0.281,
`dedup_counting` 0.308, `instant_event_alert` 0.327. The hatch is not marginal — it is
~0.06–0.25 below threshold everywhere, always.

**Consequences.** Every sentence in the 6 new ICL blocks that instructs the model to
maintain `count` or `phase` is instructing it to do something the decoder physically
prevents. That is ~1/3 of the semantics of `snapshot_counting`, `cumulative_counting`,
`dedup_counting` and `realtime_state_monitor`. Pillar 7c is not "missing" — it is
**wired to a slot that never opens**. Pillar 4b (deferred question, the football example
MISSION §7 marks ✅) is dead at runtime for the same reason: `pending_q` is always `""`.

### 0.2 `count` and `phase` are never rendered back into the prompt

Even if the hatch opened, the loop is not closed. The per-tick prompt
(`controller.py:342-373`) is assembled from exactly: the task ICL, `pending_q`, the `seen`
trace, the `reported` list, and `notes`. It **never contains `count`, `phase`, `fps` or
`next_check_s`**. So `_SEMANTICS_REALTIME_STATE_MONITOR`'s core instruction —

> `prompts.py:518` `"true ONLY on the tick where you SEE the state become different from phase"`

— asks the model to compare against a variable it cannot see. Same for
`_FORMAT_FIELD_COUNT` (`prompts.py:246-249`) *"Carry it forward on every tick"*.

The one memory that **is** wired and does work is `reported` ("WHAT YOU HAVE ALREADY TOLD
THE USER") and the `seen` trace. Every prompt that leans on *those* is answerable; every
prompt that leans on `count`/`phase` is not.

---

## 1. The table

`level?` = does the block define `have_enough_info` as a LEVEL ("true NOW") rather than an
EDGE ("true only on the tick where it changes")?

| task | prompt exists? | actually wired? | audio-clean? | level-not-edge? | conditional rules found | answer matches scorer? | `have_enough_info` well-posed? | VERDICT |
|---|---|---|---|---|---|---|---|---|
| `instant_event_alert` | ✅ | ✅ | ⚠️ example is an audio event | ✅ LEVEL | 6 | ✅ (`TIME_ONLY` — content never checked) | ✅ yes | **NEEDS-FIX** |
| `semantic_condition_alert` | ✅ | ✅ | ❌ `"crowd roaring"` in a `seen` example | ✅ LEVEL | 6 | ✅ judge; but scorer/dataset disagree on kind | ✅ yes | **NEEDS-FIX** |
| `explicit_target_grounding` | ✅ | ✅ | ✅ | ✅ LEVEL | 6 | ❌ **scorer cannot return 2 of 9 cells** | ⚠️ conjunctive | **NEEDS-FIX** |
| `snapshot_counting` | ✅ | ✅ | ✅ (block); ❌ real questions say "when the narrator mentions…" | ✅ LEVEL | 7 | ✅ digit-first, verified 5/5 | ⚠️ diluted by a 2nd conjunct | **NEEDS-FIX** |
| `cumulative_counting` | ✅ | ✅ | ✅ | ❌ **EDGE** | 8 | ✅ digit-first, verified 1/1 | ❌ no | **BROKEN** |
| `dedup_counting` | ✅ | ✅ | ✅ | ❌ **EDGE** | 10 | ✅ digit-first, verified 25/25 | ❌ no | **BROKEN** |
| `realtime_state_monitor` | ✅ | ✅ | ✅ | ❌ **EDGE** | 8 | ⚠️ points at the wrong vocabulary source | ❌ no (compares to invisible `phase`) | **BROKEN** |
| `event_narration` | ✅ | ✅ | ✅ | ❌ **EDGE** | 7 | ⚠️ 20-40 words vs a 32-token cap | ❌ no (no natural level exists) | **BROKEN** |
| `sequential_step_instruction` | ✅ | ✅ (never actually run) | ✅ | ❌ **EDGE** | 7 | ❌ mandates a `"Step N — "` prefix GT does not use | ❌ no | **BROKEN** |

**Wiring, confirmed:** `prompts.py:716-728` → `config.py:196-197` → `system5_adapter.py:124`
(`task_controller_prompts.get(sample.task, controller_prompt)`). All 9 keys match
`dataset.py:ALL_TASKS` exactly, so the generic fallback is **never** selected. There is no
orphan prompt. `async_omni_v2/task_prompts_all9.py` **no longer exists** — it was folded
into `prompts.py`; the audit brief's mention of it is stale. `_ETG_WRITER_PROMPT` and
`TASK_WRITER_PROMPTS` are reachable only under `gate_mode="probe"` (default is
`"controller"`), i.e. dormant but not orphaned.

---

## 2. Cross-cutting defects (present in all or most blocks)

### 2.1 Every block still carries the conditional emission rules PROMPT_STRATEGY bans

`PROMPT_STRATEGY.md` §1: *"never write a prompt rule of the form 'emit X only if Y'"* —
**zero remaining** is a §7 success criterion. Current count: **65 across the 9 blocks** (6/6/6 in the hand-tuned blocks, 7-10 in each new one).
The repeat offenders, all still live:

`prompts.py:222-224` (in `_FORMAT_HOW`, therefore in all 6 new blocks; the same sentence is
inlined at `:54-56`, `:115-116`, `:173-175` for the 3 hand-tuned blocks):

> `"Include fps, next_check_s or question_for_next ONLY when they change from their current value (otherwise omit them to stay short)."`

This is *the* rule measured at 0/15 compliance in `ICL_DIFF_CONTROLLER.md` §2(b). It is now
also **moot** — the schema decoder makes those fields unemittable on the default path — so
it is pure dead weight teaching a rule the model cannot follow.

`prompts.py:239-240`:

> `"answer : REQUIRED whenever have_enough_info is true -> the sentence the user will actually receive; else \"\""`

`PROMPT_STRATEGY.md` §2 Move 2 explicitly retires this: *"Decoder only enters the answer
slot when the `hit` logit fires — the model never decides."* That is exactly what
`controller.py:219-232` does. Delete the line.

Also still present: `event_time_s : when have_enough_info is true -> …` (`:235-238`),
`count : … raise it by exactly 1 when you report a new item` (`:248-249`),
`phase : … change it only when the state really changes` (`:253-254`), and in dedup
`"If it matches one, it is a REPEAT -> have_enough_info stays false"` (`:441-443`).

**Suggested rewrite** — delete `_FORMAT_HOW`'s conditional sentence entirely and replace
the field list with the three slots the decoder actually samples:

```
Each turn you fill three slots, always, in this order:
  seen   3-8 words: what is on screen now that matters for your task
  hit    is the thing you are watching for TRUE ON SCREEN RIGHT NOW?
  answer the sentence the user receives (the code opens this slot for you)
The system decides when to speak. You only judge what is true now.
```

### 2.2 `NEVER copy the example text` appears in all 9 blocks

`PROMPT_STRATEGY.md` §3 rule 1: *"small models follow do X better than don't do Y, and a
negation still puts the forbidden thing in context."* Occurrences: `:75`, `:132`, `:193`,
`:314`, `:383`, `:460-461`, `:535`, `:609`, `:682`. Rewrite as
*"Describe the video you are watching now."*

### 2.3 Every block is 2–3× over the token budget

`PROMPT_STRATEGY.md` §7: *"Prompt length ≤ 500 tokens of semantics per task."*
Measured (chars/3.7):

| task | total ~tok | role / format / semantics |
|---|---:|---|
| `explicit_target_grounding` | 764 | (monolith) |
| `instant_event_alert` | 1049 | (monolith) |
| `semantic_condition_alert` | 1121 | (monolith) |
| `snapshot_counting` | **1371** | 141 / 488 / **741** |
| `realtime_state_monitor` | **1595** | 112 / 489 / **993** |
| `cumulative_counting` | **1602** | 109 / 488 / **1004** |
| `event_narration` | **1648** | 107 / 446 / **1094** |
| `sequential_step_instruction` | **1660** | 98 / 446 / **1115** |
| `dedup_counting` | **1663** | 131 / 488 / **1042** |

The six new blocks are **~1.7× larger than the blocks that already regressed** under the
project's own "more prompt = worse" finding, and their *semantics halves alone* are 1.5–2.2×
the entire budget. The 446–489-token `_FORMAT_BLOCK_*` is already fenced
(`prompts.py:213-274`) and can be deleted mechanically today; that alone brings every block
under ~1200 and is `ROADMAP` item 1.2.

### 2.4 The worked examples demonstrate a JSON shape the decoder cannot produce

Every new block's example looks like:

> `prompts.py:321` `{"seen":"baker mixing dough in a bowl","have_enough_info":false,"count":0,"fps":1.0,"question_for_next":"Has the tray come out of the oven yet?"}`

The schema walk force-feeds `,"more":` immediately after `have_enough_info` — `count`, `fps`
and `question_for_next` cannot appear there. Per `PROMPT_STRATEGY.md` §1 ("examples teach
WHAT TO NOTICE, not HOW TO TYPE") the *format* mismatch is harmless, but the *semantic*
lesson these examples carry — "carry the count forward every tick" — is unlearnable, and
they cost ~35% of each block. Once the format section is stripped, the examples should be
rewritten to the reachable shape (`seen` / `hit` / `answer` only).

### 2.5 The model is never told it cannot hear — and 6/90 evaluated questions are audio

The nine ICL blocks are, on inspection, free of the words audio/sound/speech/hear. But the
**system prompt** — the highest-authority text, `SYSTEM_PROMPT.format(instruction=…)` at
`input_ingester.py:31`, fed the raw OmniPro question — is not. From the 90 samples in the
saved runs:

| task | audio_dependency | question |
|---|---|---|
| `instant_event_alert` | **none** | *"Let me know if you **hear** a referee blow a whistle."* |
| `instant_event_alert` | **none** | *"Let me know when the match ends with a victory **announcement**."* |
| `instant_event_alert` | **none** | *"Let me know when the reviewer starts the **sound test** for the earbuds."* |
| `snapshot_counting` | **none** | *"When the **narrator mentions** access via the stairs, count how many people are on the staircase."* |
| `instant_event_alert` | helpful | *"Let me know if any sudden **noises** in the room distract the speaker."* |
| `semantic_condition_alert` | helpful | *"Let me know whenever the **narrator identifies** a specific reason…"* |

So the model *is* being handed audio instructions, at `audio_dependency=none`, and no prompt
anywhere reconciles that with a vision-only pipeline. This is not a prompt-text bug so much
as a missing prompt: **the invariant is stated in comments, never to the model.**

**Suggested addition, one line, at the top of every block:**

> `You receive VIDEO FRAMES ONLY — you cannot hear this video. When the instruction names something audible, judge it from its VISIBLE signs (the referee's whistle at his lips and his raised arm; a title card where a narrator would speak).`

Two genuine audio leaks inside the blocks themselves:

- `prompts.py:84` — `{"seen":"crowd roaring in the stadium","have_enough_info":false}`.
  "Roaring" is an audio percept, demonstrated in a `seen` slot. Change to
  `"crowd on their feet, arms raised"`.
- `prompts.py:79`, `:207` — the SCA narrative *"the crowd roaring"* and the IEA
  `{"seen":"applause stopped, new speaker approaching"}`. "Applause stopped" is not
  visible. Change to `"hands lowered, new speaker walking up"`.
- Softer: the whole IEA worked example is *"Let me know when the audience starts
  clapping"* (`:194`), an intrinsically audio-named event. It is *rendered* visually
  ("the camera cuts to the audience clapping") so it is defensible, but if a clean
  example is cheap, prefer one with no audio reading at all (e.g. "let me know when the
  scoreboard changes").

---

## 3. Per-task findings

### 3.1 `instant_event_alert` — NEEDS-FIX (the level is right; trim it)

`have_enough_info` means: **"the event is visibly happening on screen right now."**
Well-posed, answerable from the current frames, and correctly a LEVEL:

> `prompts.py:177-178` `"true the moment the event is happening on screen NOW; stays true while it continues; back to false once it is over"`

The gate paragraph is exactly what MISSION §5 asks for:

> `prompts.py:186-190` `"YOU DO NOT DECIDE WHEN TO ALERT — the system does… Your only job each tick: judge honestly whether the event is happening NOW"`

**Scorer:** `instant_event_alert` ∈ `TIME_ONLY` (`metrics.py:25`) — content is **never
checked** (`_content_correct` returns `True` immediately). The audit brief's "others =
Gemini judge" is wrong for this task. So the "ONE natural sentence UNDER 20 words" rule at
`:181-182` is unscored. Keep it only because the answer's non-emptiness is a fire
precondition (`controller.py:460`) and because `_word_sim` dedup reads it.

**Fixes:** delete the conditional field rules (§2.1), the negation (§2.3), the audio-flavoured
example line `:207`, add the vision-only line (§2.5). Target ~450 tokens.

### 3.2 `semantic_condition_alert` — NEEDS-FIX (audio leak + a scorer ambiguity)

`have_enough_info` means: **"the user's condition is satisfied by what is on screen right
now."** Well-posed and a LEVEL (`:58-59`). This is the block that produced F1 0.255; do not
restructure it, just clean it.

**Bug 1 — audio in a `seen` example**, `prompts.py:84`, quoted in §2.5.

**Bug 2 — the scorer disagrees with itself.** `metrics.py:32` puts SCA in `JUDGE_TASKS`
(Gemini, score ≥3). `dataset.py:39` says `"semantic_condition_alert": "time_only"`.
`TASK_CONTENT_KIND` is dead code (grepped: defined, never imported), so `metrics.py` wins
and the prompt's "state WHAT happened AND WHY, under 25 words" (`:62-63`) is correct — but
one of the two tables is a lie and should be deleted before it misleads someone.

**Bug 3 — the 25-word cap vs the decoder.** `schema_max_answer_tokens=32`
(`config.py:106`) ≈ 24 words. The instruction and the hard cap are within a word of each
other; SCA answers measured 12.2 words mean, 13 max, so it is not biting today, but the two
numbers should be made consistent deliberately rather than by luck.

### 3.3 `explicit_target_grounding` — NEEDS-FIX (the scorer, not the prompt)

`have_enough_info` means: **"the trigger is happening now AND the target is visible."**
The conjunction is a small hazard — if the target is briefly occluded during a sustained
trigger the level drops and re-arms the rising edge, allowing a spurious second fire — but
it is a defensible level.

**The real bug is in `metrics.py`, and it makes 2 of 9 grid cells unwinnable no matter what
the prompt says.** `_extract_position` (`metrics.py:69-74`) scans `_POSITIONS`
(`metrics.py:34-35`) **in list order** and returns the first substring hit:

```python
_POSITIONS = ["top-left", "top-center", "top-right", "center-left", "center",
              "center-right", "bottom-left", "bottom-center", "bottom-right"]
```

`"center"` sits at index 4, before `"center-right"` (5) and `"bottom-center"` (7), and
`"center"` is a substring of both. So a perfect answer *"…his red cap is in the
center-right of the frame"* extracts as `"center"` and is scored **wrong**. Measured on the
saved ETG ground truth: positions are `center` ×5, **`center-right` ×1, `bottom-center` ×1**
— **2 of 7 GT triggers cannot be scored correct.**

**Fix (scorer):** sort by length descending before matching, or match on word boundaries:

```python
for p in sorted(_POSITIONS, key=len, reverse=True):
```

The prompt's *"it MUST contain exactly one cell name"* (`:122-123`) is correct and should
stay; it just isn't sufficient today.

### 3.4 `snapshot_counting` — NEEDS-FIX (level is right, the second conjunct dilutes it)

`have_enough_info` means, per `prompts.py:300-302`:

> `"true while the trigger condition is visibly true on screen AND you can see the things to be counted; false before it and after it is gone."`

This is the only *new* block that keeps a LEVEL, and it is nearly well-posed. Two problems:

1. **The second conjunct is almost always true** ("you can see the things to be counted"),
   so it adds no discriminative signal and softens the trigger judgment. AUC 0.511 ≈ chance
   is consistent with a level dominated by a near-constant conjunct. **Rewrite:**
   `"true while the trigger moment named in your task is visibly happening on screen; false before it and after it has passed."` Counting is what the answer is for; it does not
   belong in the level.
2. **The trigger is frequently audio** (*"When the narrator mentions access via the
   stairs…"*, `audio_dependency=none`). With no audio and no vision-only instruction
   (§2.5), the model has nothing to lock onto. This is the likelier root cause of the 0.511
   than the prompt wording.

**Answer format: correct and verified.** `metrics.py:64-66` takes `re.findall(r"-?\d+")[0]`;
the prompt mandates a digit-initial answer (`:304-308`) and warns off leading years/prices.
Measured 5/5 emitted answers contained an integer.

**Dead text to delete:** *"Before the trigger, keep count at 0 and keep looking"* (`:302`)
and the whole `count` field description — unreachable (§0.1), and `count` is not what the
scorer reads anyway (it reads the *answer string*).

### 3.5 `cumulative_counting` — BROKEN (the level was turned into an edge)

`have_enough_info` means, per `prompts.py:368-371`:

> `"true ONLY on the tick where a NEW occurrence is visibly happening. As soon as that occurrence has finished, go back to false — if you stay true, the system cannot fire for the NEXT occurrence. The rhythm is false -> true (report) -> false -> true (report) -> ..."`

Three invariant violations in one paragraph:

1. **It is an EDGE, not a LEVEL.** MISSION §5 / `ICL_DIFF_CONTROLLER.md` §7: *"the model
   only ever reports a LEVEL"*, and code does `rising = state.hit and not prev_hit`
   (`controller.py:457`). The prompt asks the model to differentiate the signal, and the
   code then differentiates it **again**. A correctly-obeyed edge signal produces one
   `True` tick, which the rising-edge detector converts to one fire — that part survives —
   but any tick-to-tick jitter now produces spurious edges, and the *continuous* `p_hit`
   that the whole trigger thesis rests on becomes the confidence of a derivative rather
   than a state. That is a plausible mechanism for AUC ≈ 0.5.
2. **It tells the model how the firing machinery works** (*"if you stay true, the system
   cannot fire"*), i.e. it hands the model the timing decision MISSION §5 removed from it.
3. **It contradicts the same block, 100 lines earlier.** `_FORMAT_FIELDS_CORE_B`
   (`prompts.py:233-235`), included verbatim in this block, says:
   > `"have_enough_info : a LEVEL, not an edge — true while the thing you are asked to report is true ON SCREEN NOW"`

   and `_FORMAT_GATE` (`:256-261`) says *"you will NEVER double-alert by keeping it true."*
   The block instructs the model to do the opposite of itself, twice.

**Also unreachable:** the entire "HOW YOU REMEMBER THE COUNT … Restate the total in count
each tick, and keep a reminder in question_for_next" (`:363-367`) — both slots are behind
the dead `more` hatch (§0.1).

**Suggested rewrite of the level:**

> `What have_enough_info means here: true WHILE an instance of the action you are counting is visibly under way on screen right now; false between instances. Do not try to decide whether it is a new one — report the level honestly and the system will alert once per instance.`

Then keep the derivation of the *number* where it already works — from `reported`, which is
in the prompt: *"the number of lines in WHAT YOU HAVE ALREADY TOLD THE USER, plus one."*

### 3.6 `dedup_counting` — BROKEN (edge + an inventory split across a live and a dead channel)

`have_enough_info` means, per `prompts.py:445-448`:

> `"true ONLY on the tick where a thing you have NEVER counted is clearly featured on screen. Once that first appearance is over, go back to false so the next NEW thing can be reported."`

The researcher's hypothesis is **confirmed and can be sharpened**: this asks *"is a target I
have not yet counted visible now?"*, which is (a) an edge, (b) a novelty predicate, and (c)
only half-answerable. The half that works is `reported` — it **is** in the prompt
(`controller.py:366-369`) and the block correctly points at it (`:437-440` *"That list IS
your inventory"*). The half that does not is `count`, which the block also demands
(*"Restate it in count and mirror it into question_for_next"*, `:443-444`) and which is
unreachable and invisible. So the prompt splits one job across one working channel and two
dead ones. Measured `p_hit` mean 0.440 with the widest spread of any task (0.085–0.990) and
AUC 0.483 — the model is producing a confident signal about *something*, just not something
the gate can use.

**The fix follows the project's own most successful pattern (level → edge in code):**

- **Prompt (level):** `true WHILE a single subject of the kind you are counting is prominently featured on screen right now (one clear subject filling the shot, not a wide establishing shot). Do not judge whether you have seen it before.`
- **Code (dedup):** `controller.py:458-459` already has `_word_sim`, but `distinct` is only
  evaluated *inside* a true stretch (`level and prev_level`). Extend it to the whole
  `reported` list as a **fire precondition** for this task:
  `fire = answer and (rising or distinct) and max(_word_sim(answer, a) for _, a in reported) < thr`.
  That is the identical mechanism `EXPERIENCE.md` records as fixing `new_event` (which
  re-fired one alert 7× when left to the model).

**Answer format is the one thing that is working:** 25/25 emitted answers contained an
integer, digit-initial, e.g. *"2 different things so far — children playing with stones in a
dimly lit street."* `metrics.py:233` compares that first integer to `gt_item["count"]`.
Keep `:449-453` verbatim.

### 3.7 `realtime_state_monitor` — BROKEN (compares against an invisible variable; and points at the wrong vocabulary)

`have_enough_info` means, per `prompts.py:518-521`:

> `"true ONLY on the tick where you SEE the state become different from phase. Once the new state has settled in, go back to false and just carry the new phase forward"`

**Unanswerable as written.** `phase` is never emitted (dead hatch) and never rendered
(§0.2), so "different from `phase`" compares against nothing. It is also an edge. The block
does offer a working fallback in the previous sentence — *"the state you named most
recently in [the reported list] is the state you are currently in"* (`:514-516`) — which
**is** answerable, so this task is the closest of the four to being rescuable by wording
alone. Its AUC 0.619 (the best of the four) is consistent with that.

**The right fix is architectural and reuses the mechanism the project already trusts:**
make `phase` a **first-class forced slot** in the schema walk, decoded every tick like
`seen`, and let **code** fire on `phase != prev_phase`. That is precisely "the model reports
a state, code detects the edge", costs ~4 decodes/tick, and removes the ill-posed question
entirely. The same slot then serves `event_narration` and `sequential_step_instruction`
(§3.8, §3.9) — one mechanism, three tasks.

**Second, independent bug — the answer points at the wrong vocabulary.** The scorer requires
the GT `state_to` as a case-insensitive substring (`metrics.py:77-82`, `:238-239`). The
prompt says:

> `prompts.py:524-527` `"Name the new state in the PLAINEST, most literal words — the words the video itself would use"`

But the GT state names are drawn from **the task instruction's own enumeration**, not from
the video. Measured example:

| | |
|---|---|
| question | *"Monitor the screen display and update me whenever it switches between **graphics**, **text cards**, or **range footage**."* |
| GT `state_to` | `"text title card"`, `"shooting range footage"`, `"intro graphic"` |

**Suggested rewrite:**

> `Name the new state using the words YOUR TASK INSTRUCTION uses for it. If the instruction lists the possible states, copy one of those phrases verbatim into your sentence; add the video's own noun for it as well if they differ ("switched to the text title card — the range footage has ended").`

Note also the stale rationale at `prompts.py:501-502` ("name the old one too for a second
chance at the substring") — `_extract_state` only ever matches `state_to`; naming the old
state is harmless but buys nothing. Fix the comment or drop the claim.

### 3.8 `event_narration` — BROKEN (there is no level here; the question is genuinely ill-posed)

`have_enough_info` means, per `prompts.py:593-595`:

> `"true ONLY on the tick where a new stage has visibly begun. Once you have described it, drop back to false and watch the stage play out"`

**Verdict on the researcher's hypothesis: correct, and stronger than stated.** For an alert
task the level is a property of the *frame*. For narration there is no property of the
frame that answers "should I speak now?" — the honest answer is a property of the *pair*
(what is on screen, what I last said), and only the second half is in the prompt. The
block's own hedge gives it away: *"if your next sentence would say roughly what one of those
lines already says, the answer is no"* (`:591-592`) — that is a text-similarity computation
being asked of a logit read. AUC 0.550 is what you get when a near-constant "yes, something
is happening" is asked to carry a change signal.

**What should be asked instead — two honest options:**

1. **Reuse the phase mechanism (preferred if the task is kept).** Decode a short
   `stage` label every tick (*"crew unloading van"* → *"panels going up the ladder"* →
   *"rails being bolted"*), fire in code on label change, and generate the long narration
   sentence only on the fire. Narration then becomes a state monitor with a free-text
   state, and `have_enough_info` disappears from the prompt entirely.
2. **Drop it.** `MISSION.md` §9.7 already says *"drop `event_narration` (n=5)"* and §3 says
   it *"cannot carry a claim."* Spending the fix budget here contradicts the roadmap.

**Independent bug — the length instruction exceeds the decoder cap.** `prompts.py:599-600`
asks for *"ONE or TWO sentences, 20-40 words"*; `schema_max_answer_tokens=32`
(`config.py:106`) ≈ 24 words. Measured narration answers: mean 21.2, **max 26** words — the
cap is already clipping the top of the requested range. Either raise the cap for judge
tasks or lower the instruction to 15-25 words.

### 3.9 `sequential_step_instruction` — BROKEN (never run, edge-framed, and the answer format is invented)

`have_enough_info` means, per `prompts.py:664-668`:

> `"true ONLY on the tick where a new step visibly BEGINS — give the instruction as the step starts… Then drop back to false while the step is carried out, or the system cannot fire on the NEXT step."`

Same edge violation as §3.5–3.8; same fix (a `step_label` slot + code-side change
detection). No AUC exists for this task — it has never been run (0 samples in every saved
`online_pred.jsonl` with a controller tick log).

**Independent bug — the mandated prefix is not the ground-truth style.** `prompts.py:673-674`:

> `"Start with 'Step N — ' using the number you worked out from the reported list"`

Measured ground truth for this task:

> `"Start by holding up a rectangular piece of white paper. Flip it over to demonstrate…"`
> `"Now, fold the paper in half by bringing the top edge down to the bottom…"`

Imperative — **but no `Step N` prefix.** Scoring is a Gemini judge against that string
(`metrics.py:32`, `:240-241`). A prefix carrying a *wrong* index (and the index is derived
from a `reported` list that is empty whenever an earlier step was missed) is a gratuitous
factual error placed at the front of the answer, where a judge weighs it most. The
imperative-mood instruction at `:670-672` is correct and valuable and should stay; the
`Step N` prefix should go — **unless** the researcher chose it deliberately from the
example in `config.py:248`, which *does* show `"Step 1 — Boil water in a kettle."` See §5.

**Length:** 15-35 words requested vs a ~24-word cap — same conflict as §3.8.

---

## 4. Ranked fixes (highest impact first)

1. **Make `phase`/`count`/`question_for_next`/`note` reachable, or delete every sentence
   about them.** Zero of 3,049 ticks opened the `more` hatch (max `p_more` 0.438 vs a 0.5
   threshold). Right now ~1/3 of four blocks' semantics is unlearnable text, and pillars 7c
   and 4b are silently dead. Cheapest experiment: lower `more_threshold` to ~0.35 and
   re-measure. Correct fix: promote `phase` (and a per-task `count`) to **forced slots in
   the schema spine**, and render them back into the prompt (`controller.py:342-373`).
   *Nothing else on this list can be evaluated cleanly until this is resolved.*

2. **Turn the five EDGE definitions back into LEVELS.** `prompts.py:368-371`, `:445-448`,
   `:518-521`, `:593-595`, `:664-668`. Each currently instructs the model to do the edge
   detection that `controller.py:457` already does, and each directly contradicts
   `_FORMAT_FIELDS_CORE_B` ("a LEVEL, not an edge") inside the same block. This is invariant
   #2 and it is violated in exactly the four tasks whose AUC was measured at chance.
   *One variable per experiment: change only this, and re-measure AUC on the same shards.*

3. **Fix `_extract_position` in `metrics.py:69-74`.** Two of nine grid cells
   (`center-right`, `bottom-center`) are **unreturnable**, and both occur in the saved ETG
   ground truth — 2 of 7 triggers are unwinnable regardless of the model. One-line fix
   (`sorted(_POSITIONS, key=len, reverse=True)`), and it invalidates every ETG number
   recorded so far.

4. **Point `realtime_state_monitor`'s answer at the instruction's vocabulary, not the
   video's** (`prompts.py:524-527`). The scorer needs the GT `state_to` verbatim as a
   substring, and GT state names are paraphrases of the question's own enumeration. This is
   a pure wording change on the task with the best measured AUC (0.619) and 98 vision-only
   samples — the best effort/return ratio on the list.

5. **Strip the format section and the conditional rules from all 9 blocks.**
   `_FORMAT_BLOCK_*` (`prompts.py:213-274`) is already fenced for mechanical deletion:
   -446/-489 tokens per block, removes the 0/15-compliance `fps` rule and the
   `answer REQUIRED whenever` rule the decoder already enforces, and moves every block
   toward the ≤500-token budget. `ROADMAP` 1.2, unblocked, zero accuracy risk by the
   project's own "more prompt = worse" finding.

Runners-up: add the one-line "you cannot hear this video" statement to every block (§2.5);
fix the two audio percepts at `prompts.py:84` and `:207`; reconcile the answer-length
instructions with `schema_max_answer_tokens=32`; delete the dead `TASK_CONTENT_KIND` table
in `dataset.py:37-48` that contradicts `metrics.py`.

---

## 5. Where I need the researcher to state intent

1. **`sequential_step_instruction` — is the `"Step N — "` prefix deliberate?** The real
   benchmark GT does not use it (*"Start by holding up a rectangular piece of white
   paper…"*), but the example you pasted into `config.py:248` does
   (*"Step 1 — Boil water in a kettle."*). Which is authoritative? If the benchmark, the
   prefix should go; it front-loads a possibly-wrong integer into a judge-scored answer.

2. **`event_narration` — keep or drop?** `MISSION.md` §9.7 says drop it (n=5, "cannot carry
   a claim"), yet it has a 1,648-token block and was one of the four tasks you measured
   AUC on. If it stays, it needs the `stage`-label restructure (§3.8), which is real work.
   If it goes, items 1/2 above get cheaper.

3. **`semantic_condition_alert` content scoring — judge or time-only?** `metrics.py:32`
   judges it; `dataset.py:39` says `time_only`. The prompt is written for the judge. One of
   the three is wrong and it changes what SCA's F1 has ever meant.

4. **Should `dedup_counting`'s identity check move into code?** §3.6 proposes
   `_word_sim(answer, every reported answer) < thr` as a fire precondition — the same
   level→edge-in-code move that fixed `new_event`. It is the right shape architecturally,
   but it makes the *flagship* task's dedup a code heuristic rather than a model capability,
   which weakens the "the frozen model already knows" story. Your call which framing the
   paper wants.

5. **`explicit_target_grounding`'s conjunctive level** (*"trigger is happening AND target is
   visible"*, `prompts.py:118-119`). A brief occlusion drops the level and re-arms the edge,
   permitting a second fire on a single trigger. Intended, or should the level be the
   trigger alone with target visibility handled in the answer?

6. **Answer-length policy.** `schema_max_answer_tokens=32` is a single global cap, but the
   tasks want 20 words (IEA/ETG), 25 (SCA), 20-40 (narration), 15-35 (step instruction).
   Should the cap become per-task, or should the instructions be lowered to fit 32 tokens?
