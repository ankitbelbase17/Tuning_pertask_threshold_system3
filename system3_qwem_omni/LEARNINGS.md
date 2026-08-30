# LEARNINGS — why this is slow, and what actually broke

**Written 2026-07-30**, after the first end-to-end measurement of the schema-walk
controller. Companion to `AUC_DIAGNOSIS.md`, `PROMPT_AUDIT.md`, `EVAL_REPORT.md`.

---

## 1. The answer to "why is this taking so long"

**Not because the ideas are wrong. Because every failure so far has been SILENT.**

Twelve independent defects, found in one day. **None of them raised an error.** Every
one produced plausible-looking JSON, a finished run, and a number you could put in a table.

| # | defect | how it presented | how long it hid |
|---|---|---|---|
| 1 | **Frozen perception** — `seen` byte-identical for whole videos | sensible-looking descriptions | since the ICL controller existed |
| 2 | Self-chaining eval never resubmitted (`wait` made the tail unreachable) | job "completed" | 1 generation |
| 3 | SLURM `--export=ALL,TASKS=a,b,c` splits on commas → 1 task, `OUT_NAME` dropped | ran, wrote to wrong dir | until the log was read |
| 4 | `dump_scores.py` labels ticks with the **shard dir** (`g00`), not the task | AUC computed on pooled tasks | every AUC before today |
| 5 | `video_id` isn't unique across tasks → 29% of ticks scored against another task's GT | plausible AUC | every AUC before today |
| 6 | `_POSITIONS` order: `"center"` shadows `center-right`/`bottom-center` | 2 of 9 grid cells unwinnable | every ETG number ever |
| 7 | `more` escape hatch opens 0.5% of ticks → `count`/`phase`/`note`/`question_for_next` unreachable | fields simply never appeared | since the schema decoder |
| 8 | `writer_greedy=True` short-circuited Qwen's own sampling config | deterministic, "clean" output | since the beginning |
| 9 | `task_hit_thresholds` fitted on 3 videos against a chance-level signal | looked tuned | since fitted |
| 10 | 70% of GT triggers involve speech; we ingest no audio | low scores read as model failure | since task selection |
| 11 | `run_fast.sh` pointed at a dataset dir holding 34 of 932 videos | "11 SCA videos" was just all there was | months |
| 12 | git object store + conda stdlib corrupted by the scratch filesystem | commands failed cryptically | recurring |

**The lesson: this project's bottleneck is not ideas, it is observability.** A wrong
idea that fails loudly costs an hour. A wrong idea that fails silently costs a week,
and worse, it *contaminates every measurement taken while it hides*.

---

## 2. What "frozen perception" was, concretely

`seen` is the system's only report of what it is looking at.

- **20 of 58 videos emitted exactly ONE byte-identical `seen` for the entire video.**
- 31 of 58 emitted ≤2 distinct descriptions ever.
- `realtime_state_monitor::4QLU8CNu6GQ`: six GT screen-switches at 7/12/180/184/304/308s.
  The model emitted identical JSON at **all 470 ticks**, including `"event_time_s":0.0`
  while the clock read 308s. It never left the intro card.
- Failure-mode split over 87 GT triggers: **PERCEPTION 81.6%**, JUDGMENT 3.4%, TIMING 6.9%.
- **Decisive**: a null model using only the timestamp — zero pixel access — beat `p_hit`
  on 3 of 4 tasks (dedup .698 vs .506, rtsm .636 vs .623, snapshot .772 vs .536).
  `p_hit` correlated with video time at |r| up to 0.93.

**The system was scoring worse than a clock.** Everything downstream — the gate, the
thresholds, the 9-task ICL suite — was operating on a constant. That is why AUC sat at
0.5 no matter what we changed.

### This explains the "everything fires at 0–1s" observation
Frozen `seen` → near-constant `p_hit` → at t=0 `prev_level=False`, so the first tick over
threshold fires **immediately**, then `level` latches TRUE and the rising-edge gate can
never fire again. Measured: 19.5% of all fires in the first 2s (0.7% of a 300s timeline),
6/43 videos latched TRUE and never re-armed, only **1.42 rising edges per video** against
**3.5 GT events per video** — a hard recall ceiling of 0.44 before the model is consulted.

### Two causes, both now flagged off
1. **The memory trace fed back into the prompt** put the model's own last `seen` a few
   tokens before the slot it had to refill. Greedy decode copies it; "repeats collapsed"
   then shortens the trace and makes copying *more* attractive. Self-reinforcing.
   → `seen_trace_in_prompt=False`
2. **Greedy decoding.** Between ticks the context grows by one frame — ~185 tokens of
   ~100k. That barely moves the logits, so argmax returns a byte-identical string *by
   construction*. Greedy did not merely permit the repetition, it guaranteed it.
   → `writer_greedy=False`; Qwen's own config (T=0.7, top_p=0.8, top_k=20) now active.
   Reproducibility is unaffected: the generator is seeded and the walk is lockstep.

### The structural cause, NOT yet fixed
```
[~100k vision tokens] … [newest frame] → [1,400 tokens of ICL instructions] → {"seen":"
```
The newest frame sits **1,400 tokens** from the point of generation, behind a wall of
instruction prose. Nothing in the architecture privileges the present; the model has to
*retrieve* "what is happening now" out of a haystack. **Fix: move the constant ICL into
the pinned sink and leave only a short cue after the newest frame**, so the last thing the
model sees before answering is the video, not the manual. (Filed in ROADMAP 1.3-D as a
latency experiment — it is not one. It is the perception fix.)

---

## 3. What is working — do not touch these

| component | evidence |
|---|---|
| **Schema-walk decoder** | 100% valid JSON (was 85.8%), 100% field compliance, `fps` spam 63.4% → 0.02%, example-copying 7% → 0%, ~3× faster |
| **Logit read of the boolean** | agrees with a free decode **647/647**; gives a continuous confidence a generated token cannot |
| **No frame loss** | `frames_emitted == frames_ingested` on every shard (56/56, 139/139, 109/109, 31/31) |
| **Snapshot clone** | 8.6 ms mean — not a bottleneck, despite suspicion |
| **Model load** | ~9 s — not a bottleneck, despite suspicion |
| **Lockstep determinism** | bit-reproducible runs |
| **The measurement rig** | `compliance.py` + `auc.py` + `report.py` found all twelve defects above. The old 33-event F1 could not have. |

**Building the rig before chasing accuracy was the correct call and it paid.**

---

## 4. What is not working

| problem | measured |
|---|---|
| Perception (fixes flagged on, unverified) | 20/58 videos one description |
| **The rising-edge gate** | 1.42 firings/video vs 3.5 GT events → recall ceiling 0.44 |
| Memory / accumulators | behind an escape hatch that opens 0.5% of ticks |
| 5 task prompts define the level as an **edge** | contradicts the level instruction in the same block; code differentiates again |
| Prompt length | 1371–1663 tokens against a ≤500 budget |
| Task selection | 70% of GT triggers involve speech; `event_narration` GT are periodic checkpoints a perfect model scores ~0.5 on |

---

## 5. Throughput — the real arithmetic

Measured per tick: `snapshot_clone` 8.6 ms · `ingest.frame` 103 ms · **`ctrl_decode_s` 890 ms**.
Decode is ~99% of tick cost; nothing is contending with anything.

At one tick per video-second the pipeline runs at **~1–2× real time**. With
`max_seconds=600`, one sample is 10–20 minutes of wall clock — which is exactly the
observed 20 samples per 90-minute generation. Nothing is broken here; it is arithmetic.

**Lever:** `seen_mode="off"` removes the `seen` decode entirely, leaving only the logit
read — ~0.15 s/tick, a **~6× eval speedup**. It is also the Priority-1 experiment already
in the roadmap. Do not enable it until perception is verified working, or the two effects
confound.

---

## 6. The fix that matters most: make failures LOUD

Every defect above was silent. The single highest-leverage change is not a model change —
it is **assertions that abort a run rather than let it produce a plausible number**:

1. **Frozen-perception assert** — if `distinct(seen)/ticks` for a video is below a floor,
   fail the run. This alone would have caught defect #1 on day one.
2. **Degenerate-signal assert** — if `p_hit` is near-constant, or has fewer than N distinct
   values, fail. (It has only **78 distinct values across 9,921 ticks**.)
3. **Config echo + assert** — print the resolved task list, output dir, decode mode and
   sampling params at run start and assert they match what was requested. Catches #3.
4. **Frame-conservation assert** — `frames_emitted == frames_ingested`. Already measured;
   make it fatal.
5. **Always report the timestamp-only null model beside every AUC.** If the system cannot
   beat a clock, no other number means anything. This is the one that turns a silent
   catastrophe into an obvious one.
6. **Join on `(task, video_id)`, never `video_id`.** Fixes #4 and #5.

---

## 6b. 2026-07-31 — the measurement itself was wrong

Five more silent defects, all in the **scorer**, found in one day. Same pattern as
section 1: every one produced a finished run and a number you could put in a table.

| # | defect | how it presented | how long it hid |
|---|---|---|---|
| 13 | LLM judge fell back to **word overlap** on any exception | log said `google-genai judge active` | every judged run |
| 14 | `judge_cache.json` clobbered, not merged, by 4 concurrent writers | 24h of judging → 42 entries | since parallel eval |
| 15 | `semantic_condition_alert` LLM-judged; upstream scores it `time_only` | our SCA joint-F1 strictly too low | since task setup |
| 16 | `_extract_count` didn't strip clock times → `"At 01:23, count is 5"` = **1** | correct answers marked wrong | since counting existed |
| 17 | state matched by **substring**, not equality → one sentence satisfied two different GT states | RSM content accuracy inflated | since counting existed |
| 18 | `--dry-run` and `--rescore-only` **called the paid API** (report() → score_sample → judge) | invisible; only harmless because no judge worked | since judge_offline.py |

**How #13 was caught, and the general technique:** re-score the run in a bare shell
with **no API key at all**. It reproduced the shipped `content_acc` to four decimals.
A real judge and no judge cannot agree exactly. *Run the pipeline with a dependency
deliberately removed; if the number does not move, that dependency was never used.*

### The rule this produced

**A metric we could not measure is withheld, never estimated.** `ContentJudge.score()`
returns `1.0` / `0.0` / `None`, and `None` propagates: `content_acc` and `joint_*` come
back as `None` with `content_complete=false` and explicitly-named `*_lb` lower bounds.
There is no fallback judge and there must never be one again. A blank cell in a table is
a fact; a fabricated cell is a lie that survives into the paper.

**Corollary: a verdict you cannot inspect is not evidence.** The cache maps an
irreversible hash to a bare 0/1 — it could not tell you what was judged or why. Every
judgement now appends the full triple, the raw 1–5 score, and the judge's own explanation
to `judge_trace.jsonl` (`judge_audit.py` reads it back). Auditing is append-only and never
read by the scorer, so it cannot affect a number — it can only let you check one.

### Prose where the benchmark wants a token

Aligning to upstream's scorer showed two tasks are **structurally unwinnable**, no matter
how good perception gets:

| task | matched | unparsed | wrong | acc | diagnosis |
|---|---|---|---|---|---|
| `realtime_state_monitor` | 296 | 0 | **296** | **0.000** | emits sentences; upstream compares whole payload to `state_to` by equality |
| `explicit_target_grounding` | 6 | 0 | 4 | 0.333 | says "center" descriptively instead of the 9-cell label |
| counting (×3) | 841 | **0** | 494 | 0.29–0.47 | format fine — the counts are genuinely wrong |

`unparsed = 0` on counting is the column that matters: it separates *"we wrote it wrong"*
from *"we saw it wrong"*. Two different problems that the old lenient scorer blended into
one mediocre number.

**Lesson: implement the benchmark's scorer before optimising against your own.** Six weeks
of prompt tuning was measured against rules that did not match the ones our results will
be judged by. Read the reference implementation, not just the paper — `TASK_CONTENT_KIND`
came straight from upstream's source and is now the single dict every task set derives
from, so they cannot drift apart again.

### And the biggest one: the diff never diffed

**79% of all 15,280 emits are byte-identical repeats** of an earlier emit. When the same
text repeats, the self-reported `event_time_s` spans a **median of 14 s** (max 561 s), and
**66% of repeat-groups are wider than the entire ±3 s window** — unmatchable by
construction. Precision 0.112 is a *content* failure wearing a *gating* failure's costume,
which is why 15 gate strategies all lost to a fixed threshold: the gate cannot tell a
restatement from a recurrence. See `CONTROLLER_DIAGNOSIS.md` §3b.

---

## 7. Standing rules earned the hard way

- **A diff is a decoder constraint, not a prompt instruction.** Every conditional emission
  rule ever written was obeyed ~2%; every unconditional slot, 100%.
- **The model cannot decide *whether* to emit a field.** It fills slots you hand it. The
  `more` hatch violated this rule the same day the rule was written down.
- **Do not put anything load-bearing behind a model's discretion.** Memory, counters, and
  state must be code-owned or unconditional.
- **One variable per experiment.** Still true, still violated under time pressure.
- **Read the logs, not just the metric.** The AUC said "chance". The logs said "it is
  describing the intro card at t=308s". Only one of those was actionable.
- **Implement the benchmark's scorer before optimising against your own.** Read the
  reference *source*, not just the paper. Six weeks of tuning was measured against rules
  our results will not actually be judged by.
- **Withhold, never estimate.** A metric that could not be measured must come back `None`,
  never a plausible substitute. Any fallback that produces a number on failure will
  eventually be reported as if it were real.
- **To test whether a dependency is used, remove it.** Re-running with no API key
  reproduced the "judged" scores exactly — which is what proved the judge was never
  running.
- **A verdict you cannot inspect is not evidence.** Persist the judge's raw score and
  reasoning next to the text it judged, or you cannot defend a single content number.
