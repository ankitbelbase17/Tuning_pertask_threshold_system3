# CONTROLLER_DIAGNOSIS — the diff doesn't diff, and the clock drifts

**Written 2026-07-31**, from the 507-sample `output_full9` run (commit `d945291`,
schema decode, all 9 tasks). Companion to `LEARNINGS.md`, `AUC_DIAGNOSIS.md`,
`EVAL_PROTOCOL.md`.

**One-line summary: precision 0.112 is not a threshold problem. 79% of every emit we
produce is a byte-identical repeat of an earlier one, and when the model repeats itself
it reports a different timestamp each time.**

---

## 1. What the controller actually emits

The schema walk works. A real tick, straight from the log:

```json
{"seen":"excavator dumping earth into truck","have_enough_info":true,
 "event_time_s":17.0,"answer":"1 dump so far — the excavator dumps a load of earth into a truck."}
```

Valid JSON, all four fields, semantically correct, count right. Valid-JSON rate is 100%
and the logit read agrees with argmax 647/647. **The decoder is not the problem.**

### Cost breakdown per tick (measured, 1,843 ticks)

| | |
|---|---|
| wall per 1 s of video | **2.56 s** (median 2.60, p90 4.20) |
| controller `gen` | 2.42 s — **94% of the tick** |
| quiet tick | `gen` 0.9–2.0 s, `ntok` 4–12 |
| firing tick | `gen` 2.7–2.8 s, `ntok` 25 |

**The emit decision itself is nearly free.** `p_hit` is a *logit read* at the
`"have_enough_info":` slot — zero decode steps, one forward pass. The 2.4 s is spent
generating **text** (`seen` before the flag, `answer` after). Decision and payload are
serialised into one tick, so each decision waits behind the previous tick's prose.

That matters for the always-on controller (ROADMAP priority 1): the <150 ms trigger
already exists as a logit read. What has to move off the critical path is the prose.

---

## 2. Latency is NOT what breaks the ±3 s tolerance

The scored timestamp is not the tick time — the model self-reports `event_time_s` and
the system uses it:

```
vid 24.0s → 📢 @17.0s
vid 29.0s → 📢 @19.0s
vid 34.0s → 📢 @24.0s
```

So the model back-dates, and 2.4 s of compute latency does **not** automatically cost
temporal accuracy. Good.

**But look at those three lines: same event, byte-identical answer text, reported time
drifting 17 → 19 → 24.** One event, three emits, three different timestamps. At most one
can match; the other two are guaranteed false positives.

### This is systemic, not anecdotal

Grouping emits by identical answer text within a sample (same text ⇒ the model believes
it is the same event), across all 507 samples:

| | |
|---|---|
| total emits | 15,280 |
| distinct answer texts | 3,225 |
| **repeated emits of an identical text** | **12,055 — 79% of all emits** |

And when the same text repeats, the self-reported `event_time_s` spans:

| median | mean | p90 | max |
|---|---|---|---|
| **14.0 s** | 49.9 s | 156 s | 561 s |

**1,371 of 2,085 repeat-groups (66%) span more than 6 s** — wider than the entire ±3 s
matching window, so no threshold can rescue them.

Per task (median span of a repeat group):

| task | median span | n |
|---|---|---|
| snapshot_counting | 140.0 s | 57 |
| instant_event_alert | 48.0 s | 57 |
| explicit_target_grounding | 41.4 s | 12 |
| semantic_condition_alert | 35.1 s | 44 |
| realtime_state_monitor | 25.0 s | 273 |
| dedup_counting | 17.0 s | 759 |
| event_narration | 12.0 s | 315 |
| cumulative_counting | 10.0 s | 278 |
| sequential_step_instruction | 5.0 s | 290 |

`snapshot_counting` is the extreme case and matches its metrics exactly: tP 0.030,
1,041 emits for 33 ground-truth events.

**This is the direct cause of precision 0.112 and the 6.4× over-fire rate.** It is a
*content* failure — the model cannot hold a stable belief about *when* something
happened — wearing the costume of a *gating* failure.

---

## 3. The diff does not diff

Consecutive ticks:

```
[vid 24.0s] ctrl.raw {"seen":"excavator dumping earth into truck","have_enough_info":true,"event_time_s":17.0,"answer":"1 dump so far — ..."}
[vid 25.0s] ctrl.raw {"seen":"excavator dumping earth into truck","have_enough_info":true,"event_time_s":17.0,"answer":"1 dump so far — ..."}
```

Byte-identical. The model restates its entire state every tick; the only thing stopping a
double emit is the rising-edge gate (`rise=False → fire=False`).

**So deduplication is done by the gate, not by the model emitting a delta.** The
config-json-as-diff design — the core idea in `MISSION.md` — is not actually being
exercised. What runs is "full state every tick, deduplicated downstream".

### And the structured fields are dead

Every tick in the entire run logs `count=0 notes=0`. The count exists only in prose
("1 dump so far"), so `_extract_count` has to regex it back out of the answer text. We
round-trip structure → prose → structure for no reason, and the prose path is what the
benchmark then scores.

This is defect #7 from `LEARNINGS.md` still unresolved: no ICL block ever demonstrates
`count`/`phase`/`note`, so the model never emits them.

---

## 3b. TOP PRIORITY FOR NEXT ITERATION — how to actually make the diff diff

**The trap to avoid first.** The obvious fix — "tell the model to emit only what
changed" — is precisely the pattern this project has already proved does not work. From
`LEARNINGS.md`: *a frozen 8B reliably fills a slot you hand it, but will not reliably
decide whether to fill one.* Conditional-emission compliance was measured at **2.1%**
over 2,821 ticks. Any instruction of the form "only include X if Y" will be ignored, and
we will have spent another week discovering that again.

So every fix below moves the decision **out of the model's prose and into either
(a) deterministic code, or (b) a forced binary slot read off the logits** — the one
mechanism that has been shown to work here (`p_hit` agreed with argmax 647/647).

### Fix 1 — Mechanical repeat suppression (do this first; ~20 lines, no model change)

The controller already holds the previous tick's decoded state. Compare and drop:

```python
# in _schema_tick, after decoding `answer`
key = _norm(answer)                      # lowercase, collapse whitespace
if key == state.get("last_emit_key"):
    fire = False                         # identical restatement -> not an event
else:
    state["last_emit_key"] = key
```

This is safe against the obvious objection: a *genuine* second occurrence changes the
text, because the answer carries the running count (`"1 dump so far"` → `"2 dumps so
far"`). Identical text is, by construction, a restatement.

Expected effect: directly targets the 79% of emits that are byte-identical repeats.
This is the "code owns the skeleton, model fills blanks" principle applied to dedup.

### Fix 2 — Freeze the timestamp once an event is committed (kills the drift)

The drift exists because the model **re-estimates** `event_time_s` from scratch every
tick. It should not be allowed to. Keep an emitted-events table and reuse the committed
time verbatim:

```python
committed = state.setdefault("committed", {})   # norm(answer) -> t_sec
if key in committed:
    event_time = committed[key]      # re-report the ORIGINAL time, never re-estimate
else:
    committed[key] = event_time      # first commit wins
```

Expected effect: removes the 14 s median / 561 s max span. Combined with Fix 1 most
repeats never reach the writer at all, and any that do carry a stable timestamp.

### Fix 3 — Replace the judgment with a forced binary slot (the architectural fix)

Fixes 1–2 are string matching, which is blunt: a paraphrase of the same event
(`"the excavator dumps earth"` vs `"a load of earth is dumped"`) slips through. The
durable version asks the model a question it *can* answer, in the format it is reliable
in — a single forced token whose logits we read:

```python
# memory of what the user has already been told is ALREADY rendered into the prompt
logits = step(b.embed_text(',"is_new_event":'))
p_new  = _read_bool(logits, true_id, false_id)     # 0 decode steps, ~free
```

`is_new_event` is a *filled slot*, not a conditional emission — the same shape as
`have_enough_info`, which works. The code, not the prose, then acts on it. This also
gives a second continuous signal to threshold offline, and `dump_scores.py` can persist
it exactly like `p_hit`.

**Validate it the cheap way before trusting it:** we already have 12,055 labelled
repeats and 3,225 distinct events on disk. Replay them and measure AUC of `p_new`
against "is this text new within this sample". If AUC is at chance, the slot is not
carrying signal and Fixes 1–2 remain the answer — that is a one-evening experiment with
no GPU-hours at risk.

### Fix 4 — Make `count` / `note` real, so dedup has structure to work on

While `count` is stuck at 0 the only dedup handle is free text. An ICL block that
demonstrates `count` incrementing turns dedup into an integer comparison, which is
exact. This is defect #7 and it blocks the clean version of Fixes 1–3.

### Ordering and expected payoff

| # | fix | effort | touches | expected |
|---|---|---|---|---|
| 1 | mechanical repeat suppression | ~20 lines | `controller.py` | attacks 79% of emits |
| 2 | freeze committed timestamps | ~10 lines | `controller.py` | removes 14 s median drift |
| 3 | `is_new_event` forced slot | ~1 day + replay validation | `controller.py`, `prompts.py` | paraphrase-robust; new dev signal |
| 4 | populate `count`/`note` | prompt work | `prompts.py` | exact dedup key |

**Measure Fixes 1–2 before building Fix 3.** They are nearly free and, if the 79% figure
is right, should move precision more than any threshold tuning has. Report the effect on
held-out time-F1 via `refit.py`, not on the fitted set.

**What NOT to do:** do not attack this with more gate tuning. `gates.py` already screened
15 firing strategies and every adaptive gate lost to a fixed threshold; `refit.py` finds
that five of nine tasks want `thr=0.0` and win purely on a refractory period. The gate is
being asked to undo repetition it cannot distinguish from recurrence. Fix it at the
source.

---

## 4. What this changes about priorities

Ranked by (metric impact ÷ effort), from this evidence:

1. **Stabilise `event_time_s`.** 66% of repeat groups are unmatchable purely because the
   timestamp moves. Costs no compute. Candidate: once an event is emitted, its time
   should be *frozen in memory* and re-reported verbatim rather than re-estimated — the
   memory block already carries what was told to the user.
2. **Make the diff a real diff.** If `seen` is unchanged from last tick, there is nothing
   to say. 79% of emits are the model restating itself. Suppressing exact repeats at the
   source is strictly better than catching them in the gate, because the gate cannot tell
   a genuine second occurrence from a restatement — but identical text plus a *drifting*
   timestamp is exactly the signature of a restatement.
3. **Populate `count` / `note`.** Needs an ICL block that demonstrates them.
4. **Then** the 2.4 s tick. It only becomes the binding constraint once the controller is
   always-on; today it costs throughput, not accuracy.

The offline threshold refit (`refit.py`) lifts held-out time-F1 from 0.185 → 0.302 for
zero compute, and that is worth banking — but note that five of nine tasks pick
`thr=0.0` and win purely on a **refractory period**. The gate is being asked to undo the
repetition, which is the wrong layer.

---

## 5. Reproducing these numbers

```bash
cd omniprofast
# repeat-rate and event_time_s drift (section 2)
python - <<'EOF'
import json, glob, collections, statistics
drift=[]; dupes=0; tot=0
for p in glob.glob('output_full9/*/online_pred.jsonl'):
    for line in open(p, errors='replace'):
        if not line.strip(): continue
        r=json.loads(line); by=collections.defaultdict(list)
        for e in r.get('predictions',[]):
            tot+=1; by[(e.get('raw') or '').strip()].append(float(e['t_sec']))
        for txt,ts in by.items():
            if txt and len(ts)>1:
                dupes+=len(ts)-1; drift.append(max(ts)-min(ts))
print(f"{dupes}/{tot} repeats ({100*dupes/tot:.0f}%), median span {statistics.median(drift):.1f}s")
EOF

# per-tick cost (section 1)
grep -oE "gen=[0-9.]+s" output_full9/g00/run_*.log | sort | uniq -c | sort -rn | head
```
