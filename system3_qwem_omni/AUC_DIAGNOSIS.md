# AUC_DIAGNOSIS — why the per-tick confidence signal is at chance

Forensic read of `output_all/g0{0..3}` (run `20260730_142302`, 24 samples, 9,921 ticks)
against `/iopsstor/scratch/cscs/dbartaula/omnipro_data/benchmark.json`.
Read-only; no pipeline code or git state was touched.

---

## VERDICT

**The AUC numbers are real. The cause is not the prompt, not the gate, not the label
window, and not the logit read. It is that the controller's perception channel is
frozen: `seen` locks onto whatever is on screen in the first 1–5 seconds of the video
and never updates again for the rest of the run.**

The model is being asked "is the monitored condition satisfied RIGHT NOW?" while its
own description of *now* is a still image from t≈1s. In **16 of 24 samples the model
emitted ≤ 2 distinct scene descriptions for the entire video**; in 9 of those it emitted
**exactly one string, byte-identical, for 139–477 consecutive seconds**. Under those
conditions `p_hit` cannot carry event information, and it does not.

Three consequences, in order of importance:

1. **81.6% of ground-truth triggers are failure mode (a) PERCEPTION** — `seen` does not
   describe the event at all. Mode (b) JUDGMENT is **3.4%**. So the answer to your crux
   question is unambiguous: **(a), not (b)**. The prompt is not asking the wrong
   question. There is nothing at the input for the question to be asked *of*.
2. **This is not "the 8B cannot see it."** The same video under a different task prompt
   *does* track the scene (evidence §3). Frames are arriving. The model is not re-reading
   them.
3. **The residual above-chance signal in `p_hit` is a clock, not a percept.** A null
   model that only knows the video timestamp — zero access to pixels — scores
   AUC 0.698 on dedup_counting and 0.636 on realtime_state_monitor, **beating `p_hit`
   itself** (0.506 / 0.623). `p_hit` carries no measurable perceptual information above a
   monotone drift with video time.

Two things in your framing are wrong, and one of them matters:

- **The join is buggy (as you flagged) — but it is not the cause.** `video_id` is not a
  unique key: 5 of 18 videos carry 2–3 tasks each, so **2,873 / 9,921 ticks (29.0%)**
  were scored against a *different task's* ground truth. Fixing it moves every number by
  < 0.03. Fix it anyway, but it is not the story.
- **Two of the four tasks should never have been in this metric.** `event_narration`
  ground truth is a set of *periodic narration checkpoints*, not event onsets;
  `snapshot_counting` ground truth is a single **audio-cued** instant in a vision-only
  pipeline. Neither is a fair target for a ±3s per-tick AUC. Details in §5.

---

## 1. Sanity-check of your join and clock (your item 5)

| check | result |
|---|---|
| ±3s positive rate | **5.8%** pooled (you said ~5%) — ✅ correct |
| `vt` vs GT `t_sec` clock | ✅ same clock. Ticks are exactly 1.0 s apart at `fps=1.0`, first tick `vt=1.0`, and per-sample `vt_max` matches the video duration implied by the GT span. |
| tick count | 9,921 in `scores.jsonl`, exactly the `ctrl.gate` count in `run_20260730_142302.log`. Your 9,396 is that set minus the videos whose GT lookup failed. |
| `scores.jsonl` freshness | ⚠️ it contains **only** the 14:23 run. `online_pred.jsonl` was appended to by the 18:49 run. The two files are not from the same set of samples. |

### The join bug, quantified

`auc.py`'s own docstring warns about this:

> `# CRITICAL: one video appears under SEVERAL tasks with DIFFERENT ground truth ... Merging them inflates the positive rate and makes every number meaningless.`

…and then `parse_scores()` keys the tick dict on `m.group(2)` — the video id — anyway.
`dump_scores.py` compounds it by writing the *shard dir name* into the `task` field.

Videos carrying more than one task in this run:

```
CR55TVLjTzc  ->  realtime_state_monitor + dedup_counting + event_narration   (3 × 594 ticks)
TJMrW-vgNz0  ->  realtime_state_monitor + dedup_counting                     (2 × 392)
JJPTYTswPCI  ->  dedup_counting + event_narration                            (2 × 473)
0dJSPsXCujc  ->  snapshot_counting + dedup_counting                          (2 × 420)
Uaa-Mz84vC8  ->  realtime_state_monitor + event_narration                    (400 + 314)
```

**Effect of fixing it** (same ticks, correct per-sample GT via the
`[online] ===== [i/N] task::video::id gt_times=[...]` markers in the raw log):

| task | your AUC | corrected AUC | corrected AP | ticks | videos |
|---|---|---|---|---|---|
| event_narration | 0.550 | **0.564** | 0.079 | 2,323 | 5 |
| dedup_counting | 0.483 | **0.506** | 0.061 | 4,075 | 10 |
| realtime_state_monitor | 0.619 | **0.623** | 0.098 | 2,519 | 7 |
| snapshot_counting | 0.511 | **0.536** | 0.014 | 1,004 | 2 |
| POOLED | — | **0.557** | — | 9,921 | 18 |

Note your per-task *video counts* were also wrong (you had 6 dedup / 2 rtsm; the truth is
10 / 7) — a symptom of the same last-write-wins dict. **Conclusion: the join is broken,
the verdict survives it.**

---

## 2. The primary finding: frozen perception

### Frozen-perception census (all 24 samples, `seen` normalised: lowercase, leading article stripped)

| task | video | span | distinct scenes | longest identical run |
|---|---|---|---|---|
| event_narration | LqZBXB5HP9A | 465 s | **1** | **465 ticks** |
| dedup_counting | y9XYO9d9H94 | 590 s | **1** | **590 ticks** |
| realtime_state_monitor | 4QLU8CNu6GQ | 470 s | **1** | **470 ticks** |
| realtime_state_monitor | 5_LRMz5WxUk | 139 s | **1** | **139 ticks** |
| dedup_counting | JJPTYTswPCI | 473 s | **1** | **473 ticks** |
| snapshot_counting | 0dJSPsXCujc | 420 s | **1** | **420 ticks** |
| dedup_counting | I1ECITk7WTY | 207 s | **1** | **207 ticks** |
| event_narration | KE1RZcZvWMw | 478 s | **1** | **477 ticks** |
| realtime_state_monitor | noOM42oLy_s | 404 s | **1** | **404 ticks** |
| dedup_counting | 2C1PWbzKIn0 | 345 s | **1** | **345 ticks** |
| dedup_counting | CR55TVLjTzc | 594 s | 2 | 488 |
| realtime_state_monitor | TJMrW-vgNz0 | 392 s | 2 | 309 |
| dedup_counting | 3HIvIoG-P1s | 583 s | 2 | 504 |
| event_narration | JJPTYTswPCI | 473 s | 2 | 268 |
| realtime_state_monitor | Uaa-Mz84vC8 | 400 s | 2 | 362 |
| realtime_state_monitor | v2Tlscx1Rzc | 120 s | 2 | 116 |
| dedup_counting | Wtnw9ooAW98 | 413 s | 4 | 405 |
| dedup_counting | TJMrW-vgNz0 | 392 s | 3 | 343 |
| dedup_counting | 0dJSPsXCujc | 420 s | 3 | 189 |
| dedup_counting | le0B8XH-W1I | 59 s | 4 | 31 |
| realtime_state_monitor | CR55TVLjTzc | 594 s | 3 | 227 |
| snapshot_counting | UZQZ2fAoTn4 | 584 s | 5 | 202 |
| event_narration | Uaa-Mz84vC8 | 399 s | 5 | 188 |
| event_narration | CR55TVLjTzc | 594 s | 5 | 248 |

**16 / 24 samples describe the whole video with ≤ 2 distinct scenes.** Where a
"change" occurs it is frequently a paraphrase of the *same* opening description — e.g.
y9XYO9d9H94 oscillates between `'a desert landscape with mountains and clouds'` and
`'desert landscape with mountains and clouds'` (an indefinite article) across 590 seconds.

### The single most damning excerpt

`realtime_state_monitor :: 4QLU8CNu6GQ` — "Monitor the screen display and update me
whenever it switches between graphics, text cards, or range footage." Six ground-truth
switches at 7, 12, 180, 184, 304, 308 s. The model emits a **byte-identical JSON object
at all 470 ticks**:

```
[  30.1s | vid    3.0s] ctrl.raw  [4QLU8CNu6GQ] {"seen":"graphic of skull with guns and flames","have_enough_info":true,"event_time_s":0.0,"answer":"The screen switched from text to a graphic."}
[ ...    | vid  180.0s] ctrl.raw  [4QLU8CNu6GQ] {"seen":"graphic of skull with guns and flames","have_enough_info":true,"event_time_s":0.0,"answer":"The screen switched from text to a graphic."}
[ ...    | vid  184.0s] ctrl.raw  [4QLU8CNu6GQ] {"seen":"graphic of skull with guns and flames","have_enough_info":true,"event_time_s":0.0,"answer":"The screen switched from text to a graphic."}
[ ...    | vid  304.0s] ctrl.raw  [4QLU8CNu6GQ] {"seen":"graphic of skull with guns and flames","have_enough_info":true,"event_time_s":0.0,"answer":"The screen switched from text to a graphic."}
[ ...    | vid  308.0s] ctrl.raw  [4QLU8CNu6GQ] {"seen":"graphic of skull with guns and flames","have_enough_info":true,"event_time_s":0.0,"answer":"The screen switched from text to a graphic."}
```

`event_time_s: 0.0` at video-time 308 s. It is still reporting the intro card. `p_hit`
over that stretch wanders 0.378 → 0.622 with no relationship to the six transitions.

The corresponding `ctrl.gate` lines show the gate correctly refusing to fire on this —
the gate is doing its job on garbage input:

```
[ ... | vid  180.0s] ctrl.gate [4QLU8CNu6GQ] fps=1.0 level=False rise=False new_occ=False fire=False next=1.0s gen=1.2s ntok=7 q='' p_hit=0.593 p_more=... agree=True argmax='false' count=0 notes=0
```

### `event_time_s` is independently stale

`realtime_state_monitor :: CR55TVLjTzc`, quoting the model's own timestamps:

```
vid 363.0s  {"seen":"inside a car at night", ... "event_time_s":254.0, ...}   # 109 s stale
vid 483.0s  {"seen":"black screen with animated logo", ... "event_time_s":24.0, ...}   # 459 s stale
vid 594.0s  {"seen":"black screen with animated logo", ... "event_time_s":586.0, ...}
```

The writer output in `online_pred.jsonl` for this sample is the same string eight times —
`"The setting switched from black screen to a black screen with animated logo."` at
t = 24, 517, 537, 543, 546, 549, 558, 586 — i.e. at 586 s into a 10-minute abandoned-property
exploration the system still believes it is looking at the intro logo.

---

## 3. It is NOT a frame-delivery bug, and NOT a model-capability floor

Decisive control: **the same video, the same frames, three different task prompts**
(`CR55TVLjTzc`, run 14:23, samples 1–3).

```
--- realtime_state_monitor :: CR55TVLjTzc ---
vid   1.0s p_hit=0.047  {"seen":"black screen with logo animation", ...}
vid  35.0s p_hit=0.562  {"seen":"inside a car at night", ...}        <-- tracks the scene
vid 254.0s p_hit=0.321  {"seen":"inside a car at night", ...}
vid 580.0s p_hit=0.095  {"seen":"black screen with animated logo", ...}   <-- reverts

--- dedup_counting :: CR55TVLjTzc ---
vid   1.0s p_hit=0.294  {"seen":"black screen with a glowing blue and green logo", ...}
vid 209.0s p_hit=0.469  {"seen":"black screen with glowing blue and green logo", ...}
vid 483.0s p_hit=0.531  {"seen":"black screen with glowing blue and green logo", ...}
vid 594.0s p_hit=0.469  {"seen":"black screen with a glowing blue and green logo", ...}   <-- frozen 594 s

--- event_narration :: CR55TVLjTzc ---
vid   1.0s p_hit=0.953  {"seen":"a glowing blue and green logo appears", ...}
vid 483.0s p_hit=0.148  {"seen":"a glowing blue and green circular logo with the letters 'L", ...}
vid 580.0s p_hit=0.294  {"seen":"two men in a car at night", ...}     <-- finally tracks, at 580 s
```

Identical pixels reach the cache in all three runs. Under one prompt the model reports
`"inside a car at night"` at t=35; under another it reports the intro logo at t=594.
**Therefore: the frames are arriving; the model is not re-reading them.** That rules out an
ingester/vision-stream fault and rules out "an 8B cannot resolve these scenes."

The two mechanisms consistent with this evidence, both in `async_omni_v2/controller.py`:

- **Self-anchoring via the memory feedback.** The controller prompt injects
  `"WHAT YOU HAVE SEEN so far (your own observations, newest last; repeats collapsed)"`.
  With duplicates collapsed, that block is a *single line* — `@1s <first description>` —
  for the entire run. `schema_max_seen_tokens = 12` makes it a short, highly copyable
  phrase. Copying it is the lowest-loss continuation, and the copy re-enters the trace,
  closing the loop. The lock-in time (t = 1–5 s, i.e. the moment the trace acquires its
  first entry) matches this exactly.
- **No recency signal in the cache.** The controller reads `mgr.snapshot_clone()` — a KV
  cache holding *every* frame from t=0 (≈196 vision tokens/frame; 594 frames ≈ 116k
  tokens, under the 262,144 budget, so **no eviction occurred** and this is not a
  cache-overflow bug). The text prompt is spliced on the end with no marker for which
  frames are current. An attention-sink-biased read returns the opening frames — which is
  precisely what every frozen sample reports.

Both are cheap to discriminate; see §7.

---

## 4. Failure-mode breakdown (your item 1)

Classification over all 87 in-range ground-truth triggers. `seen` in a ±3 s window is
compared against the GT `response` + `event_description` + `state_to` by content-word
overlap; `p_hit` is compared against the per-video median.

| mode | definition | count | share |
|---|---|---|---|
| **(a) PERCEPTION** | `seen` unchanged across [t−10, t+6] and overlap < 0.15 | **71** | **81.6%** |
| (b) JUDGMENT-FAIL | `seen` matched the event but `p_hit` ≤ per-video median | 3 | 3.4% |
| (b+) JUDGMENT-OK | `seen` matched and `p_hit` above median | 7 | 8.0% |
| (c) TIMING | `seen` changed within ±15 s but not within ±3 s | 6 | 6.9% |

Per task:

| task | n | (a) | (b) | (c) |
|---|---|---|---|---|
| dedup_counting | 35 | 29 | 3 | 3 |
| event_narration | 25 | 24 | 0 | 1 |
| realtime_state_monitor | 25 | 17 | 6 | 2 |
| snapshot_counting | 2 | 1 | 1 | 0 |

**Caveat that strengthens the verdict:** 3 of the 7 "(b+) JUDGMENT-OK" cases are
spurious — they are GT triggers at **t = 1 s**, where the frozen opening description
trivially overlaps the GT text (`0dJSPsXCujc` t=1, `2C1PWbzKIn0` t=1, `4QLU8CNu6GQ` t=7).
Genuine judgment failures are essentially nil.

### Representative mode-(a) rows, verbatim

```
dedup_counting  JJPTYTswPCI t=  43  seen='Georgia Tech logo displayed on screen'  | GT='First scorer — Jonathan Dwyer (#21) crosses the goal line for a touchdown'
dedup_counting  JJPTYTswPCI t= 268  seen='Georgia Tech logo displayed on screen'  | GT='Fourth scorer — Scott Blair (#4) kicks the extra point through the uprights'
dedup_counting  I1ECITk7WTY t=  92  seen='black screen with white text'           | GT='Second interviewee — Tom Saunders, co-founder of SPRYE, shares his experience'
dedup_counting  y9XYO9d9H94 t=  83  seen='desert landscape with mountains and clouds' | GT='First founder — Ibrahim Shamsi is introduced in a purple shirt and begins speaking'
dedup_counting  Wtnw9ooAW98 t= 183  seen="a person's back in a dark room"         | GT='Fifth person — Evie, the bartender, starts her interview segment'
rtsm            TJMrW-vgNz0 t= 137  seen="title screen with 'BEST CAR' text"      | GT='Now on a racetrack, leaving the coastal road. Coastal road to racetrack'
rtsm            Uaa-Mz84vC8 t= 155  seen='woman speaking to camera'               | GT='Now using a styling tool instead of a cutting tool. Cutting to styling'
event_narration KE1RZcZvWMw t= 265  seen='man in green shirt standing in hallway' | GT='A wheel swap is underway on the grey M5. The old wheels were removed'
event_narration LqZBXB5HP9A t= 450  seen='drummer playing with visible sticks'    | GT='The drumming has stopped, and he is now wiping his face with a white towel'
```

The one clean success in the whole set — `snapshot_counting :: UZQZ2fAoTn4`, GT at 387 s,
`seen` changes to `'staircase with people moving up and down'` at 390 s — is drowned by a
`p_hit` ramp that peaks 150 s later (see §6).

---

## 5. The label definition (your item 3) — and where your framing is unfair to the tasks

### Label-sensitivity sweep

AUC of `p_hit` under seven labelling schemes, correct per-sample attribution:

| task | ±1 s | ±3 s | ±5 s | ±10 s | onset only | [t, t+5] | persist to next |
|---|---|---|---|---|---|---|---|
| dedup_counting | 0.535 | 0.506 | 0.509 | 0.520 | 0.548 | 0.503 | 0.565 |
| event_narration | 0.572 | 0.564 | 0.562 | 0.575 | 0.564 | 0.566 | 0.602 |
| realtime_state_monitor | 0.626 | 0.623 | 0.625 | 0.608 | 0.635 | 0.617 | 0.537 |
| snapshot_counting | 0.541 | 0.536 | 0.555 | 0.559 | 0.504 | 0.581 | **0.917** ⚠ |
| **POOLED** | 0.573 | 0.557 | 0.559 | 0.558 | 0.574 | 0.560 | 0.705 ⚠ |

Positives per scheme (pooled, of 9,921): ±1 s → 240, ±3 s → 572, ±5 s → 890,
±10 s → 1,625, onset → 86, [t,t+5] → 493, persist → 7,854.

**Finding: AUC is insensitive to the label window.** Every point-labelling scheme lands
within 0.03 of ±3 s. **Your ±3 s choice is not hiding a signal.** This is itself
diagnostic — a model with real but mistimed perception would show a strong peak at some
tolerance; this one is flat, which is what you get when the score is unrelated to the
label at any offset.

⚠ **Do not read the `persist` column as good news.** `snapshot_counting` jumps to 0.917
only because that task has *one* GT event at t=386 in a 584 s video, so "persist to next
trigger" labels the entire second half positive — and `p_hit` on that video is a monotone
ramp (r(vt) = +0.930). It is measuring the ramp, not the event. Same mechanism inflates the
pooled figure to 0.705.

### Two tasks should not be in this metric at all

**`event_narration` GT are periodic narration checkpoints, not onsets.** From
`benchmark.json`, `LqZBXB5HP9A`: triggers at 75, 165, 255, 390, 450 s with
`event_description` values `"Initial performance phase"`, `"Mid-performance dynamics"`,
`"Technical drum fill"`, `"Late-set rhythmic complexity"`, `"Post-drumming interaction"`.
The other four event_narration samples are equally regular (CR55TVLjTzc: 160/280/430/580;
Uaa-Mz84vC8: 60/135/210/290/375). There is no perceptual difference between t=160 and
t=157. **A perfect model would score near 0.5 here.** ±3 s AUC on event_narration is not a
measurement of anything.

**`snapshot_counting` GT is a single audio-cued instant.** `UZQZ2fAoTn4`: *"When the
narrator **mentions** access via the stairs, count how many people are on the staircase"*
— `event_description: "Narrator says 'access via stairs' at 06:25"`. `0dJSPsXCujc`:
`"Woman speaks to Atlas at 06:25"`. With n=1 event per video the AUC has effectively one
independent positive; it is noise regardless.

### Audio dependence across the whole evaluated set

The pipeline ingests **no audio** (`dataset.py`: *"system_5 ingests NO audio"*), yet:

| task | GT events | involving speech/sound |
|---|---|---|
| dedup_counting | 40 | 29 (72%) |
| event_narration | 25 | 22 (88%) |
| realtime_state_monitor | 30 | 17 (57%) |
| snapshot_counting | 2 | 0 |
| **total** | **97** | **68 (70%)** |

Trigger-type census: `visual+speech` 51, `visual` 29, `visual+sound` 15, `speech` 2.
**Only 29 of 97 triggers (30%) are purely visual.** `realtime_state_monitor` has the
highest purely-visual fraction (13/30) and is also the highest-AUC task (0.623) — the one
place the ordering is consistent with the system doing real work.

---

## 6. Is `p_hit` degenerate? (your items 2 and 4)

### Yes, on four independent measures.

**(i) Resolution.** Across all 9,921 ticks there are only **78 distinct `p_hit` values**.
Consecutive values sit on a **~0.125-nat logit grid** (median gap 0.127) — the signature
of a bf16 logit readout. Since the log rounds `p_hit` to 3 decimals (≈0.004 logit
resolution near 0.5), the quantisation is *upstream of logging*, i.e. real. Per video
only **11–30 distinct levels** are ever observed over 400–600 ticks.

**(ii) Near-constant, heavily autocorrelated.** Per-sample lag-1 autocorrelation is
**+0.54 to +0.99** (median ≈ +0.89). Per-sample σ is 0.06–0.19 for 20 of 24 samples.
`LqZBXB5HP9A`: 465 ticks, σ = 0.061, range 0.202–0.469, 11 distinct values. It is a slow
drift, not a per-frame response.

**(iii) It tracks the clock, not the content.** Correlation of `p_hit` with video time:

```
 4QLU8CNu6GQ  r(vt) = -0.898      UZQZ2fAoTn4  r(vt) = +0.930
 event_narr JJPTYTswPCI r = -0.866   0dJSPsXCujc r(vt) = +0.788
 dedup 0dJSPsXCujc  r(vt) = -0.755   dedup JJPTYTswPCI r = -0.717
 le0B8XH-W1I  r(vt) = -0.679      LqZBXB5HP9A  r(vt) = +0.625
```
Sixteen of 24 samples have |r(vt)| > 0.3. Correlation with `len(seen)` is comparably large
in several samples (JJPTYTswPCI event_narration: **+0.935**; Uaa-Mz84vC8: −0.801) — i.e.
`p_hit` moves with *how many tokens the model just emitted*, not with what happened.

**(iv) The null-model test — the decisive one.** AUC at ±3 s of scores that cannot see
the video at all:

| task | `p_hit` | clock `vt` | negated clock `−vt` | per-video z-scored `p_hit` | \|Δp_hit\| | `len(seen)` |
|---|---|---|---|---|---|---|
| dedup_counting | 0.506 | 0.302 | **0.698** | 0.508 | 0.524 | 0.475 |
| event_narration | 0.564 | 0.556 | 0.444 | 0.533 | 0.476 | 0.457 |
| realtime_state_monitor | 0.623 | 0.364 | **0.636** | 0.590 | 0.517 | 0.492 |
| snapshot_counting | 0.536 | **0.772** | 0.228 | 0.572 | 0.556 | 0.437 |
| **POOLED** | 0.557 | 0.401 | **0.599** | 0.541 | 0.513 | 0.474 |

**On three of four tasks a bare timestamp beats the model's confidence.** GT triggers
cluster early in these videos (new entities are introduced early), so "earlier = more
likely positive" is a genuinely predictive heuristic — and `p_hit` does not beat it.
Per-video z-scoring (which removes the large per-video offset: mean `p_hit` ranges from
0.009 on `0dJSPsXCujc` to 0.603 on `5_LRMz5WxUk`) *lowers* pooled AUC to 0.541, confirming
that part of the apparent pooled signal was between-video offset rather than within-video
discrimination.

### Ceiling and floor

- **`p_hit` > 0.9: 41 / 9,921 ticks (0.41%).** Of those, **2 fall within ±3 s of a GT
  trigger — 5%, identical to the 5.8% base rate.** The ceiling is pure noise.
  The largest cluster (24 of the 41) is `UZQZ2fAoTn4` at vt = 535–563 s, which is
  **148–176 seconds away** from that video's single GT event at 387 s — the tail of the
  r(vt)=+0.93 ramp. The next cluster is `event_narration :: JJPTYTswPCI` at vt = 1–6 s,
  92–97 s from the nearest GT, while `seen` reads `'Georgia Bulldogs logo displayed on
  screen'` (also the wrong team — the video is Georgia Tech).
- **`p_hit` < 0.05: 638 ticks (6.43%)**, and they are not spread out:
  `snapshot_counting :: 0dJSPsXCujc` contributes **420 — its entire run**. That sample's
  `p_hit` never exceeded **0.018** across 420 ticks (σ = 0.004). It is a dead channel, yet
  it scores AUC 0.757 under ±3 s, which should be read as a warning about trusting any
  single-video AUC here.

### One mechanism that is NOT broken

The logit read itself is sound. `verify_logit_read` shows the unconstrained argmax at the
boolean slot is **always** a boolean (`false` 7,294 / `true` 2,627 — never a third token),
and it agrees with `p_hit ≥ 0.5` on **100%** of ticks. The `agree=False` rate of 30% in the
logs is comparing the argmax against the *task-specific* threshold
(`task_hit_thresholds`: dedup 0.75, rtsm 0.14), not against 0.5. **Do not spend effort
here — the readout is fine, its input is not.**

---

## 7. What to change, ranked

1. **Fix the frozen `seen` before measuring anything else.** Nothing downstream is
   interpretable while 16/24 samples describe a whole video with ≤2 strings. Two cheap
   discriminating experiments, both one-flag:
   - **Ablate the memory feedback** — run with the `WHAT YOU HAVE SEEN so far` block
     empty. If `seen` starts tracking, it is the self-anchoring copy loop, and the fix is
     to stop feeding `seen` back verbatim (feed a *count* of scene changes, or the last
     entry only, or drop it and keep only `reported`).
   - **Ablate the order** — `seen_mode = "after"` already exists in `config.py`. If reading
     the boolean *before* describing breaks the anchor, the copy loop is confirmed and
     `p_hit` gets a clean input for free.
   Add a **recency delimiter** in the prompt regardless (an explicit "frames since your
   last check begin here" marker), since the controller currently gives the model no way to
   tell which of ~116k cached vision tokens are current.
2. **Add a frozen-perception assertion to the run itself.** A one-line guard — "if the
   normalised `seen` has not changed in N consecutive ticks, log a WARNING" — would have
   caught this in the first 60 seconds of the first sample instead of after 9,921 ticks.
   This is the highest-value permanent change in the list.
3. **Restrict the AUC metric to `realtime_state_monitor` and the purely-visual subset.**
   Drop `event_narration` (periodic checkpoints, not onsets) and `snapshot_counting`
   (n=1, audio-cued) from the dense metric entirely. Filter GT to
   `trigger_type == "visual"` while the pipeline has no audio — that is 29 of 97 events,
   and it is the only subset the system could in principle get right.
4. **Always report the negated-clock null alongside AUC.** `−vt` scores 0.599 pooled.
   Any future AUC that does not clear its own clock baseline is not evidence of
   perception. This is a two-line addition to `auc.py` and it would have short-circuited
   this entire investigation.
5. **Fix the join keys.** Make `dump_scores.py` emit the real task and a sample id (both
   are already on the `===== [i/N] task::video::id` marker line it walks past), and key
   `auc.py`'s `parse_scores` on `(task, video_id, sample_id)` rather than `video_id`.
   29.0% of ticks are currently scored against the wrong ground truth. Worth ~0.03 of AUC,
   but it makes every future number defensible.
6. **Widen the `p_hit` dynamic range.** 78 distinct values over 9,921 ticks, on a
   0.125-nat grid, is not enough resolution for threshold tuning to be meaningful. Read the
   two logits in fp32 before the softmax, and log more than 3 decimals.
7. **Exclude truncated samples.** The 14:23 run was killed mid-sample three times:
   `le0B8XH-W1I` stopped at 59 s of 286 s (5 of 8 GT unreachable), `5_LRMz5WxUk` at 139 s
   of 306 s (2 of 4), `v2Tlscx1Rzc` at 120 s of 196 s (3 of 4). They are currently pooled
   in at full weight.
8. **Do not tune the gate.** Only 46 of 9,921 ticks fired. Given §6, any threshold or
   hysteresis change is fitting noise. `config.py` already flags
   `task_hit_thresholds` as *"FITTED ON ONLY 3 VIDEOS PER TASK … provisional"* — leave
   them until the perception input is fixed.

---

### Appendix — artifacts read

- `omniprofast/output_all/g0{0,1,2,3}/run_20260730_142302.log` (primary; 24 samples,
  9,921 `ctrl.gate` + 9,921 `ctrl.raw` lines), plus the `184923` / `191451` continuation
  logs (28 further samples, not in `scores.jsonl`).
- `omniprofast/output_all/g0*/online_pred.jsonl`, `scores.jsonl`.
- `/iopsstor/scratch/cscs/dbartaula/omnipro_data/benchmark.json`.
- `omniprofast/auc.py`, `omniprofast/dump_scores.py`, `omniprofast/dataset.py`,
  `async_omni_v2/controller.py`, `async_omni_v2/config.py`.

Ticks were attributed to samples via the
`[online] ===== [i/N] task::video_id::sample_id gt_times=[...]` markers in the raw logs,
which are unambiguous where `video_id` alone is not.
