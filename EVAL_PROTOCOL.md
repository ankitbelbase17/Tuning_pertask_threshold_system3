# EVAL_PROTOCOL — what our scorer measures, and how it differs from OmniPro (it doesn't, any more)

**Written 2026-07-31.** Companion to `LEARNINGS.md`, `EVAL_REPORT.md`.

This document exists because on 2026-07-31 we discovered that a full 24-hour run had
reported **word-overlap scores under the label "google-genai judge active"**, and that
our scorer disagreed with the benchmark's own scorer on three separate rules. Both are
fixed. This is the record of what the rules actually are.

---

## 1. The rule that was silently broken

`metrics.py` had a lexical-overlap fallback:

```python
try:
    r = self._judge.judge(question, gt, pred)
    ...
except Exception:
    pass
return 1.0 if _lexical_sim(gt, pred) >= 0.3 else 0.0    # <- DELETED
```

The Gemini free tier was returning `503 UNAVAILABLE`. Every call fell through to word
overlap. The init log still printed `[judge] google-genai judge active: model=...`, so
the run looked judged. **The tell:** re-scoring the run in a bare shell with no API key
at all reproduced the shipped `content_acc` to four decimals (0.3125). A real judge and
no judge cannot produce the same number.

**The fix is a tri-state verdict.** `ContentJudge.score()` now returns `1.0`, `0.0`, or
`None`. `None` means UNJUDGED and propagates all the way to the reported metrics, which
then refuse to publish:

```
content_acc      : None          # withheld, NOT zero
joint_f1         : None
content_complete : False
n_judged         : 756
n_unjudged       : 428
content_coverage : 0.6385
content_acc_lb   : 0.2694        # lower bound, named so it cannot be mistaken
joint_f1_lb      : 0.0611
```

Timing metrics never involve the judge and are always exact — verified unchanged at
`time_f1 = 0.1945` before and after.

**Rule going forward: a metric we could not measure is withheld, never estimated.**
There is no fallback judge and there must never be one again. If you are tempted to add
one "just so the table has a number", that is exactly the failure this cost us a day to
find.

### Related fix: the judge cache was being clobbered

Four eval processes shared `judge_cache.json`. Each loaded it at init and wrote its own
dict straight over the top — last writer wins, other three erased. After 24 hours of
judging the cache held **42 entries**. `_cache_put` now re-reads, merges, writes a temp
file and `os.replace`s it, so concurrent writers accumulate and a reader never sees a
half-written file.

---

## 2. Three places our scorer disagreed with the benchmark

Verified against the reference implementation
(`github.com/RuixiangZhao/OmniPro`, `metrics/online/scorer.py` +
`utils/online_parser.py`) and the paper (§3.2.1). All three are now aligned.

| # | what we did | what OmniPro does | effect |
|---|---|---|---|
| 1 | judged `semantic_condition_alert` with the LLM | `time_only` — content not scored at all | our SCA joint-F1 was **strictly lower** than the same system scores under the benchmark |
| 2 | first integer in the emit | strips clock times **first**, then first integer | `"At 01:23, count is 5"` scored as **1**, not 5 — correct answers marked wrong |
| 3 | substring search: is the GT state named anywhere in the text? | whole payload, lowercased, **exact** equality to `state_to` | our `realtime_state_monitor` content accuracy was **inflated** |

The canonical mapping now lives in `metrics.py::TASK_CONTENT_KIND`, copied verbatim from
upstream:

```python
"instant_event_alert":         "time_only",
"semantic_condition_alert":    "time_only",
"explicit_target_grounding":   "position",
"snapshot_counting":           "count",
"cumulative_counting":         "count",
"dedup_counting":              "count",
"realtime_state_monitor":      "state",
"event_narration":             "gpt_judge",
"sequential_step_instruction": "gpt_judge",
```

`TIME_ONLY` / `COUNT_TASKS` / `POSITION_TASKS` / `STATE_TASKS` / `JUDGE_TASKS` are all
**derived** from this one dict, so they cannot drift apart again.

Consequence: only **2** tasks need the LLM judge, not 3. Paper-conformant call count for
the current run is **472**, not 512.

Note the reference scorer independently arrived at the same tri-state design — its
`_score_content` returns `None` for `gpt_judge` when no judge is supplied. Our `None`
means the same thing and is safe to compare.

---

### Consequence discovered by the alignment: our RSM writer emits the wrong format

Aligning `state` to exact-match dropped `realtime_state_monitor` content accuracy from
**0.396 → 0.000**. That is not the scorer being too strict — it exposes a real defect in
our system.

```
GT state_to : 'main bathroom'
our emit    : 'The setting switched from the main bathroom area to the shower area.'
GT state_to : 'shower area'
our emit    : 'The setting switched from the main bathroom area to the shower area.'
```

Two things are wrong here, and the old substring rule hid both:

1. **The benchmark constrains this task's output to a bare state name.** Upstream's parser
   takes the *whole payload* as the state and compares it to `state_to` by exact equality.
   A sentence can never match. Our writer emits prose, so it scores 0 under the protocol
   the paper actually uses.
2. **The old substring rule let ONE emit satisfy TWO different ground-truth states** —
   that sentence contains both `main bathroom` and `shower area`, so it counted correct
   for whichever it was matched against. The 0.396 was inflated by construction.

**The check was run across every constrained-format task** (all matched emits in
`output_full9`). Two are format failures, three are not:

| task | kind | matched | unparsed | wrong | correct | acc | diagnosis |
|---|---|---|---|---|---|---|---|
| `realtime_state_monitor` | state | 296 | 0 | **296** | **0** | **0.000** | **FORMAT** |
| `explicit_target_grounding` | position | 6 | 0 | 4 | 2 | 0.333 | **FORMAT** |
| `dedup_counting` | count | 551 | 0 | 292 | 259 | 0.470 | perception |
| `snapshot_counting` | count | 44 | 0 | 27 | 17 | 0.386 | perception |
| `cumulative_counting` | count | 246 | 0 | 175 | 71 | 0.289 | perception |

`unparsed = 0` everywhere is the key column: the counting tasks always yield a clean
integer, so their errors are genuine miscounts, not formatting. Error direction runs both
ways — exactly −1 is the most common single error (17%), 29% undercount, but
`cumulative_counting` puts 24% at ≥ +3 — so there is no systematic offset to correct.

**Action:** tracked as PRIORITY 0 in `ROADMAP.md`. The `realtime_state_monitor` writer
prompt must emit only the destination state name, and `explicit_target_grounding` must
emit one of the nine region labels (upstream prefers an explicit `Position: <region>`
anchor). Both are prompt-only fixes; counting is a perception problem and ranks behind
them.

### Defect 4 (found 2026-08-01): joint-F1 used the wrong denominator

The paper and the reference scorer agree: a response is valid only if it is **both**
within ±tol **and** content-correct, and then

```
joint_precision = valid / ALL RESPONSES   = tp_content / n_emits
joint_recall    = valid / ALL GT TRIGGERS = tp_content / n_gt
```

Upstream writes these as `tp_content / (tp_time + fp)` and `tp_content / (tp_time + fn)`,
which are identical because `tp_time + fp == n_emits` and `tp_time + fn == n_gt`
(verified on 642/642 of our samples).

We were computing `_prf(tp_content, fp, fn)` = `tp_content / (tp_content + fp)`. That
denominator **quietly drops every emit that matched in time but failed on content** rather
than counting it as a false positive. Effect:

| content accuracy | inflation of our joint-F1 |
|---|---|
| perfect (C == M) | 1.00× |
| 0.47 | 1.02× |
| 0.29 | 1.05× |
| 0.10 | **1.18×** |

**It flattered us most exactly where we were weakest.** Timing metrics were never
affected. Fixed; overall `joint_f1_lb` moved 0.053 → 0.047.

---

## 3. The judge

Paper (§3.2.1): *"we employ Gemini-3-Flash as an LLM judge to score each prediction
against the ground truth on a 1–5 scale; a score ≥3 is considered correct"*, and only for
*"open-ended generation tasks (i.e., Event-Narr. and Step-Inst.)"*.

`GEMINI_MODEL` now defaults to `gemini-3-flash` (was `gemini-3.5-flash` — a different
model than the protocol specifies).

**We do not have Gemini quota.** So in practice Gemini returns `None` and content is
withheld, and the judging is done separately by `judge_offline.py` against OpenAI
`gpt-5-mini` with Structured Outputs.

### Why judging offline at all

The GPU eval and the LLM judge fail for completely unrelated reasons — one needs four
A100s for a day, the other needs API quota. Coupling them is what let a rate-limited
judge silently degrade a 24-hour run. They are now separate processes:

- the eval writes every prediction's **text** to `online_pred.jsonl` and marks content
  UNJUDGED;
- `judge_offline.py` picks that up later, judges only the triples a matched emit needs,
  merges verdicts into the shared cache, and recomputes.

**Verified: nothing is lost by deferring.** Across 491 stored samples there are
**0 missing** `question`, `predictions[].raw`, or `ground_truth[].response`.

### Comparability caveat — read before reporting

Switching judge model changes the numbers. Our results are **not** directly comparable to
published OmniPro baselines judged by Gemini-3-Flash. The mitigation the user has chosen
is to **re-run the baselines under the same GPT judge**, which restores a fair
comparison. Any table mixing GPT-judged ours with Gemini-judged theirs is invalid and
must not be published.

### Cost — measured, not estimated

512 triples, median 261 input + ~254 output tokens per call.

| model | this run (496 samples) | full 932-sample eval | via Batch API (−50%) |
|---|---|---|---|
| gpt-5-nano | $0.011 | $0.020 | $0.010 |
| **gpt-5-mini** | **$0.27** | **$0.51** | **$0.26** |
| gpt-5.1 | $0.27 | $0.51 | $0.25 |

Cost is not a constraint (budget $50 ≈ 100+ full passes). **Do not pick the judge on
price — it is the instrument that measures the paper.** Frugality is already structural:
only emits that *matched* a GT event within ±3 s are ever judged, so 512 of 14,332 emits
(3.6%) cost anything. The over-firing that wrecks precision is free to evaluate.

---

## 4. What is safe to report

| metric | status |
|---|---|
| `time_precision` / `time_recall` / `time_f1` | **exact, always.** No judge involved. |
| `content_acc` / `joint_*` on the 7 non-judge tasks | **exact.** Deterministic extraction, no API. |
| `content_acc` / `joint_*` on Event-Narr + Step-Inst | only after `judge_offline.py` runs. Until then `None`. |
| anything from `output_all/` | **discard.** Contaminated by the `--export` comma bug — SCA only, 2–11 samples/shard. |
| thresholds fitted and reported on the same samples | **diagnostic only.** Use `--dev 0.5` and report held-out. |

---

## 5. Files

| file | role |
|---|---|
| `omniprofast/metrics.py` | scorer; `TASK_CONTENT_KIND` is the single source of truth |
| `omniprofast/judge_offline.py` | deferred judging; OpenAI + Batch backend |
| `omniprofast/resweep.py` | offline threshold sweep on emit **times** (no content) |
| `omniprofast/refit.py` | offline re-threshold + **full** rescore incl. content |
| `omniprofast/judge_cache.json` | shared verdict cache, merge-on-write |
