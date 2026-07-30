# PROMPT_STRATEGY — why instructions aren't landing, and how to fix it measurably

**Problem statement (the user's):** *"the prompt engineering has been the hardest part of
this research. I have ideas but the prompts keep degrading somehow. My intention is not
being understood properly."*

This document turns that from a feeling into a measurement.

---

## 1. The diagnosis

Collect every prompt-following observation we have. A pattern falls out immediately.

| Instruction | Kind | Result |
|---|---|---|
| "ALWAYS start with `seen`, then `have_enough_info`" | **unconditional slot** | ✅ works — `seen` was the single biggest accuracy lever (F1 0.0 → 0.255) |
| "emit `fps`/`next_check_s` ONLY when they change" | **conditional emission** | ❌ **0/15 ticks complied** |
| "`new_event`: true ONLY if this occurrence is NEW" | **conditional emission** | ❌ re-fired one identical alert **7×** |
| "emit `{}` when nothing changed" | **conditional emission** | ❌ model went passive — 0 emits in ~300 ticks |
| "`answer` REQUIRED whenever `have_enough_info` is true" | **conditional emission** | ⚠️ V1 went silent; only worked once `seen` forced grounding first |

> ### The rule
> **A frozen 8B will reliably fill a slot you hand it. It will not reliably decide
> *whether* to fill a slot.**
>
> Every unconditional slot we have ever added worked. **Every conditional emission rule we
> have ever written has failed.** Not sometimes — every one.

**Corollary — never write a prompt rule of the form "emit X only if Y."** Either always
emit X and let *code* decide what to do with it, or don't offer the slot at all.

This single principle explains all five rows above and predicts the fix for each.

### Second diagnosis: the ICL has almost no authority over format

Every worked example in `config.py` is written compact:
```
{"seen":"fans buying green body paint","have_enough_info":false,...}
```
The model emits, with spaces:
```
{"seen": "dark screen with glowing blue particles", "have_enough_info": false, ...}
```

It is **neither obeying the rule nor copying the example.** It reverted to generic
pretrained JSON style. But the *content* style of `seen` (short noun phrases) clearly does
follow the examples.

> **Examples teach the model WHAT TO NOTICE. They do not teach it HOW TO TYPE.**
> Anything mechanical must be enforced in the decoder, not requested in prose.

### Third diagnosis: length hurts

Our own record: V3 added a second example + calibration language and **regressed**. The
finding was written down as *"over-instructing a frozen 8B hurts."* Current blocks are
669–944 tokens. Roughly 400 of those teach format — which we just established does not work.

---

## 2. The fix, in three moves

### Move 1 — Format goes to the decoder, permanently
The schema-walk decoder (`ICL_DIFF_CONTROLLER.md` §7) forces keys, punctuation and field
order as prefilled literals; the model only fills value slots. Format stops being a request
and becomes a **guarantee**.

**Effect:** ~400 tokens deleted from every ICL block. Malformed JSON becomes impossible.
`fps` spam becomes impossible. And per our own "more prompt = worse" finding, the shorter
prompt should *help* accuracy — latency and accuracy pull the same direction here.

### Move 2 — Every conditional rule becomes an unconditional slot + code logic

| Was (failed) | Becomes |
|---|---|
| "emit `fps` only when it changes" | Model never emits `fps` on the default path; the `more` escape hatch exists for the rare deliberate change |
| "`new_event` true only if new" | Model reports a **level** ("is it true NOW"); **code** does rising-edge detection |
| "`answer` only when `have_enough_info`" | Decoder only *enters* the answer slot when the `hit` logit fires — the model never decides |
| "emit `{}` when quiet" | No such option; every tick fills `seen` + `hit` unconditionally |

Note this is already the project's most successful pattern — level→edge in code was called
*"the right architecture"* in `EXPERIMENTS.md`. **We are generalising a fix that already
worked once.**

### Move 3 — What's left in the prompt is only semantics
What counts as the event · how to phrase the answer · what to pay attention to.
That is the part a language model is genuinely needed for.

---

## 3. Writing rules for the semantics half

Derived from our own evidence, not from general prompting folklore:

1. **Positive over negative.** "NEVER copy the example text" / "NEVER re-report" — small
   models follow *do X* better than *don't do Y*, and a negation still puts the forbidden
   thing in context. Rewrite as what TO do.
2. **Concrete over abstract.** The worked examples are the highest-value tokens. Abstract
   calibration language ("more-likely-than-not") is what regressed in V3.
3. **One example, fully traced.** V3's second example hurt. Prefer one example that shows
   the *whole* arc — quiet ticks, the transition, the still-true tick, the return to false.
4. **Perception before judgment.** `seen` first is the single most robust finding in the
   project. Keep this ordering in every task.
5. **Budget the length.** Target ≤500 tokens of semantics. If a block grows, something must
   come out. Measure, don't assume.
6. **Say what the scorer wants.** Counting tasks need a parseable number; state monitor
   needs the state name; grounding needs exactly one grid cell. Verify against
   `omniprofast/metrics.py` — this is the most common silent failure.

---

## 4. The compliance probe — make "is my intention understood?" measurable ⭐

**This is the main new tool.** Today the only feedback on a prompt is task F1, which is
noisy, slow, and conflates *"the model misunderstood my instruction"* with *"the model
understood but got the video wrong."* Those are completely different bugs and we currently
cannot tell them apart.

Build `omniprofast/compliance.py`: over any run's tick log, compute — **with no ground
truth, no judge, and no GPU**:

| metric | catches |
|---|---|
| valid-JSON rate | parser fragility |
| required-field presence rate | fields silently dropped |
| **conditional-rule compliance** (e.g. was `fps` omitted when unchanged?) | **the 0/15 failure, directly** |
| field-order adherence | is `seen` actually coming first? |
| answer length vs the instructed cap ("under 25 words") | ignored constraints |
| **example-copy rate** (n-gram overlap with the ICL's own example text) | the "NEVER copy the example" failure |
| value-vocabulary violations (non-grid-cell in ETG, non-numeric in counting) | answers the scorer will auto-fail |
| tick-to-tick field churn | instability / flip-flopping |

**Runs in seconds on saved logs.** Use it as a *gate*: if compliance is low, the prompt was
never read — fixing semantics is pointless until it rises. If compliance is high and
accuracy is still low, the prompt *was* understood and the problem is comprehension or the
gate. **Two different bugs, now separable.**

Run it on the existing `output_*` logs immediately — we likely have months of unread
evidence about which instructions were ever followed.

---

## 5. The tuning ladder — sequential, one variable each

Only valid after the Phase-1 rig (one-pass multi-variant sweep + AUC) exists. Because all
rungs are evaluated in a **single forward stream** per video, every rung sees byte-identical
context — the comparison is exact, not statistical.

```
0. BASELINE            current V2-pure, on the new rig            -> AUC_0, compliance_0
1. FORMAT STRIPPED     delete format section, decoder enforces    -> expect compliance ~1.0
2. NEGATIONS FLIPPED   rewrite every "NEVER X" as "do Y"
3. LENGTH SWEEP        900 / 700 / 500 / 350 tokens of semantics  -> find the knee
4. EXAMPLE COUNT       1 vs 2 fully-traced examples               -> re-test V3's claim honestly
5. SEEN ABLATION       seen on/off                                -> confirm the 0.255 story
6. SEEN LENGTH         cap 6 / 12 / 20 tokens                     -> cheapest grounding that works
7. TRIGGER PHRASING    10-20 variants of the probe question       -> free, high-variance upside
8. ORDER               seen-then-hit vs hit-then-seen             -> is ordering the mechanism?
```

**Rules:** one variable per rung. Record AUC *and* compliance for every rung. Never advance
while compliance is falling — that means the change broke instruction-following, whatever
it did to accuracy.

---

## 6. When to hand over to GEPA

**Not before rung 8.** Hand-search first tells you whether the space is worth automating —
if rungs 1–8 move AUC by 2%, an optimizer will not save the idea; if they move it by 20%,
automation is clearly worth it.

Then: standalone [`gepa-ai/gepa`](https://github.com/gepa-ai/gepa) (not DSPy — its adapters
fight the KV-splice pipeline), Gemini as the reflection model, metric returning
`{"score": AUC, "feedback": <failure trace in words>}`, search space = **semantics only**,
because the decoder has already locked format.

That last point is the whole reason this ordering works: **every move in §2 shrinks what
GEPA has to search.**

---

## 7. Success criteria

- Compliance ≥ 0.95 on every mechanical metric (format is enforced, so this should be ~1.0)
- Prompt length ≤ 500 tokens of semantics per task
- Zero conditional emission rules remaining in any prompt
- Every accuracy claim carries a compliance number beside it
- A prompt change can be evaluated in < 5 min with error bars

When those hold, "my intention is not being understood" stops being a guess and becomes a
number you can watch move.
