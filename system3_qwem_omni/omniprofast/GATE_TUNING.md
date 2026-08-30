# Gate tuning — how the firing threshold is fitted, and the rules for doing it honestly

**Read this before touching `task_hit_thresholds` / `task_gate_modes` / `task_refractory_s`
in `async_omni_v2/config.py`, or before claiming any gate result.**

---

## 1. What is being tuned

The controller reads **`p_hit` = P(true)** off the logits every tick (a logit read of the
boolean slot — zero decode steps, so it is nearly free and, unlike a decoded `true`/`false`,
it is a *continuous* confidence). The output gate turns that number into an emission.

Every tick is logged:

```
[ 40.5s | vid  1.0s] ctrl.gate  [VIDEOID] fps=1.0 level=False ... p_hit=0.021 ...
```

Those log lines are the **entire dataset** for gate tuning. No GPU, no model, no judge —
the whole search is a replay of numbers already on disk.

A gate is **three coupled knobs**, not one:

| knob | meaning |
|---|---|
| `hit_threshold` | the cut-off `p_hit` must cross |
| `mode` | `edge` = fire only on the rising crossing (false→true); `level` = fire on every tick above |
| `refractory_s` | debounce: suppress any fire within *r* s of the previous one |

**Using a fitted threshold without its mode + refractory does not reproduce the fitted
number.** They are one config.

---

## 2. Tools

| tool | what it does | needs a judge? |
|---|---|---|
| `sweep.py` | **the big search.** All gate families (fixed + adaptive) in one fair arena, per-task, held-out. Traces cached → thousands of configs in seconds. | no |
| `resweep.py` | the original per-task fit (fixed family only), `time_f1` | no |
| `auc.py` | AUC/AP of the per-tick confidence + threshold sweep. The *dense* metric: ~3000 labelled tick decisions/run vs ~33 F1 events — use it to iterate above the noise floor. | no |
| `refit.py` | filters the run's **actual predictions** and rescores full content → `content_acc`/`joint_f1` | **yes** |
| `gates.py` | the original adaptive-gate presets. Superseded by `sweep.py` (see §5). | no |

```bash
python sweep.py output_full9                  # held-out (dev=0.5), all families
python sweep.py output_full9 --dev 0           # ceiling (fit-on-all)
python sweep.py output_full9 --families fixed  # just the baseline family
python sweep.py output_full9 --emit-config     # config.py-ready dicts
```

First run parses ~60 MB of logs and caches to `output_full9/_traces_cache.json`;
after that, iteration is instant. Use `--rebuild-cache` after a new eval run.

---

## 3. Non-negotiable rules

1. **A bigger grid ALWAYS inflates fit-on-all.** It is an argmax over a superset, so the
   fitted number is mathematically guaranteed to be ≥ the smaller grid's. Reporting
   "bigger grid → higher F1" from fit-on-all alone is a fake result. **Always compare
   held-out.** If held-out moves with fit-on-all, the gain is real; if it stays flat, the
   extra resolution is fitting noise.
2. **Watch for rail-pinning.** If the winning config sits at the edge of the search range
   (min threshold, max refractory), the grid is *truncating the search* — widen it and
   re-fit. This is how the 300–600 s refractories for one-shot tasks were found; the old
   30 s cap was hiding them.
3. **Compare families at equal budget.** Never pit a per-task-tuned family against
   hand-picked global presets (see §5).
4. **This is a SCREEN, not a verdict.** Offline replay assumes `p_hit` is independent of
   the gate. It weakly is not: firing appends to `reported`, which is fed back into the
   prompt, which shifts later `p_hit`. **Confirm the winner on GPU.**
5. **The eval is not bit-reproducible.** Treat differences under ~0.01 F1 as noise, and
   average over runs (N≥3) for anything reported.
6. **`time_f1` is judge-free; `joint_f1`/`content_acc` are not.** Do threshold search on
   `time_f1` (free, fast). Only pay for a judge when you need the content columns.
   Judge backend: `OMNIPRO_JUDGE_BACKEND=openai` (OpenAI has quota; **Gemini does not**).

---

## 4. What is established (2026-08-05, `output_full9`, 932 samples, 160,915 ticks)

Pooled `time_f1`, fixed family:

| config | fit-on-all | held-out |
|---|---:|---:|
| old global `p_hit > 0.5` | 0.190 | — |
| best *single* global | 0.254 | — |
| per-task, original 156-config grid | 0.316 | 0.308 |
| per-task, wide 1292-config grid | 0.334 | **0.327** |

- Per-task beats global decisively, and **the fit barely overfits** (held-out is only
  ~0.008 below fit-on-all) — so report held-out; it costs almost nothing.
- Widening the grid was a **real** gain (held-out rose in step, +0.019).
- **Grid tuning is saturated:** pushing refractory past 180 s gained +0.0006 (noise).
- The **F1 surface is flat near the optimum** — fit-on-all and held-out picked *different*
  configs yet landed within 0.007 F1. Do not over-trust the exact constants. The robust,
  reportable finding is the **regime**:

| task type | regime |
|---|---|
| one-shot (ETG, instant_event_alert, snapshot_counting) | high threshold + very long refractory ≈ *"fire once, then shut up"* |
| dense / continuous (sequential_step_instruction, event_narration) | **low** threshold + short refractory |
| counting (dedup, cumulative) | very high threshold + short refractory |

Judged table under the fitted thresholds (OpenAI `gpt-5-mini`, deployable refit filter):
as-run old gate 24,593 emits / `time_f1` 0.190 / `joint_f1` 0.071 → fitted per-task
5,572 emits / **0.268 / 0.118** (+41% time, +66% joint, 4.4× fewer emissions).

---

## 5. A mistake to not repeat

An earlier screen concluded *"fixed per-task thresholds beat adaptive stream-relative
gates (0.316 vs 0.211), so per-video self-calibration doesn't pay off."* **That conclusion
was invalid.** The fixed family had been searched over ~1300 configs per task including
refractory; the adaptive family was run as 15 hand-picked **global** presets, applied
identically to every task, with **no refractory at all** — and refractory is one of the
biggest levers there is. It compared a tuned family against an untuned one.

`sweep.py` exists to prevent exactly this: every family emits a per-tick boolean and
passes through the **same** edge/level + refractory layer, and every family gets the same
per-task search budget. If you add a new gate idea, add it as a `sig_*` signal function so
it inherits the fair arena automatically.

**Open question:** with the arena fixed, do adaptive/hybrid gates beat fixed per-task
thresholds? Run `python sweep.py output_full9` and read the `family wins:` line.

---

## 6. Wiring status (important)

`config.py` carries all three per-task dicts, but **only `task_hit_thresholds` is wired**
(`omniprofast/system5_adapter.py`, per-sample via `sample.task`). `task_gate_modes` and
`task_refractory_s` are **not yet injected**, and the live controller still uses
`gate_strategy="hysteresis"`, which is *not* the edge/level + refractory rule the fit
assumes. **Until that is wired, a live run does not reproduce the fitted numbers.**
