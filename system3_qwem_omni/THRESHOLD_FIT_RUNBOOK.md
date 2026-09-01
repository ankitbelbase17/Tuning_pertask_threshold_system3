# THRESHOLD FIT — runbook, deviations and measurements

Operational companion to **`THRESHOLD_FIT_v2.md`** (the spec) for the run executed
by user `dthapa`. The spec says *what* to measure; this says *how it is actually
being run here*, and records every place the run departs from the spec **and why**.
Referenced from `CLAUDE.md` §4.

Experiment root: `/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit`

---

## 0. Account and path differences from the spec

The spec was written by `dbartaula` and every path in its §0 points into that
scratch. Verified 2026-08-28 from the `dthapa` account:

| resource | status |
|---|---|
| `dbartaula/system3_qwem_omni` (code) | readable |
| `dbartaula/omnipro_data` (benchmark + 1,262 videos, 31 GB) | readable — **read in place, not copied** |
| `dbartaula/miniforge3/envs/prosync_env` (python 3.12.13, torch 2.12.1, transformers 5.12.1) | readable — used as-is |
| `dbartaula/omni_s3_eval` (chain harness to copy patterns from) | readable |
| **`dbartaula/hf_cache` (model weights)** | **mode 700 — NOT readable** |

So the only thing that had to be re-obtained is the backbone.
`bin/fetch_weights.py` pulls `Qwen/Qwen2.5-Omni-7B` (20 files, 21 GB, 344 s) into
`/iopsstor/scratch/cscs/dthapa/hf_home`. It must run on a **login node** —
compute nodes have no outbound network. `HF_HUB_DISABLE_XET=1` is required because
`prosync_env` has no `hf_xet` and is not writable by this account.

The code under test is a **copy**, not a git worktree:
`omni_thr_fit/repo/`. The spec asks for a worktree, but this checkout carries 13
modified and 15 untracked files that a worktree would not include — the state under
test is the *working tree*, not `HEAD`. A copy captures it exactly and leaves the
user's checkout untouched.

**Ported fix.** The user's copy predates the 2026-08-28 `--max_dur` ordering fix, so
`evaluate.py` was applying the window-fit cap *inside* `load_samples`, i.e. **before**
the subset stride — the exact defect `THRESHOLD_FIT_v2.md` §2 warns is invisible in
the metrics. Upstream's fixed file was copied in; the rest of both trees is
byte-identical (`diff -rq` clean).

## 1. Fleet — two chains in parallel, both resumable

| fleet | shape | window | launcher |
|---|---|---|---|
| **debug** | 4 nodes × 4 GH200 = 16 lanes | 22 min, self-chaining | `bin/run_chain.sbatch` |
| **login** | ln003, GPUs with ≥ 45 GB free | 90 min/generation | `bin/login_chain.sh` |

Both pull from **one worklist** through one atomic claim (`lib/worklist.py`), and
completion is a property of the **cell**, not of a lane, so they cannot duplicate or
corrupt each other's work. Claims carry a 30-minute TTL because a lane SIGKILLed at
the wall cannot release its own claim and a permanent lock would deadlock the run;
the worst case after expiry is duplicated work, which per-sample resume absorbs.

Only ln003 is used: `ssh` between login nodes is refused for this account
(`Permission denied (publickey)`), so ln001/002/004 cannot be driven from here.

Account is **`a0264`**, not `a168` — see `CLAUDE.md` §5.

## 2. THE GATE — the substantive deviation, and the reason for it

`THRESHOLD_FIT_v2.md` §1 states that `OMNIPRO_HIT_THRESHOLD` is the firing knob and
"no code change is needed to sweep thresholds". **Measured, that is not true.**

Under the shipped `gate_strategy="hysteresis"` (`controller.py:608`) the fire test is

```python
fire = bool(answer) and armed and p_hit >= cfg.gate_high_thr and (vt - last_fire_vt) > cfg.debounce_s
```

`cfg.gate_high_thr` is a fixed global **0.5**. `cfg.hit_threshold` only decides
whether an `answer` gets decoded at all (`controller.py:268`, `if hit:`).

Evidence, from the **completed 2,700-sample phase-A run** of this same fork
(`bin/audit_gate_rail.py`, `bin/audit_gate_rail2.py` over 16 lane logs):

```
581,403 gate ticks | 50,433 fires | ZERO fires below p_hit 0.500

task                           shipped thr   min p_hit@fire   p10    verdict
event_narration                      0.100          0.5000  0.547   inert, ran at 0.5
sequential_step_instruction          0.010          0.5000  0.531   inert, ran at 0.5
instant_event_alert                  0.450          0.5000  0.531   inert, ran at 0.5
explicit_target_grounding            0.500          0.5000  0.562   binds (equals the rail)
realtime_state_monitor               0.800          0.5080  0.818   binds
cumulative_counting                  0.925          0.5230  0.932   binds
snapshot_counting                    0.985          0.5310  0.985   binds
semantic_condition_alert             0.980          0.8030  0.982   binds
dedup_counting                       0.992          0.5160  0.570   binds (45 fires only)
```

Only **0.1 %** of fires sit below their task's own threshold (a small leak through
the `more` escape hatch, which can supply an `answer` on a tick where
`have_enough_info` was false). So for the five tasks above 0.5 the threshold does
bind — but for the three below it, the configured value never mattered, and **every
grid point ≤ 0.5 would have produced the identical operating point**: 5 of the
spec's 10 pass-1 thresholds, ~70 GPU-h of duplicate cells.

Separately, `task_gate_modes` and `task_refractory_s` were defined in `config.py`
and **read nowhere** — the §1.1(a) defect, confirmed by grep.

**Fix applied** (§1.1(a) option 1, extended to cover the rail). A new
`gate_strategy="fitted"` implements the exact rule the offline fitters replay:

```python
above   = p_hit >= cfg.hit_threshold
crossed = above if cfg.task_gate_mode == "level" else (above and not prev_above)
fire    = bool(answer) and crossed and (vt - last_fire_vt) >= cfg.refractory_s
```

`system5_adapter.py` injects all three knobs per sample from the task dicts.
`OMNIPRO_GATE_FIT=0` restores the shipped hysteresis gate so the incumbent stays
runnable from the same checkout for the Table-4 ablation.

**A limitation this creates, and it must reach the paper.** The mode and refractory
now in force come from `config.py`, i.e. from the **vision-only Qwen3-VL-8B** fit —
only the *threshold* is being refitted for the omni model. §5's screen suggests
those inherited values are wrong for this backbone: it wants ~20 s of refractory for
`instant_event_alert`, not the shipped 600 s, and ~10 s for `explicit_target_grounding`,
not 300 s. So the result is a threshold fitted **conditional on another model's
mode/refractory**, which is defensible (it is one coherent, shipped config, and the
threshold is the knob the spec set out to fit) but is *not* the same as a full
per-task gate refit. `sweep.py` does the full three-knob search offline for free;
running it after pass 1 is the cheap way to bound how much is being left on the table.

Verified: `bin/test_fitted_gate.py` (no GPU) asserts edge/level/refractory semantics
and that all ten grid points now give distinct, monotone emission counts; the live
smoke run then showed a fire at vid 1.0 s followed by correct suppression at vid
7.0 s and 8.0 s under `semantic_condition_alert`'s 10 s refractory.

The live rule is now term-for-term identical to `resweep.simulate` — same
`p >= thr`, same edge/level test, same `vt - last >= refractory`, and `prev` updated
on every tick including suppressed ones — **with one gap the offline replay cannot
model**: the live gate also requires `bool(answer)`, i.e. that the schema walk
actually produced something to say. The replay has no access to that, so it will
systematically **over-count emissions** relative to a live run at the same
threshold. Treat §5's screen as an upper bound on emission volume, and never quote
a replayed F1 as a measured one.

### 2.1 The first tick is not a rising edge — found live, fixed 2026-08-28

The gate above was launched, and pass 1's first cells came back with a signature
that could not be right: **every** `instant_event_alert` sample emitted exactly
once, at video time **1.0 s**, against ground truth at 26–98 s. Time-F1 was
therefore structurally 0 at every threshold in the grid. The trace says why:

```
vid  1.0s  p_hit=0.792  rise=True  fire=True     <- essentially no context yet
vid 24.0s  p_hit=0.958  rise=True  fire=False    <- the real event, suppressed
```

Three things compose into that:

1. `p_hit` is **high on the first tick** — the model answers the boolean from the
   prompt before it has seen any video.
2. `prev_above` was seeded `False`, so tick 1 counted as a **rising edge**.
3. `instant_event_alert` carries a **600 s refractory** (inherited, unfitted),
   which then locks out the rest of the video.

**The fix is definitional, not a tuned constant: a rising edge requires a
predecessor, and the first tick has none.** `prev_above` now starts `None` and the
edge test is `above and prev_above is False`. `level` mode is deliberately
unchanged — every tick above the threshold is a level hit, including the first.

Measured offline over the finished 2,700-sample phase-A run
(`bin/audit_first_tick.py` → `FIRST_TICK_AUDIT.json`), best time-F1 across the
10-point grid, before → after:

| task | mode | refractory | as-is | edge-needs-predecessor |
|---|---|--:|--:|--:|
| snapshot_counting | edge | 600 s | 0.0535 | **0.0900** |
| instant_event_alert | edge | 600 s | 0.0636 | **0.0751** |
| explicit_target_grounding | edge | 300 s | 0.0813 | **0.0844** |
| cumulative_counting | edge | 7 s | 0.2505 | 0.2512 |
| dedup_counting | edge | 5 s | 0.2180 | 0.2171 |
| the four `level` tasks | level | 5–10 s | — | identical by construction |

The pattern is the point: **the fix matters exactly in proportion to the
refractory.** A short-refractory task recovers on the next tick, so wasting tick 1
costs ~0.001; a 600 s refractory turns it into the whole video.

The first-tick artifact is universal, not a quirk of one task — over all 2,700
phase-A samples the first tick is ≥ 0.05 for **98–100 %** of samples in every task,
median 0.34–0.64 (against an all-tick median of 0.15–0.68).

**What the audit also showed, correcting the alarm that started it.** The grid is
*not* collapsing: even as-is it produces 10 distinct emission sets for 8 of the 9
tasks (`instant_event_alert` gave 9). The n=37 single-task sample that raised the
flag was not representative. The fix is worth making because it is free and
strictly non-harmful, not because the sweep was about to be wasted.

A `warmup` knob (suppress all fires before *W* s) was measured too. It is the only
thing that also helps `level` mode, but the gains there are +0.002–0.006 — inside
the §6.2 noise band — and it would add a **fourth, unfitted** knob mid-experiment.
Rejected on those grounds.

**Applied to the live gate and to all five offline replays** (`resweep.py`,
`sweep.py`, `grid.py`, `gates.py`, `auc.py`) in the same commit: a screen that does
not match the live rule screens the wrong system. Previous versions kept as
`*.bak_before_edge1`.

**Cost of the correction: 37 sample-evals**, the only work banked at that point,
quarantined with its provenance under `.invalidated/edge1_20260828_2010/` (moved,
never deleted). Only `edge`-mode cells are invalidated by the change; the four
`level` tasks are bit-identical and none of them had banked yet.

### 2.2 The environment moved under the run — 2026-08-28 19:34

Twenty minutes after the relaunch above, every lane died at **`import torch`**:

```
RuntimeError: Unable to find torch_shm_manager at
  .../dbartaula/miniforge3/envs/prosync_env/lib/python3.12/site-packages/torch/bin/torch_shm_manager
```

`torch/__init__.py` ends with `_C._initExtension(_manager_path())`, and
`_manager_path()` raises if that binary is absent — so this is not a degraded
mode, it is a hard import failure.

**Cause: the environment belongs to another account and was being rewritten.**
`find -newermt` showed **594 files changed that afternoon** under
`dbartaula/.../prosync_env`, including a fresh `pip-26.2.dist-info` and a fresh
`torch-2.12.1+cu130.dist-info` — a pip reinstall of torch, in flight. `torch/bin/`
and `torch/lib/libshm*` were empty; `torch/lib/*.so` (dated Aug 14) were not. The
same sweep took out `pip/_vendor/certifi/cacert.pem`, so even `pip` could not run.

This was always the real risk, and it is not one that careful reading catches:
the spec's paths were correct when written. **A read-only dependency owned by an
account that is actively changing it cannot underpin a multi-day run.**

**Fix — own the whole runtime** (`bin/repair_env.sh`):

1. Private copy of the env → `/iopsstor/scratch/cscs/dthapa/envs/prosync_env`.
2. **torch replaced wholesale** from the official
   `torch-2.12.1+cu130-cp312-cp312-manylinux_2_28_aarch64.whl`, not patched. The
   source was being rewritten *while we copied it*, so the copy could be torn in
   ways no file-by-file check would catch; unpacking the exact build the env
   reports makes the one package that was in flux verifiably consistent. The
   possibly-torn copy is kept as `site-packages/.torch.torn_<stamp>`.
3. `pip`'s vendored CA bundle restored from the env's own `ssl/cacert.pem`.
4. Package set verified **identical to the original** — both lack
   `torchaudio`, `decord`, `librosa`, `soundfile`, `scipy`, `qwen_omni_utils`,
   which is why the pipeline never noticed. `matplotlib` added for §9's figures.
5. Verified: torch 2.12.1+cu130, CUDA visible, 4 devices; the omni backend loads
   in **67 s / 16.6 GiB**.

**The dataset was copied too** — 31 GB, 1849 files, every file verified
byte-identical in size. Not because it had changed, but because it is the one
remaining cross-account dependency whose corruption would be **silent**: a
replaced video does not raise, it produces different predictions. `env.sh` now
points at `/iopsstor/scratch/cscs/dthapa/omnipro_data`.

The model weights were already private (dbartaula's `hf_cache` is mode 700, which
forced a fresh download on day one — an accident that turned out to be the right
architecture).

**Nothing was deleted from the other account's tree at any point.**

### 2.3 What holding the vision-only refractory actually costs — measured

RUNBOOK §2 recorded "only the threshold is refit; mode and refractory keep their
vision-only values" as an open limitation. `bin/audit_refractory.py` replays the
finished 2,700-sample phase-A tick stream across 12 refractory values x the
10-point threshold grid and turns that limitation into a number.

Best time-F1 over the threshold grid, at each refractory (offline screen, edge1
gate). `ship` is the value currently in force:

| task | ship | 5 s | 7 s | 20 s | 600 s | best | gain vs shipped |
|---|--:|--:|--:|--:|--:|--:|--:|
| sequential_step_instruction | 7 | 0.356 | **0.414** | 0.294 | 0.087 | 7 | +0.000 |
| cumulative_counting | 7 | **0.256** | 0.251 | 0.197 | 0.074 | 5 | +0.005 |
| event_narration | 7 | 0.207 | **0.241** | 0.199 | 0.033 | 7 | +0.000 |
| realtime_state_monitor | 7 | 0.199 | **0.237** | 0.181 | 0.085 | 7 | +0.000 |
| dedup_counting | 5 | 0.217 | **0.228** | 0.184 | 0.121 | 7 | +0.011 |
| semantic_condition_alert | 10 | 0.177 | **0.200** | 0.155 | 0.061 | 7 | +0.003 |
| explicit_target_grounding | 300 | 0.078 | 0.083 | 0.088 | 0.085 | 60 | +0.012 |
| snapshot_counting | 600 | 0.081 | 0.088 | **0.096** | 0.090 | 20 | +0.006 |
| **instant_event_alert** | **600** | 0.087 | 0.095 | **0.109** | 0.075 | **20** | **+0.033** |

**The conclusion is reassuring, and it was not the expected one.** The inherited
vision-only refractory is at or beside its optimum for **eight of nine tasks** —
every gain is inside the §6.2 noise band of 0.03. The vision-only fit *transfers*.
Holding mode and refractory fixed while fitting the threshold is therefore a much
smaller compromise than §2 assumed, and that is now measured rather than hoped.

Two things this does **not** say:

1. **Refractory matters enormously in general** — it is simply already tuned.
   `sequential_step_instruction` runs 0.414 at 7 s and 0.087 at 600 s, a 4.8x
   spread. Nothing here licenses treating the threshold as a free-standing knob.
2. **`instant_event_alert` is the one genuine exception**, and it is the same task
   §2.1 caught: at its shipped 600 s refractory a >=233 s video gets exactly one
   emission, so time-F1 is ~0 at *every* threshold and the pass-1 grid for that
   task will most likely come back FLAT -> keep-shipped, i.e. produce no fit at
   all. At 20 s it reaches 0.109 (+0.033, just past the noise band).

`snapshot_counting` also carries 600 s but only gains +0.006 — the earlier worry
that every long-refractory task was compromised was too broad.

**Not acted on.** §3/§4 scope this experiment to a threshold fit, and widening it
mid-run is a scope change, not a bug fix. The number is recorded here so the
decision can be made on evidence: refitting `instant_event_alert`'s refractory to
~20 s is the single highest-value extension available, and it is the only one
whose expected gain clears the noise band. Note the caveat that governs the whole
table — this is an offline **screen** that cannot model the live `bool(answer)`
requirement and therefore over-counts emissions, so read the column comparison,
never the absolute values.

### 2.4 The scheduler spun instead of working — 2026-08-28, 26,277 wasted claims

Seven generations after the relaunch the dead-chain guard stopped the run at
**82 of 1410** samples. Progress per generation: `40 → 76 → 79 → 80 → 81 → 81 → 81`.
The guard was right; the diagnosis is the interesting part.

One lane logged **76 units attempted in a single 22-minute window** — twelve
seconds each, less than a model load. Across the seven generations the lanes made
**26,277 claims, every one of them for `instant_event_alert`**, while **84 of the
94 cells were never touched once**.

**Cause: a hole between two definitions of "done".**

| question | asked by | answer for a long-tail cell |
|---|---|---|
| are all 15 frozen ids banked? | `cell_complete` | **no** |
| is there work this window can finish? | nobody | **no** |

A cell whose remaining samples are all longer than the generation's `--max_dur`
is **incomplete but not actionable**. `worker.sh` claimed it, `evaluate.py`
filtered every sample out and exited **0** in ~12 s, `worker.sh` released it —
and `next_unit` returns *the first incomplete claimable unit in worklist order*,
so it handed back **the same unit again**. Forever.

`next_unit` already had a `max_dur` parameter. **It was never read and never
passed** — the filter was designed and left unwired, which is exactly why nothing
looked wrong on inspection.

**Fix, in two layers, because the unit and the cell are not the same thing.**

1. `worklist.cell_actionable(pass, task, thr, max_dur)` — skip cells with no
   remaining sample inside the window; `claim_cli.py` now passes `--max_dur`.
   Kills the broad loop.
2. A unit is a **shard** of a cell: shard 0 can be barren while shards 1-3 are
   not, which the cell-level check cannot see. `worker.sh` now counts the cell's
   rows **before and after** — not the exit code, which is 0 whether evaluate.py
   finished samples or filtered them all away — and adds genuinely barren shards
   to a per-generation `--skip` list.

Verified before relaunch: one lane pulling 12 units skips the 8 exhausted
`instant_event_alert` cells, takes the 2 with fitting work, and advances to
`semantic_condition_alert`. Verified after relaunch: a **second task** began
banking within minutes — something that had not happened once in seven
generations.

**Also cleared: stale claim directories.** `CLAIM_TTL` is 1800 s but a 4-node
generation is 1320 s, so a claim held by a SIGKILLed lane **outlives the
generation that made it** and blocks the next one. The TTL was chosen to be
longer than any single unit; it is also longer than the whole window, which is
the wrong comparison. Claims left by the aborted chain were removed before
relaunch.

**The 82 banked samples are unaffected** — the bug wasted scheduling, not results.

### 2.5 `p_hit` does not rank event ticks above quiet ones — measured 2026-08-29

The whole fit assumes `p_hit` carries timing information and that the only open
question is where to cut it. On the five pass-1 tasks that had a complete
reference cell, **it does not.**

| task | AUC @ ±3 s | 95 % CI | best AUC over any offset in ±10 s | `p_hit` range (distinct) |
|---|---|---|---|---|
| instant_event_alert | 0.551 | [0.45, 0.65] | 0.551 | 0.001–0.990 (247) |
| snapshot_counting | 0.526 | [0.43, 0.63] | 0.542 | 0.001–0.995 (280) |
| explicit_target_grounding | 0.512 | [0.40, 0.63] | 0.512 | 0.000–0.985 (232) |
| cumulative_counting | 0.469 | [0.43, 0.53] | 0.509 | 0.003–0.997 (284) |
| semantic_condition_alert | 0.431 | [0.36, 0.53] | 0.487 | 0.002–0.991 (253) |

Chance is 0.5 and **every interval contains it.**

**Three innocent explanations were excluded before this was written down**
(`bin/audit_perception.py`, results in `PERCEPTION_AUDIT.json`):

- *A clock offset* between the log's `vid Ns` stamp and the benchmark's
  `trigger_time_sec` would destroy AUC while leaving the score informative. The
  tick clock was swept over ±10 s in 2 s steps: AUC stays inside 0.39–0.55
  throughout with **no peak anywhere**. A real offset would show one.
- *Too tight a tolerance* — if the model anticipated or lagged events by more
  than 3 s the ±3 s label would score its correct high confidences as negatives.
  Re-run at ±10 s: unchanged.
- *A degenerate score* — AUC is near 0.5 by construction if `p_hit` is pinned or
  coarsely quantised. It is neither: 232–284 distinct values spanning essentially
  the full unit interval, with an interquartile range of 0.2–0.7.

So the score is well spread and genuinely uninformative about **when**.
`semantic_condition_alert` is worse than that: it sits *below* chance and falls
monotonically as the offset goes positive (0.487 at −10 s → 0.389 at +10 s), i.e.
its confidence is systematically **lower** near an event.

**Fig 3 says the same thing from the other side, with no shared code path.**
Across the full grid, precision is FLAT — cumulative ≈ 0.15, dedup ≈ 0.17,
narration ≈ 0.16, state monitor ≈ 0.18, semantic ≈ 0.11, sequential ≈ 0.25 —
while recall falls monotonically as the threshold rises. Firing less often buys
no precision. That is the F1-space signature of an uninformative score, and it is
what an AUC of 0.5 predicts.

**What this means for the fit, stated before the numbers land so it cannot be
back-fitted:** with precision threshold-invariant, time-F1 is maximised by
whatever threshold emits most, so the per-task argmax will pin to the **low rail**
of the grid for every task that emits at all. `GATE_TUNING.md` names rail-pinning
as a result to distrust; here we will have the mechanism for it in advance, which
is the difference between a suspect number and an explained one. `pick.py` must
report the rail, and §6.1's fit-disjoint number is what decides whether the fit
generalises or is noise-fitting.

**The caveat travels with the number.** n = 15 videos per task on the frozen fit
subset, and the CIs straddle 0.5, so what this supports is *"indistinguishable
from chance at this n"* — never *"proven to be chance"*. Re-run
`bin/audit_perception.py --stage3 <dir>` on the full evaluation, where n = 300
per task, before the claim goes in the paper at full strength.

**This is a result, not a blocker.** The experiment was built to be able to
detect exactly this, and a negative answer to "does per-task threshold fitting
help?" is reportable provided the mechanism is measured rather than assumed. Do
not respond to it by widening the grid: no threshold can recover information the
score does not carry.

### 2.6 The fit does not survive its own bootstrap — pass 1 complete, 2026-08-30

Pass 1 finished at 04:24 on 2026-08-30: **1410/1410 samples, 94/94 cells, every
cell `reliable`**. `lib/pick.py --pass p1` produced a winner for all nine tasks.
Before treating any of them as a fitted parameter, `bin/audit_fit_noise.py` asks
the only question that matters about a maximum taken over a 10–12 cell grid on 15
videos: **would the same threshold win again on a different sample of videos?**

The test is a *paired* bootstrap — every cell of a task ran the same fifteen
frozen ids, so a draw holds the video set fixed across thresholds and removes
between-video variance, which dominates. That makes it the most favourable test
the fit could ask for. It replays `pick.rank` itself on each draw rather than a
bare argmax, because §3's tie-band is part of the rule and on three tasks it moves
the pick off the raw maximum. 2000 draws:

| task | fitted | re-selected | chance | span | ΔF1 vs best rival, 95 % CI |
|---|---|---|---|---|---|
| sequential_step_instruction | 0.15 | 61 % | 8 % | 0.286 | [−0.027, +0.046] |
| event_narration | 0.65 | 60 % | 10 % | 0.151 | [−0.038, +0.043] |
| realtime_state_monitor | 0.45 | 58 % | 10 % | 0.215 | [−0.021, +0.057] |
| cumulative_counting | 0.85 | 44 % | 10 % | 0.193 | [−0.116, +0.137] |
| snapshot_counting | 0.55 | 28 % | 10 % | 0.200 | [−0.133, +0.333] |
| dedup_counting | 0.35 | 25 % | 10 % | 0.185 | [−0.077, +0.034] |
| explicit_target_grounding | 0.05 | 25 % | 10 % | 0.067 | [−0.207, +0.182] |
| semantic_condition_alert | 0.25 | **6 %** | 8 % | 0.060 | [−0.059, +0.021] |
| instant_event_alert | 0.05 | 100 %† | 10 % | **0.000** | [0, 0] |

† degenerate, not stable: see below.

**Every interval contains zero.** On no task is the fitted threshold significantly
better than the best alternative in its own grid, under the most generous pairing
available. `semantic_condition_alert` is worse than that — its pick is re-selected
**6 %** of the time against an 8 % chance rate, i.e. the rule reproduces itself
*less* often than picking a cell at random; the bootstrap prefers 0.15 (33 %).

Two tasks are not fits at all and must not be reported as such:

- **`instant_event_alert` scores `time_f1 = 0.000` at all ten thresholds.** The
  100 % re-selection is `pick.rank` breaking a ten-way tie deterministically, not
  agreement. `pick.py` already flags it `FLAT` and keeps the shipped 0.45, which
  is the right behaviour. The system never lands a single emit within ±3 s on this
  task at any gate setting.
- **`explicit_target_grounding` alternates 0.065 / 0.000 / 0.061 / 0.000** across
  the grid. With 15 emits against 16 GT events, `F1 = 2·tp/31`: those values are
  **one matched event, or zero**. The whole curve is a single coin flip per cell,
  and its CI is correspondingly the widest of any task, [−0.207, +0.182].

**The one thing that IS reproducible is the high-rail collapse.** Every task falls
off a cliff at 0.95 (F1 0.000–0.170, against 0.046–0.361 elsewhere) — the gate
firing too rarely to match anything. Below ~0.75 every curve is flat inside the
§6.2 noise band. So the threshold does control something real, but only *emission
volume*, and only at the extreme; it does not select *which* moments to emit at.
That is exactly what §2.5's AUC ≈ 0.5 predicts, reached through a completely
separate code path — one measures the score's ranking, this one measures the
selection's reproducibility.

**A prediction of mine was wrong, and it is recorded because it was made in
advance.** §2.5 predicted the fit would *pin to the low rail on every task that
emits*. It did not: only two of nine railed low, and the picks scattered across
the mid-grid (0.15–0.85). Flat-curve-plus-noise produces mid-grid scatter, not
rail-pinning, so the specific prediction was wrong even though the underlying
claim it was drawn from — that there is no signal to fit — is what the CIs above
confirm. The reasoning error was assuming a flat *precision* curve implies F1
rises monotonically with recall; it does not, because the 0.95 collapse makes the
curve non-monotonic at the top end and noise dominates everywhere below it.

**Advance prediction for pass 2 and stage 3**, recorded now, before pass 2
finishes, so it cannot be back-fitted:

1. Pass 2 refines around the pass-1 picks on a finer grid. Its ΔF1-vs-rival CIs
   will contain zero on **at least seven of nine** tasks.
2. On the held-out 2,700-sample stage-3 run, `time_f1` under `FINAL_THRESHOLDS`
   will land **within the 0.03 noise band** of the same run under the shipped
   thresholds, on the overall pooled number.

If (2) is falsified the fit is real and §2.5–2.6 are wrong; that is the point of
writing it down. §6.1's fit-disjoint comparison is the arbiter, not the fit's own
cells. Evidence → `omni_thr_fit/FIT_NOISE_AUDIT.json`,
`omni_thr_fit/bin/audit_fit_noise.py`, `omni_thr_fit/P1_PICKS.json`.

### 2.7 The gate has ONE identifiable degree of freedom, not nine — 2026-08-30

`lib/ablation.py` builds Table 4 on the pass-1 cells (9 tasks, 135 videos), pooling
tp/fp/fn micro-averaged exactly as `metrics.aggregate` does, with every arm scored
on the same videos so the arm-to-arm differences can be bootstrapped **paired**
(2000 draws):

| arm | time-P | time-R | time-F1 | Δ vs best single global | 95 % CI | off-grid |
|---|---|---|---|---|---|---|
| `global_0.5` | 0.142 | 0.565 | 0.2271 | −0.0198 | [−0.039, −0.003] | 9/9 |
| **`best_single_global`** (0.15) | 0.155 | 0.601 | **0.2469** | — | — | 0/9 |
| `shipped_per_task` | 0.151 | 0.356 | 0.2115 | −0.0354 | [−0.055, −0.016] | 7/9 |
| `fitted_per_task` | 0.163 | 0.582 | 0.2544 | **+0.0075** | **[−0.011, +0.027]** | 0/9 |

**Nine fitted thresholds do not beat one.** The fitted arm's advantage over a
single global threshold is +0.0075 with an interval straddling zero — and this is
the *in-sample* comparison, where the nine thresholds were fitted on these very
videos and the single global one has one free parameter against their nine. Nine
degrees of freedom cannot beat one even with the generalisation penalty removed.

**But the global threshold itself is real, and that is the contrast that matters.**
Applying §2.6's own test one level up — re-running the whole single-global sweep
inside every bootstrap draw — 0.15 is re-selected in **94 %** of draws against a
10 % chance rate (next best: 0.55 at 2 %). Set against the per-task fits'
25–61 %, the picture is sharp:

> Pooled over 135 videos the gate's operating point is **identified**; split nine
> ways over 15 videos each it is **not**. The data supports one threshold, not a
> table of nine. It is a question of statistical power, not of the tasks being
> secretly identical — the per-task curves differ, they are just each estimated
> too noisily to rank.

`shipped_per_task` is significantly *worse* than a flat global 0.15 (−0.0354, CI
excludes zero), driven by recall: 0.356 against 0.601. Seven of its nine values sit
off-grid (0.992, 0.985, 0.98, 0.925 …) and are served by the nearest cell actually
on disk — every substitution and its distance is recorded in `ABLATION.json`
under `off_grid`, and the arm must be reported as an approximation. The same
caveat applies with more force to `global_0.5`: the grid steps 0.45 → 0.55, so all
nine of its cells are really **0.45**. Its −0.0198 should be read as "0.45 is worse
than 0.15", not as a measurement at exactly 0.5.

**Why 0.5 is the right baseline to name anyway.** Under the shipped
`gate_strategy="hysteresis"` the fire test was `p_hit >= gate_high_thr`, a fixed
global **0.5**, and `hit_threshold` was inert (§2). So `global_0.5` is not a
strawman — it is what the system actually did before this study, and moving the
single global operating point from 0.5 to 0.15 is the one intervention here with
a stable, reproducible effect.

This does not retire §2.5. AUC ≈ 0.5 says `p_hit` cannot say *when*; §2.6–2.7 say
the threshold therefore only buys emission *volume*, and volume has exactly one
useful setting. The three findings are one mechanism measured three ways.

Re-run on stage 3 (`--pass p2 --picks FINAL_THRESHOLDS.json`) for the paper's real
Table 4; these numbers are the fit-subset screen. Evidence →
`omni_thr_fit/ABLATION.json`, `omni_thr_fit/lib/ablation.py`.


### 2.8 Pass 2 complete — advance prediction 1 CONFIRMED, 9/9 — 2026-08-30

Pass 2 finished 675/675 samples, 45/45 cells, every cell `reliable`. `pick.py
--pass p2` produced `FINAL_THRESHOLDS.json`; `bin/audit_fit_noise.py --pass p2`
re-ran the paired video bootstrap on the refined grid.

**The prediction registered in §2.6 before pass 2 finished was: "its ΔF1-vs-rival
CIs will contain zero on at least seven of nine tasks." The outcome is 9 of 9.**

| task | fitted | rival | ΔF1 | 95 % CI | re-sel. | chance | grid span |
|---|---|---|---|---|---|---|---|
| cumulative_counting | 0.617 | 0.500 | −0.0017 | [−0.075, +0.085] | 48 % | 20 % | 0.048 |
| dedup_counting | 0.417 | 0.383 | +0.0243 | [−0.025, +0.074] | 70 % | 20 % | 0.055 |
| event_narration | 0.567 | 0.233 | +0.0037 | [−0.027, +0.031] | 74 % | 20 % | 0.022 |
| explicit_target_grounding | 0.717 | 0.317 | +0.0039 | [−0.182, +0.194] | 47 % | 20 % | 0.065 |
| instant_event_alert | 0.100 | 0.067 | +0.0645 | [+0.000, +0.200] | 63 % | 20 % | 0.065 |
| realtime_state_monitor | 0.350 | 0.300 | −0.0093 | [−0.033, +0.020] | 38 % | 20 % | 0.029 |
| semantic_condition_alert | 0.217 | 0.183 | +0.0091 | [−0.027, +0.061] | 42 % | 20 % | 0.066 |
| sequential_step_instruction | 0.085 | 0.042 | +0.0025 | [−0.027, +0.033] | 41 % | 20 % | 0.022 |
| snapshot_counting | 0.383 | 0.217 | +0.0667 | [+0.000, +0.200] | 56 % | 20 % | 0.133 |

**Read the two boundary rows honestly.** `instant_event_alert` and
`snapshot_counting` report CI **[+0.000, +0.200]** — the lower bound is exactly
zero, so they do *not* exclude it, and the interval is two grid-quantised jumps
wide because these tasks match 0, 1 or 2 events out of 15–16. That is a
degenerate count, not a tight positive effect. Reporting them as "CI excludes
zero" would be a straightforward misreading of an inclusive bound.

**The re-selection rates look better than pass 1's and are not.** They read 38–74 %
against 25–61 %, but chance here is 20 % (5 cells per task) versus 8–10 % in pass 1
(10–12 cells). As a multiple of chance, pass 2 is 1.9–3.7× where pass 1 was
2.5–6.1× — no better, and on four tasks worse. A finer grid packs the cells closer
together, so the argmax is *less* separable, not more. Always divide by chance
before comparing re-selection across grids of different sizes.

**What pass 2 actually adds** is the span column. Across each refined
neighbourhood the entire F1 range is **0.021–0.133**, and on four of nine tasks
*every* cell sits within the 0.03 noise band of the best. Refining the grid did
not resolve a winner; it confirmed there was no gradient there to resolve. This is
§2.5's AUC ≈ 0.5 arriving a fourth time by a fourth independent route.

**Note on `FINAL_THRESHOLDS.json`:** §4's rule ranks pass-2 cells ∪ the two pass-1
candidates, so a *coarse* point can win the finalise — `realtime_state_monitor`
lands on 0.450 and `dedup_counting` on 0.550, neither of which is on the pass-2
grid. `instant_event_alert` stays at its shipped 0.450 because §2.6 flagged it
FLAT. The audit therefore falls back to the rule's own pick on the pass-2 grid
whenever the final threshold is not a cell that was actually run; the alternative
would be scoring a cell that does not exist.

**Only prediction 2 is still open** — that stage-3 `time_f1` under
`FINAL_THRESHOLDS` lands within 0.03 of the same run under the shipped
thresholds. §6.1's fit-disjoint comparison remains the arbiter. Evidence →
`omni_thr_fit/FIT_NOISE_AUDIT_P2.json`, `omni_thr_fit/FINAL_THRESHOLDS.json`,
`omni_thr_fit/results/p2/CELLS.json`.


### 2.9 Stage 3 launched as TWO arms, and why the second one is not optional — 2026-08-31

Stage 3 is the headline eval: the complete 2,700-sample OmniPro Online run with
`FINAL_THRESHOLDS` loaded from `config.py`, no override. It is running now
(`bin/run_stage3.sbatch`, 4 nodes × 4 lanes, self-chaining on `afterany`), and the
exact edit it is running under is banked as
`omni_thr_fit/STAGE3_CONFIG_APPLIED.diff` — `preflight_stage3.sh` asserts
`config.task_hit_thresholds == FINAL_THRESHOLDS.json` at every launch, and that
diff is the assertion's paper trail. Only the nine thresholds move;
`task_gate_modes` and `task_refractory_s` are inherited untouched from the
vision-only fit, which is the standing limitation of §2.

**A single arm cannot answer the question this experiment asks.** §2.7 measured
`fitted_per_task` 0.2544 against `best_single_global` (0.15) 0.2469 — **+0.0075,
CI [−0.011, +0.027]** — *in sample*, on the 135 videos the thresholds were fitted
on, where the fitted arm has nine free parameters to the global arm's one. A
stage-3 run of the fitted arm alone produces one number with nothing to subtract
it from: it would show that the fitted system scores *X*, not that per-task
fitting bought anything. So stage 3 runs twice — `fitted` (per-task from
`config.py`) and `g015` (flat `OMNIPRO_HIT_THRESHOLD=0.15` on every task).
`debug-qos` MaxJobsPU=1 makes the arms sequential regardless.

**This is the first version of that comparison with the power to resolve a sign.**
CI half-width scales as 1/√n, so 135 → 2,700 videos narrows §2.7's ±0.019 to
roughly ±0.004: an effect the size of the in-sample +0.0075 would land as a CI
excluding zero. §2.7's null was a *power* result, and this is the run that
converts it into a claim either way.

**Registered in advance, before either arm finishes.** Out of sample the fitted
arm's nine parameters no longer have the advantage of having been fitted on the
scored videos, so (i) the point estimate of ΔF1(fitted − g015) will come in
**below the in-sample +0.0075**, and (ii) |ΔF1| will fall **inside the 0.03 noise
band** (§6.2). I am *not* predicting the sign. Prediction 2 of §2.6 — stage-3
`time_f1` under `FINAL_THRESHOLDS` within 0.03 of the shipped thresholds — stays
open and is scored from the same arms.

**The arm is a pairing, not a flag.** `STAGE3_ARMS.json` maps each arm to *both*
its results directory and its threshold override, and `lib/arms.py` is the only
thing that reads it. The failure this shape prevents is specific: an arm that gets
the right directory with the wrong override completes normally and banks
well-formed predictions under the other arm's name, and **nothing downstream can
detect it**, because a prediction record does not carry the threshold that
produced it. `stage3_worker.sh` therefore *asserts* the pairing per lane rather
than trusting its caller — the fitted arm aborts if `OMNIPRO_HIT_THRESHOLD` is set
at all, the control arm aborts if the export did not take. An unset `$S3_ARM`
resolves to `fitted`, byte-identical to the pre-arm behaviour, which is what
allowed an already-queued chain to be patched underneath.

**And the records now say which arm made them.** `evaluate.py` already carried the
slot — `pred["arm"] = os.environ.get("OMNIPRO_ARM", "")` — and stage 3 was leaving
it empty, so the *directory* was the only surviving record of which gate produced a
prediction and a file moved or merged by hand lost that fact silently.
`stage3_worker.sh` now exports `OMNIPRO_ARM="$S3_ARM"`, which makes the per-lane
assertion above a belt-and-braces check rather than the only line of defence.
Records banked before 2026-08-31 carry `arm=""` and are all `fitted` — it is the
only arm that had run while the field was empty.

**Three bugs found while building it, two of them in the arm work itself:**

1. *The arm was inherited from the submitting environment.* This is the identical
   failure the pass-1/2 chain had with its worklist: a chain re-submits itself for
   a day, one dropped `--export` loses the variable, and the fallback `fitted`
   finds the fitted arm already complete and **stands the control arm down looking
   finished**. The arm is now positional, carried explicitly through both
   re-submit sites (`queue_next` and the reshape path).
2. *`eval "$(arms.py)"` created shell variables, not exported ones.* The Python
   block in `preflight_stage3.sh` read `S3_OVERRIDE` as empty and printed the
   fitted arm's nine per-task thresholds under "gate in force" — on the control
   arm, where a flat 0.15 overrides all nine. A safety gate that prints the
   opposite of what will run is worse than no gate. `arms.py` now emits `export`.
3. *Pre-existing:* `stage3_state.py` imports the eval module, whose banner goes to
   stdout, and `eval "$(st)"` then tried to **execute** it — gen 1 logged
   `dedup_counting,: command not found` five times. Harmless only because that
   text happened to contain nothing dangerous. Now filtered to `^[A-Z_]+=`.

**Progress at the time of writing:** 1,313/2,700 banked on the fitted arm, 0 torn
records, `xRT_mean` 1.944 (against the 3.343 p95 prior the planner starts from),
~207 GPU-h left on this arm.


### 2.10 Arm 2 cancelled — what the study can and cannot claim now — 2026-09-01

**Decision:** stage 3 runs the `fitted` arm only. The `g015` control (flat global
0.15 on every task, ~184 GPU-h) is not run. Recorded here because it changes what
the paper is allowed to say, and because §2.9 registered two predictions that this
cancellation makes **permanently untestable**.

**The two predictions are withdrawn, not resolved.** §2.9 predicted that
ΔF1(fitted − g015) would come in below the in-sample +0.0075 and land inside the
0.03 noise band. Both required arm 2. They are struck from the register as
*unresolved by cancellation* — not as failed, and emphatically not as confirmed.
A prediction register only means anything if withdrawals are as visible as hits.

**What is lost.** The out-of-sample comparison of *nine fitted parameters against
one*. §2.7 measured it at **+0.0075, CI [−0.011, +0.027]**, but in sample on the
135 fitting videos, where the fitted arm has nine free parameters to the global
arm's one — the exact configuration in which an advantage is expected for free.
At n=2,700 the CI half-width would have fallen ~0.019 → ~0.004, enough to resolve
the sign. That measurement will not exist.

**What survives, and it is not nothing:**

- The absolute stage-3 numbers for the fitted gate on the complete benchmark
  (`OVERALL.json`, Tables 2–3), across all three audio strata.
- **The fit does not overfit.** `overall.py`'s fit-disjoint rescore — the same
  predictions with the 135 fitting ids removed, free — puts the in-sample bias at
  **+0.0007 gross**, two orders of magnitude inside the noise band. That is a
  measurement, not an assumption, and it is the one §6.1 was designed around.
- §2.5–2.8 are untouched: AUC ≈ 0.5, the flat precision curve, the bootstrap that
  no task survives, and pass 2's 9/9 confirmation. The negative result about
  per-task threshold fitting rests on those, not on arm 2.

**What the paper must NOT say.** No claim that per-task fitting beats — or loses
to — a single global threshold *out of sample*. The only fitted-vs-global number
is §2.7's, and every mention of it must carry the words *in sample, n=135, nine
free parameters against one*. Table 4 stays, with that caveat in its caption.

**It stays cheap to revisit.** The arm machinery is built, preflighted on both
arms, and `g015` writes to its own directory (`results/full2700_g015`), so
`sbatch bin/run_stage3.sbatch 1 600 g015` starts it whenever, and resume makes it
interruptible. Nothing about this decision is one-way.


## 3. Judge — offline, and selection on `time_f1`

Decided: **no in-loop judging.** Lanes run with `GEMINI_API_KEY` / `OPENAI_API_KEY`
unset, so `ContentJudge` returns `None` instantly, `content_acc` / `joint_f1` are
**WITHHELD** (never guessed), and predictions stay on disk for `judge_claude.py
export|ingest` afterwards.

Consequence, recorded as a deviation: §3 ranks cells by `joint_f1`; `lib/pick.py`
ranks by **`time_f1`**, which is judge-free and exact. `--metric joint_f1` switches
back once verdicts exist and re-running the picker is free. The reason is the same
one `omni_s3_eval` acted on: Gemini is unusable on this key
(`gemini-3-flash` 404 / `gemini-2.5-flash` 429 / `gemini-flash-latest` 503) and a
doomed network round-trip on the critical path of a 22-minute window is pure loss.
`bin/probe_judge.sh` re-checks the models when wanted.

## 4. Grid — one targeted widening

`THR_P1` is the spec's `[0.05 … 0.95]` verbatim, **plus `0.01` and `0.02` for
`semantic_condition_alert` and `sequential_step_instruction` only**.

Reason: replaying the finished phase-A traces through `resweep.py` (507,865 ticks,
2,699 samples — free, no GPU) put the optimum for exactly those two tasks at
**0.05, the bottom rail of the search range**. `GATE_TUNING.md` §3 rule 2: a winner
at the edge of the range means the grid is truncating the search. Widening
everywhere would have cost 18 extra cells to answer a question only two tasks are
asking.

Result: **94 cells, 376 work units, 1,410 sample-evals ≈ 145 GPU-h** — within 5 % of
the spec's 139 GPU-h for its 90 cells.

## 5. The offline screen — what to expect before spending any GPU time

`resweep.py` over the completed phase-A run, per task (fit-on-all | held-out at
dev=0.5). **This is a SCREEN, not a verdict** — offline replay assumes `p_hit` is
independent of the gate, and it weakly is not.

| task | shipped | screen-optimal thr | F1 fit-all | F1 held-out |
|---|--:|--:|--:|--:|
| cumulative_counting | 0.925 | 0.40 | 0.255 | 0.222 |
| dedup_counting | 0.992 | 0.50 | 0.220 | 0.192 |
| event_narration | 0.10 | 0.30 | 0.235 | 0.250 |
| explicit_target_grounding | 0.50 | 0.70 | 0.084 | 0.080 |
| instant_event_alert | 0.45 | 0.90 | 0.092 | 0.076 |
| realtime_state_monitor | 0.80 | 0.40 | 0.252 | 0.257 |
| semantic_condition_alert | 0.98 | **0.05 (rail)** | 0.197 | 0.215 |
| sequential_step_instruction | 0.01 | **0.05 (rail)** | 0.390 | 0.399 |
| snapshot_counting | 0.985 | 0.40 | 0.092 | 0.073 |
| *single global config* | — | 0.40 | 0.198 | — |

Two things this says, and both must reach the paper:

1. **The shipped (vision-only-fitted) thresholds are badly wrong for the omni
   model.** Five of nine move by more than 0.4.
2. **The F1 surface is flat and the argmax is unstable.** Fitting on all vs. on
   half picks wildly different thresholds — `instant_event_alert` 0.90 vs 0.20,
   `dedup_counting` 0.50 vs 0.20 — that land within a few points of each other in
   held-out F1. At n=300 per task. The on-GPU cells have **n=15**, so §6.2's warning
   is not hypothetical: `lib/pick.py` therefore refuses to manufacture a winner and
   reports **FLAT → keep shipped** for any task whose whole grid fits inside the
   0.02 tie-band, and flags any pick that lands on a rail.

## 6. Frozen subset

`bin/freeze_splits.py` imports the harness's own `load_samples` rather than
reimplementing the filter, then applies the id-sorted 1-in-20 stride. Written once
to `splits_thr_fit/<task>.off{0,1}.json`; every threshold and both passes reuse it.

**135 fitting samples (9 × 15), 469 GT events, 7.45 h of video, 65 % `audio_required`** —
matching the spec's ~65 % expectation in aggregate. Per task the mix is much more
uneven than the aggregate suggests (`dedup_counting` drew 0 `required`,
`instant_event_alert` drew 14 of 15): a caveat for per-task claims, not a defect of
the draw. `off1` is the disjoint validation slice; overlap is asserted empty.

## 7. Profiling and cluster gotchas found while bringing this up

| finding | detail |
|---|---|
| **login pid cap** | `/sys/fs/cgroup/user.slice/user-1371.slice/pids.max = 1000`, ~440 already used. torch sizes its OpenMP pool from the visible core count (48 login / 288 compute), so the first smoke run died with `libgomp: Thread creation failed`. Fixed by `OMP_NUM_THREADS=4` (+ MKL/OpenBLAS/numexpr) in `env.sh` — applies to **both** fleets, since 4 lanes × 288 would be 1,152 threads on a compute node. |
| **`/dev/shm` on ln003** | 89 % full (296 G of 334 G) from other users — a plausible `SIGBUS` source; a second cause of the same crash was two smoke runs colliding on one GPU. |
| **login-node throughput** | far below compute: a contended GH200 gave per-tick controller generation of 1.9 s rising to 22 s. The login fleet is best-effort only; treat the debug chain as the primary. |
| **weights** | 21 GB, 344 s to fetch on a login node with `HF_HUB_DISABLE_XET=1`. |
| **cold-start xRT prior** | `chain_state.py` uses p95 = 3.343, measured on this fork's own 2,700-sample run (mean 1.947). At that prior a 4-node/22-min generation can only take videos ≤ 233 s — 1,022 of the 1,410 sample-evals. The rest wait for the 2- and 1-node shapes. |
| **cost is NOT flat across the grid** | The spec's §7.1 budget assumes a constant 0.103 GPU-h per sample-eval. It is not constant: `hit_threshold` gates the *answer decode*, so a tick below threshold costs **one forward pass and zero decode steps**. Measured in the smoke run: `gen=0.3 s, ntok=0` on the quiet path at thr 0.95 versus `gen=2.9–6.4 s, ntok=35–105` when the threshold admits a decode. Low-threshold cells are therefore several times more expensive than high-threshold ones, and a flat per-cell estimate will overstate the total. Dedicated-GPU xRT at thr 0.05 measured **2.72**. |
| **the tail is where the 22-minute window hurts** | Measured on stage 3, gens 40–44. The band rule in `stage3_state.py` keeps a shape while **`k > 0`** — even one remaining sample fits — so at the tail 16 lanes stay bound to a 4-node window in which only 14 of 520 samples fit, and throughput fell from **~32 to ~10 samples/generation**. It self-corrects (once `k` reaches 0 the shape steps to 2 nodes, whose 679 s cap covers the whole tail), so this is a transient, not a stall — but the transient costs ~2 generations. The deeper cost is the **300 s model load paid per lane per generation**: 23 % of a 22-min window at 4 nodes, 11 % at 2, 5.7 % at 1. The tail distribution that provokes it: 520 remaining, min 266 s, median 361 s, max 594.6 s, 55.9 video-hours. |

| **hourly unattended push** | `omni_thr_fit/bin/autopush.sh --loop 3600` mirrors both live trees into the staging repo and pushes, one cycle an hour, detached (`setsid`, PPID 1, survives the session). It is mostly assertions, because nobody reads an unattended push: `bin/sync_stage.sh` re-checks after every mirror that no `.env`, no `run.log` and no `repo/` tree copy reached staging, and the cycle treats any failure as **fatal** — no add, no commit, no push. It also refuses a blob over 50 MB *before* committing, since GitHub rejects one *after*, leaving a commit that can never leave. It never uses `--force` and never rewrites history: if the remote has moved, the push fails, the cycle logs it and a human sorts it out. State lives in `logs/autopush.log`; a lock dir stops a manual `--once` from racing the loop. **No watchdog** — a login-node reboot silently ends it, so check the log's last line before trusting that the remote is current. |


### 7.1 The paper toolchain — neither half was present

`matplotlib` was not in the environment (nor in the original: §2.2), and there is
**no LaTeX on Clariden at all** — no `pdflatex`, `xelatex`, `latexmk` or
`tectonic` on any login node. §9's deliverable is a 12-page Springer LNCS PDF, so
both halves had to be built before step 15, not discovered at it.

| need | resolution |
|---|---|
| plotting | `matplotlib 3.11.1` pip-installed into our own env (§2.2) |
| LaTeX | **`tectonic` 0.17.0**, static musl `aarch64` binary, no root, at `/iopsstor/scratch/cscs/dthapa/tools/bin/tectonic` |
| `llncs.cls` | **already in tectonic's bundle** — no Springer download needed |

Verified end to end: a minimal `\documentclass{llncs}` document compiles to a
valid PDF-1.5. Tectonic fetches packages on demand and caches them
(`~/.cache/tectonic`, 41 MB warmed), so later builds do not re-download — which
also means the PDF build belongs on a **login node**, where there is network,
not on a compute node.

## 8. File map

```
omni_thr_fit/
  env.sh                    every path, thread cap, judge and account setting
  repo/                     isolated copy of system3_qwem_omni (the code under test)
  bin/fetch_weights.py      Qwen2.5-Omni-7B -> this account's HF cache (login only)
  bin/freeze_splits.py      the frozen 5% fitting + 5% validation draws
  bin/audit_gate_rail*.py   the p_hit-at-fire audit behind §2
  bin/test_fitted_gate.py   no-GPU unit test of edge/level/refractory
  bin/smoke.sh              §10 step 4: one task, two thresholds, end to end
  bin/run_chain.sbatch      debug fleet: 4 nodes, 22 min, self-chaining
  bin/srun_lane.sh          SLURM_PROCID -> lane id
  bin/login_chain.sh        ln003 fleet: 90-min generations, best-effort
  bin/worker.sh             ONE lane, shared verbatim by both fleets
  bin/heartbeat.sh          keeps a long unit's claim fresh
  lib/worklist.py           work units, the frozen-id completion test, atomic claims
  lib/claim_cli.py          shell-facing wrapper over the above
  lib/chain_state.py        progress + band selection (4 -> 2 -> 1 nodes)
  lib/score_cells.py        cell -> metrics, via metrics.py VERBATIM
  lib/pick.py               §3 selection rule, FLAT/RAIL guards, pass-2 grid
  splits_thr_fit/           the frozen draws + MANIFEST.json
  screen_phaseA/            symlinks to the finished run's logs, for the §5 screen
  results/<pass>/<task>/thr_<t>/lane<k>/online_pred.jsonl
```
