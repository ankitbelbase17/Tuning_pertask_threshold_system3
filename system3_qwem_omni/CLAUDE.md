# CLAUDE.md — orientation for this repo

`system3_qwem_omni` is a fork of **system_3** that swaps the vision-only
Qwen3-VL-8B backbone for **Qwen2.5-Omni-7B** (vision **+ audio**). The system is a
three-thread proactive streaming pipeline; the research question is *when should a
streaming model speak*, and the benchmark is **OmniPro Online** (2,700 samples).

> There was no CLAUDE.md before 2026-08-28. This file is an index, not a manual:
> each section points at the document that already owns the subject.

---

## 1. What the system is

Three threads over one shared KV cache (`async_omni_v2/`):

| thread | file | job |
|---|---|---|
| encoder / ingester | `vision_stream.py`, `input_ingester.py`, `audio_io.py` | decode video at 1 fps + audio at 2 s chunks, append to the cache |
| controller | `controller.py` | every tick, read `p_hit = P(true)` off the logits and decide whether to speak |
| writer | `writer.py` | produce the utterance |

`async_omni_v2/config.py` is the **single source of truth** for behaviour. The eval
injects only per-video data (the task instruction and the video path).

- Architecture in depth → `SYSTEM3_TECHNICAL_ARCHITECTURE.md`
- The omni fork specifically → `FORK_QWEN25OMNI.md`, `OMNI_EXTENSION.md`
- Why Qwen2.5-Omni → `OMNI_MODEL_SURVEY.md`, `OMNI_FEASIBILITY.md`
- Running history and current goals → `MISSION.md`, `ROADMAP.md`

## 2. The output gate — read this before touching thresholds

The gate is **three coupled knobs**, never one: `hit_threshold`, `mode`
(`edge`/`level`), `refractory_s`. `config.py` says it itself: *using the threshold
without its mode/refractory does not reproduce it.*

**→ `omniprofast/GATE_TUNING.md` is mandatory reading before changing
`task_hit_thresholds` / `task_gate_modes` / `task_refractory_s` or claiming any
gate result.** It carries the non-negotiable rules (always compare held-out; watch
for rail-pinning; compare families at equal budget; offline replay is a SCREEN, not
a verdict).

**Known defect, measured 2026-08-28 and now fixed only in the threshold-fit tree:**
under the shipped `gate_strategy="hysteresis"` the fire test is
`p_hit >= cfg.gate_high_thr`, a fixed global **0.5** — so `hit_threshold` merely
decided whether an answer was decoded. Across the completed 2,700-sample phase-A
run (581,403 ticks, 50,433 fires) **not one fire occurred below p_hit 0.500**, so
the four tasks configured at or under 0.5 were all effectively running at 0.5 and
their fitted values were inert. `task_gate_modes` and `task_refractory_s` were read
nowhere at all. Details and the fix → `THRESHOLD_FIT_RUNBOOK.md` §2.

**Second defect, found live 2026-08-28 while pass 1 was running:** `p_hit` is high
on the **first tick** of every video — the model answers the boolean from the
prompt before it has seen anything (≥ 0.05 for 98–100 % of phase-A samples in
every task, median 0.34–0.64). Seeding `prev_above = False` made that tick a
rising edge, and for a task with a long refractory the spurious fire consumed the
video's only emission: every `instant_event_alert` sample emitted once, at vid
1.0 s, against ground truth at 26–98 s. Fixed definitionally — an edge requires a
predecessor, so `prev_above` starts `None`. Offline best time-F1 rose 0.0535 →
0.0900 (`snapshot_counting`) and 0.0636 → 0.0751 (`instant_event_alert`); the
short-refractory tasks moved ±0.001. **The size of the effect tracks the
refractory**, which is the practical reason the three knobs cannot be fitted
separately. Applied to the live gate *and* to all five offline replays. Details →
`THRESHOLD_FIT_RUNBOOK.md` §2.1, evidence → `omni_thr_fit/FIRST_TICK_AUDIT.json`.

**Third finding, and the important one, measured 2026-08-29:** `p_hit` does not
rank event-adjacent ticks above quiet ones. On the five pass-1 tasks with a
complete reference cell, **AUC is 0.431–0.551 and every 95 % CI contains 0.5.**
It is not a clock offset (swept ±10 s: no peak), not too tight a tolerance (±10 s:
unchanged), and not a degenerate score (232–284 distinct `p_hit` values spanning
0.000–0.997). Independently, per-task precision is FLAT across the whole
threshold grid while recall falls — firing less often buys no precision, which is
what an AUC of 0.5 predicts. Expect the fit to pin to the **low rail** on every
task that emits. This is a reportable negative result about per-task threshold
fitting, not a bug to tune around: no threshold recovers information the score
does not carry. Details → `THRESHOLD_FIT_RUNBOOK.md` §2.5, evidence →
`omni_thr_fit/PERCEPTION_AUDIT.json`, `omni_thr_fit/bin/audit_perception.py`,
figures → `omni_thr_fit/figs/fig2_roc.pdf`, `fig3_pr_bands.pdf`.

**Fourth finding — the fit does not survive its own bootstrap, 2026-08-30.** Pass 1
completed (1410/1410, 94/94 cells, all `reliable`) and `pick.py` returned a winner
for all nine tasks. A *paired* bootstrap over the frozen videos — same video set
across thresholds, replaying `pick.rank` itself rather than a bare argmax — shows
**every task's ΔF1 against its best rival has a 95 % CI containing zero**, and the
fitted threshold is re-selected only 25–61 % of the time (`semantic_condition_alert`
just **6 %**, *below* its 8 % chance rate). Two "fits" are artefacts:
`instant_event_alert` scores 0.000 at every threshold (`pick.py` correctly flags it
FLAT and keeps the shipped value), and `explicit_target_grounding`'s entire curve is
one matched event or zero out of 16. The only reproducible structure is the collapse
at 0.95 — the threshold controls emission *volume*, not *which moments* to emit at,
which is what §2.5's AUC ≈ 0.5 predicts via a separate code path. §2.5's advance
prediction of low-rail pinning was **wrong** (picks scattered mid-grid instead); the
correction and a new falsifiable prediction for stage 3 are recorded in
`THRESHOLD_FIT_RUNBOOK.md` §2.6. Evidence → `omni_thr_fit/FIT_NOISE_AUDIT.json`,
`omni_thr_fit/bin/audit_fit_noise.py`.

**Fifth finding — the gate has ONE identifiable degree of freedom, not nine,
2026-08-30.** `lib/ablation.py` (Table 4, paired bootstrap over the same videos)
on the pass-1 cells: `global_0.5` 0.2271, **`best_single_global` (0.15) 0.2469**,
`shipped_per_task` 0.2115, `fitted_per_task` 0.2544. The nine fitted thresholds beat
the single global one by **+0.0075, CI [−0.011, +0.027]** — straddling zero *in
sample*, where they have nine free parameters to its one. Yet the global threshold
itself is solid: re-running the sweep inside every draw, 0.15 is re-selected in
**94 %** of them (chance 10 %) against 25–61 % for the per-task fits. Pooled over
135 videos the operating point is identified; split nine ways over 15 videos each it
is not — a power problem, not a claim the tasks are identical. `shipped_per_task` is
significantly *worse* than flat 0.15 (−0.0354), on recall (0.356 vs 0.601). Caveat:
7/9 shipped values and **all 9** of the `global_0.5` cells are off-grid (0.5 is served
by 0.45); every substitution is recorded in `ABLATION.json`. Details →
`THRESHOLD_FIT_RUNBOOK.md` §2.7.

**Sixth finding — pass 2 complete, and it confirms a prediction registered in
advance, 2026-08-30.** Pass 2 finished 675/675, 45/45 cells, all `reliable`;
`FINAL_THRESHOLDS.json` exists. §2.6 had predicted, in writing and before pass 2
ended, that the refined grid's ΔF1-vs-rival CIs would contain zero on **at least
seven of nine** tasks. **The outcome is 9 of 9.** Across each refined
neighbourhood the whole F1 span is 0.021–0.133, and on four tasks *every* cell is
within the 0.03 noise band of the best: refining did not resolve a winner, it
confirmed there was no gradient to resolve. Two rows report CI [+0.000, +0.200] —
the lower bound is **exactly zero, so they do not exclude it**; those tasks match
0–2 events out of 15–16 and the interval is grid-quantised, not a tight effect.
Re-selection reads 38–74 % vs pass 1's 25–61 %, but chance is 20 % here (5 cells)
against 8–10 % there, so as a multiple of chance pass 2 is **no better** — always
divide by chance before comparing across grids. Only §2.6's second prediction
(stage 3 within 0.03 of shipped) is still open. Details →
`THRESHOLD_FIT_RUNBOOK.md` §2.8, evidence → `omni_thr_fit/FIT_NOISE_AUDIT_P2.json`.

## 3. Evaluation

`omniprofast/` is a self-contained OmniPro harness that spins up the **real**
pipeline unmodified and captures emissions. It does not reimplement the system.

- What it is and how to point it at a repo → `omniprofast/README.md`, `omniprofast/HANDOFF.md`
- The metric definitions and the honesty rules → `EVAL_PROTOCOL.md`
- Offline gate search tools → `omniprofast/{sweep,resweep,auc,refit,grid}.py`
- Scoring lives in `omniprofast/metrics.py`. **Do not reimplement it** — a second
  implementation is a second set of bugs.

Metrics: `time_f1` is judge-free (±3 s greedy 1-to-1 match). `content_acc` and
`joint_f1` need an LLM judge and are **WITHHELD, never guessed**, when it is
unreachable.

## 4. The active experiment — per-task threshold fit for the omni model

Spec: **`THRESHOLD_FIT_v2.md`**. Two-pass grid search on a frozen 5 % all-audio
subset, then the complete 2,700-sample eval with the fitted thresholds; deliverable
is a 12-page ECCV/LNCS PDF.

Everything about *running* it — the environment, the two fleets, the deviations
from the spec and why, the profiling numbers — lives in
**`THRESHOLD_FIT_RUNBOOK.md`**, and the work itself is in a separate tree so this
repo is never mutated by a run:

```
/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit/
```

## 5. Cluster facts you will otherwise rediscover the hard way

Verified on Clariden 2026-08-28 (`sacctmgr`, `scontrol`, `sinfo`):

| fact | value |
|---|---|
| login nodes | 4 (`clariden-ln001..004`), 4× GH200 each, **shared** with all users |
| login-node pid cap | **1000 per user** (cgroup) — cap `OMP_NUM_THREADS` or torch dies with `libgomp: Thread creation failed` |
| `debug` partition | MaxNodes **4**, MaxTime 01:30:00 |
| `debug-qos` | **MaxTRESMins node=90**, MaxTRES node=4, **MaxJobsPU=1**, MaxSubmitPU=2 |
| ⇒ shapes | 4 nodes = 16 GPUs / **22 min**; 2 = 8 / 45 min; 1 = 4 / 90 min |
| `normal` partition | 12 h, unlimited nodes |
| accounts | **`a168` carries QoS `stop` — its jobs never start.** Use **`a0264`**. |

**Own your runtime.** On 2026-08-28 the other account began a `pip` reinstall of
torch inside the shared `prosync_env` *while a sweep was running against it*: 594
files changed, `torch/bin/torch_shm_manager` vanished, and `import torch` failed
outright on every lane. Anything under another account's scratch — env, dataset,
weights — can move without warning, and the dataset case is worse than the env
case because it fails **silently**. The threshold-fit tree now runs entirely on
`/iopsstor/scratch/cscs/dthapa/{envs/prosync_env, omnipro_data, hf_home}`.
Recipe and evidence → `THRESHOLD_FIT_RUNBOOK.md` §2.2, `omni_thr_fit/bin/repair_env.sh`.

The 22-minute window is why every long run here is a **self-chaining** job that
pre-queues its successor with `--dependency=afterany` *before* doing any work: the
wall-clock kill is a SIGKILL, so nothing at the end of the script is guaranteed to
run. Chain-hardening lessons → `THRESHOLD_FIT_v2.md` §7.2.

## 6. Ground rules that came from real incidents

- **`--subset_every` before `--max_dur`.** The stride selects by *index*, so any
  filter that shortens the list first re-indexes the subset onto different
  samples. Measured: 487 of 688 evaluated ids landed outside the intended subset.
- **Resume is global, not per-lane** (`--done_glob`), because the chain reshapes
  4→2→1 nodes and the lane count changes the shard assignment.
- **The eval is not bit-reproducible.** Treat < 0.01 F1 as noise; `METHODOLOGY`
  measured run-to-run variance of F1 0.255 vs 0.051 on one fixed config.
- **Never `OMNIPRO_JUDGE_BACKEND=auto`** — `metrics.py` shares one cache namespace
  between judges, so a fallback silently serves one model's verdicts as another's.
- Post-mortems worth reading before repeating an experiment: `LEARNINGS.md`,
  `AUC_DIAGNOSIS.md`, `CONTROLLER_DIAGNOSIS.md`, `PROMPT_AUDIT.md`.
