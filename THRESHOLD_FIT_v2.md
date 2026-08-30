# THRESHOLD FIT — per-task gate thresholds for Qwen2.5-Omni-7B on OmniPro Online

**Goal.** Fit the per-task firing threshold of the system-3 output gate for
**Qwen2.5-Omni-7B**, by two-pass grid search on a 5 % OmniPro subset drawn from
**all three audio classes**, then run the **COMPLETE 2,700-sample OmniPro Online
eval with the finalized per-task thresholds** and report the resulting time-F1 and
content/joint-F1. Deliverable: an **ECCV / Springer LNCS** formatted PDF.

Three stages, all on **4 nodes / 16 GPUs**, all chained until complete:

| stage | what | evals | samples | cost |
|---|---|--:|--:|--:|
| **1** pass 1 | 10 coarse thresholds x 9 tasks on the frozen 5 % | 90 | 1,350 | 139 GPU-h |
| **2** pass 2 | 5 refined thresholds x 9 tasks, same 5 % | 45 | 675 | 70 GPU-h |
| **3** final | **full OmniPro, all 2,700, finalized thresholds** | 1 | 2,700 | 279 GPU-h |
| | | **136** | **4,725** | **488 GPU-h** |

Stage 3 is **not optional** — it is the number that goes in the paper. The 5 % cells
choose the thresholds; only the full benchmark scores them.

**Fleet.** 4 nodes x 4 GH200 = **16 GPUs**, partition `debug`, self-chaining
until complete.
**Built on top of `system_3` rather than inventing something new entirely** — every
step below drives the existing `omniprofast` harness with environment variables;
the only new code is the launcher, the scorer and the PDF builder.

---

## 0. Paths — all verified to exist

| what | path |
|---|---|
| **Code under test** (system_3 extended to the omni model) | `/iopsstor/scratch/cscs/dbartaula/system3_qwem_omni` |
| pipeline (three-thread system) | `.../system3_qwem_omni/async_omni_v2` |
| eval harness | `.../system3_qwem_omni/omniprofast` |
| entry point per lane | `.../omniprofast/evaluate.py` |
| config = single source of truth | `.../async_omni_v2/config.py` |
| **Dataset** — OmniPro annotations (2,700 samples) | `/iopsstor/scratch/cscs/dbartaula/omnipro_data/benchmark.json` |
| **Dataset** — videos (1,262 files, 31 GB) | `/iopsstor/scratch/cscs/dbartaula/omnipro_data/raw_videos/` |
| **Eval reference** — original vision-only system_3 | `/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast` |
| prior fitting tools to read, not re-invent | `.../omniprofast/auc.py`, `grid.py`, `refit.py`, `resweep.py` |
| working chained launcher to copy patterns from | `/iopsstor/scratch/cscs/dbartaula/omni_s3_eval/{run_chain.sbatch,chain_state.py,worker.sh}` |
| LNCS PDF builder reference | `/iopsstor/scratch/cscs/dbartaula/system_3/build_pdf.py` |
| **This experiment's root** | `/iopsstor/scratch/cscs/dbartaula/omni_thr_fit` |

Model, confirmed from `config.py`: `backend=qwen2_5_omni`, `model_id=Qwen/Qwen2.5-Omni-7B`,
`gate_mode=controller`, `decode_mode=schema`, `fps=1.0`.

**Work in a git worktree of `system3_qwem_omni`, never in it.** The live
`omni_s3_eval` run reads that tree.

---

## 1. What is being fitted, and the seam that makes it possible

`system5_adapter.py:156` picks the threshold per sample:

```python
hit_threshold = self.base_cfg.task_hit_thresholds.get(sample.task, self.base_cfg.hit_threshold)
if os.environ.get("OMNIPRO_HIT_THRESHOLD"):
    hit_threshold = float(os.environ["OMNIPRO_HIT_THRESHOLD"])
```

`OMNIPRO_HIT_THRESHOLD` forces **one** value across all tasks — which becomes a
*per-task* knob the moment a run is restricted to a single task with
`--tasks <task>`. **No code change is needed to sweep thresholds.** That is the
whole mechanism.

Shipped thresholds (the incumbent to beat), from `config.py`:

| task | shipped thr | mode | refractory s |
|---|--:|---|--:|
| sequential_step_instruction | 0.01 | level | 7 |
| event_narration | 0.10 | level | 7 |
| instant_event_alert | 0.45 | edge | 600 |
| explicit_target_grounding | 0.50 | edge | 300 |
| realtime_state_monitor | 0.80 | level | 7 |
| cumulative_counting | 0.925 | edge | 7 |
| snapshot_counting | 0.985 | edge | 600 |
| dedup_counting | 0.992 | edge | 5 |
| semantic_condition_alert | 0.98 | level | 10 |

These were fitted on the **vision-only Qwen3-VL-8B** `p_hit` distribution. Refitting
them for the omni model is the point of this experiment.

### 1.1 READ THIS BEFORE FITTING — two defects that change what a fit means

**(a) `task_gate_modes` and `task_refractory_s` are never read.** Both dicts exist
in `config.py` and appear **nowhere else** in `async_omni_v2/` or `omniprofast/`
(`grep -rn task_refractory_s` returns only the definition). Only
`task_hit_thresholds` is consumed. `config.py` states the consequence itself:

> All three knobs are **ONE fitted config** — using the threshold without its
> mode/refractory does not reproduce it.

Measured evidence from the completed phase-A run (2,700 samples): `instant_event_alert`
is configured `edge` + **600 s** refractory, i.e. "fire once per video", yet emitted
**3,765 times for 231 GT events (16.3x)** across 187 videos — impossible if a 600 s
refractory were in force. Applying the refractory offline to that run's predictions
moved pooled time-F1 **0.2005 -> 0.2503** and cut emissions 21,232 -> 7,986.

**Decision required before pass 1.** Either
1. **(recommended)** wire the two dicts in `system5_adapter.py` beside line 156 —
   a three-line change mirroring the existing threshold lookup — and fit the
   threshold *within the gate as designed*; or
2. fit as-is and state in the paper that the fitted threshold belongs to a gate
   **with no refractory and a single global mode**, which is not the shipped design.

Option 1 is strongly preferred: without it the sweep will systematically favour
very high thresholds, because a high threshold is the only brake available when the
refractory is inert.

**(b) `L1_noF`-style dead knob.** `vision_stream.py:45` reads
`cfg.fps if cfg.deterministic else ctrl.get_fps()`, and `deterministic=True` in every
eval — so the plan's fps never reaches the encoder. Irrelevant to threshold fitting
(fps is fixed at 1.0 either way) but do not let it into a claim about self-pacing.

---

## 2. The subset — 5 %, all audio classes, frozen

`evaluate.py --subset_every 20` keeps 1 sample in 20, deterministic by sorted id
(`evaluate.py:89-92`) = **5 %**. Each task has exactly **300** eligible samples, so
each cell is **15 samples**.

```
--tasks <one task>  --audio all  --subset_every 20 --subset_off 0
```

`--audio all` maps to `{none, helpful, required}` (`dataset.py`), which satisfies the
requirement to fit on the **whole distribution, not the audio=none subset**. Class mix
per task is roughly 65 % required / 19 % helpful / 16 % none, so a 15-sample cell holds
about 10 required + 3 helpful + 2 none.

**Freeze the draw once** into `splits_thr_fit/<task>.json` and reuse for every
threshold and both passes. Pass 1 and pass 2 must see identical samples or the
comparison is meaningless.

> **The one interaction that can silently break this: `--subset_every` and
> `--max_dur` must be applied in that order.** `--subset_every` selects by *index
> stride* over the id-sorted list, so anything that shortens the list beforehand
> re-indexes the stride onto different samples. `--max_dur` (the window-fit cap of
> §7) used to be applied inside `load_samples`, i.e. **before** the stride — measured
> on the live `phaseB_pd25_seq` run, that put **487 of 688 evaluated ids outside the
> intended 25 % subset** and left **474 of the intended 675 never attempted**, biased
> hard toward short videos (evaluated 24-138 s, missing 100-595 s). Fixed 2026-08-28
> in both repos' `evaluate.py`: the cap now runs *after* the stride, so the subset is
> frozen by the stride and the cap only decides which of those frozen samples the
> current job shape can finish. Verify before launching — the check is one line and
> the failure is invisible in the metrics:
>
> ```bash
> # every cap must attempt a SUBSET of the same frozen id set
> for cap in 177 227 559 0; do grep -c '' /dev/null; done   # see §7 verify snippet
> ```
>
> A `DONE > TARGET` reading from `chain_state.py` is the symptom: a count exceeding
> its own target can only mean the loader and the scorer disagree about which samples
> exist.

**Hold out a disjoint 5 % for validation:** `--subset_off 1` gives a non-overlapping
1-in-20 slice. Needed for §6 — see the warning there about in-sample selection.

---

## 3. Pass 1 — coarse grid, 10 thresholds at 10 % spacing

```
THR_P1 = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
```

10 thresholds x 9 tasks = **90 evals**, each on that task's frozen 15 samples
= **1,350 sample-evals**.

Output layout, one directory per cell:

```
$ROOT/results/p1/<task>/thr_<thr>/lane<K>/online_pred.jsonl
```

Report per cell: `time_p`, `time_r`, `time_f1`, `content_acc`, `joint_f1`, `n_emit`,
`n_gt`, `emit/gt`, plus `xRT` and emissions per video.

**Selection rule — and it must not be a bare argmax.** Rank by `joint_f1`
(OmniPro Online is a joint metric, §5) with `time_f1` as tie-break. Then:

- if the top two are within **0.02 joint-F1**, treat them as tied and prefer the
  threshold whose `emit/gt` is closer to 1.0 — the one that is not winning by
  luck of emission volume;
- record **best** and **second-best** for every task regardless; pass 2 needs both.

---

## 4. Pass 2 — refine between best and second-best

For each task, with `b1` = best and `b2` = second-best from pass 1, take **5**
interior points splitting `[min(b1,b2), max(b1,b2)]` into 6 equal intervals:

```python
lo, hi = sorted((b1, b2))
THR_P2 = [round(lo + (hi - lo) * k / 6, 4) for k in (1, 2, 3, 4, 5)]
```

With the 0.10 pass-1 spacing this is a ~0.0167 grid. 5 x 9 = **45 evals**,
**675 sample-evals**. Same frozen 15 samples per task, same scoring, same tie rule.

The winner of `THR_P2 ∪ {b1, b2}` is the **final threshold** for that task. Include
`b1`/`b2` in the comparison — their scores are already measured, so re-running them
would be waste, but excluding them could pick a refined point that is worse than the
coarse one.

**Total: 135 evals, 2,025 sample-evals.**

---

## 5. Judge — Gemini, with a documented failure mode

Required: `OMNIPRO_JUDGE_BACKEND=gemini`, model via `GEMINI_MODEL`
(`metrics.py:290,305`). Only `event_narration` and `sequential_step_instruction`
reach the LLM judge; the other seven tasks are deterministic extraction.

**The blocker, quoted from `omni_s3_eval/README.md`:**

> Gemini is currently **unusable** on this key — `gemini-3-flash` 404 (the configured
> default), `gemini-2.5-flash` 429 quota-exhausted, `gemini-flash-latest` 503.

Do this before launching 135 evals:

```bash
export OMNIPRO_JUDGE_BACKEND=gemini
for m in gemini-3-flash gemini-2.5-flash gemini-flash-latest gemini-2.0-flash; do
  GEMINI_MODEL=$m $PY -c "from metrics import ContentJudge; j=ContentJudge(); print('$m', j.mode)"
done
```

Pin the **one** model that answers, export it in the sbatch (not `.env`), and assert
one judge across all 135 cells:

```bash
grep -h '^\[judge\]' $ROOT/results/*/*/thr_*/lane*/launch.log | sed 's/.*model=/model=/' | sort -u
# MUST print exactly one line
```

A content verdict is a function of *(triple, judge)*. Two judges across cells makes
every `joint_f1` comparison invalid, and `metrics.py:353-360` shares one cache
namespace between the Gemini and REST judges — so an auto-fallback silently serves
one model's verdicts as another's. **Never use `backend=auto`.**

**If Gemini cannot be made to answer**, the run is still viable but must be reported
honestly: score and select on **time-F1** only, mark `joint_f1` WITHHELD, and either
(a) fall back to `OMNIPRO_JUDGE_BACKEND=openai` with `OPENAI_JUDGE_MODEL` pinned, or
(b) use `judge_claude.py export|ingest`. State the judge actually used in the paper.

**Task-specific system prompts are already wired and cover all 9 tasks.**
`system5_adapter.py:142` selects `task_controller_prompts[sample.task]`, and
`:191` selects `task_writer_prompts[sample.task]` (present for
`explicit_target_grounding` and `realtime_state_monitor`, generic fallback elsewhere).
Nothing to add. Note `prompts.py:811 TASK_PROMPT_PARTS` is **dead data** — defined,
never imported; do not wire it in mid-experiment.

---

## 6. Stage 3 — the complete OmniPro eval with the finalized thresholds

**Mandatory, and on the same fleet: 4 nodes / 16 GPUs, `debug`, chained until
complete.** Once `FINAL_THRESHOLDS.json` exists, write those nine values into the
worktree's `async_omni_v2/config.py` `task_hit_thresholds` dict and run the whole
benchmark with **no `OMNIPRO_HIT_THRESHOLD` override** — so the run exercises the
per-task lookup exactly as a deployed system would:

```bash
# stage 3: no --tasks, no --subset_every, no threshold env override
$PY -u evaluate.py \
    --tasks instant_event_alert,semantic_condition_alert,explicit_target_grounding,\
snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,\
event_narration,sequential_step_instruction \
    --audio all \
    --benchmark_json $DD/benchmark.json --dataset_dir $DD \
    --shard $K --nshards 16 --resume \
    --out $ROOT/results/full2700/lane$K \
    --done_glob "$ROOT/results/full2700/lane*/online_pred.jsonl"
```

Sanity gate before launching stage 3: assert the config now holds the fitted values,
and that no stale override is exported.

```bash
$PY -c "import sys; sys.path.insert(0,'async_omni_v2'); from config import AsyncOmniConfig as C
import json; want=json.load(open('FINAL_THRESHOLDS.json'))
have=C().task_hit_thresholds
assert have==want, ('config does not match FINAL_THRESHOLDS', have, want); print('OK', have)"
[ -z "${OMNIPRO_HIT_THRESHOLD:-}" ] || { echo 'ABORT: override still set'; exit 1; }
```

**Shape.** 4 nodes / 16 lanes is the requested and default shape. It cannot be the
*only* shape: `debug-qos` gives 4 nodes a 22-minute window, and a 594.6 s video at
the measured omni p95 xRT of 3.34 needs ~33 min, so the long tail physically cannot
checkpoint there. Use the same band selection as §7 — 16 GPUs for everything that
fits, stepping to 8 then 4 GPUs only for the residual long videos. Phase A of
`omni_s3_eval` ran exactly this way: 16 GPUs cleared 1,193 of 2,700 samples, then 8,
then 4 for the last ~20. **Do not cap long videos out of stage 3** — the full
benchmark means all 2,700 samples.

### 6.1 Three numbers, from two sources, each labelled

| number | source | what it is |
|---|---|---|
| **in-sample ceiling** | the 9 winning 5 % cells | `max` over 15 thresholds **on the samples used to choose them** — optimistically biased. Label *in-sample / headroom*, exactly as `system_3/METHODOLOGY_FOR_PAPER.md` labels its `grid_best.json` figures |
| **HEADLINE — full OmniPro** | stage 3, all 2,700 | the paper number, directly comparable to the OmniPro Table 2 Online rows |
| **fit-disjoint** | stage 3, rescored excluding the 135 fitting ids | free — no extra compute, just a filtered rescore of the same predictions. The clean generalization estimate |

The fit-disjoint number matters because stage 3's 2,700 samples **contain** the 135
used for fitting, so the headline is ~5 % in-sample. Excluding those ids at scoring
time removes the contamination at zero cost, which is why no separate held-out run is
needed. Report both; if they differ by more than the §6.2 noise band, the fit
overfitted the 5 % cells.

Report for each: **time-F1** and **content/joint-F1**, pooled and macro-over-tasks,
across the three strata OmniPro defines — `audio_required`, `audio_not_required`,
`gross`. The strata are not three views of one result: 65 % of the benchmark is
`audio_dependency: required`, so `gross` is dominated by it.

For calibration, the reference points:

| | time-F1 | joint-F1 |
|---|--:|--:|
| this run, phase A, shipped thresholds, 2,700 samples | 0.2000 | WITHHELD |
| same, with refractory applied offline | 0.2503 | — |
| original vision-only system_3, 931 samples, `audio in {none,helpful}` | 0.268 | 0.118 |
| OmniPro paper, Online, MiniCPM-o 4.5 (9B) | — | **0.209** |
| OmniPro paper, Online, MMDuet2 / LiveStar | — | 0.113 / 0.036 |
| OmniPro paper, Probe, Qwen2.5-Omni-7B | — | 0.201 (accuracy) |

**Qwen2.5-Omni-7B has no published Online-mode row** — the paper excludes it because
Online "requires models with native streaming capability". Stage 3 produces the first
such number, which is the headline claim.

### 6.2 The statistical caveat that must be in the paper

**15 samples per cell is thin.** At ~2.4 GT events per sample that is ~36 scoring
events per cell. `system_3/METHODOLOGY_FOR_PAPER.md` measured **run-to-run variance
on one fixed config at F1 0.255 vs 0.051** and concluded most historical
"improvements" were inside the noise. A 0.02-0.03 gap between adjacent thresholds
here is **not** a real difference.

Mitigations, all cheap, all required:
- report `n`, `n_gt`, `n_emit` beside every cell — never a bare F1;
- use the 0.02 tie-band of §3 rather than a bare argmax;
- report the **regime**, not the constant, as the robust finding — `config.py`'s own
  conclusion was that one-shot tasks want high threshold + long refractory, dense
  tasks low + short, counting tasks very high + short;
- if a task's F1-vs-threshold curve is flat or non-monotonic across all 15 points,
  say so and keep the shipped value rather than manufacturing a winner;
- stage 3 is the guard against all of this: a threshold that won a 15-sample cell by
  noise will not hold up over 300 samples per task, and the comparison between the
  in-sample ceiling and the full number is what exposes it.

## 7. Fleet, work decomposition and the window-fit constraint

`debug` allows **MaxNodes=4**, 4 GPUs/node = **16 GPUs**, **MaxJobsPerUser=1** +
1 queued, and **90 NODE-MINUTES per job** — so 4 nodes buys a **22-minute** window.
Hence chaining. `srun` is mandatory: the batch body runs on the head node only.

**The window problem, and it is the same one that shaped `omni_s3_eval`.** Resume is
per-**sample**. OmniPro's longest video is 594.6 s; at the measured omni xRT
(mean 1.95, p95 3.34) the worst sample needs ~20-33 min and **cannot finish inside a
22-minute window** — it would be started, SIGKILLed and retried forever.

**Work unit = `(task, threshold, shard)` with `--nshards 4`.**
135 cells x 4 shards = **540 work units**; 16 lanes run 4 cells concurrently, ~4
samples per unit. Combined with `--resume` and
`--done_glob "$CELL/lane*/online_pred.jsonl"`, a unit that overruns its window
resumes in the next generation instead of restarting.

Reuse `omni_s3_eval/chain_state.py`'s **band selection**: it sizes the shape from the
measured p95 xRT and hands each shape only the samples that fit its window
(`--max_dur`), stepping 4 -> 2 -> 1 nodes as each band empties. **`--max_dur` must be
applied after `--subset_every`, never before — see the warning in §2**; this spec uses
both knobs together, so that ordering is a correctness precondition, not a detail. Do **not** simply cap
long videos out of the fit — that would bias the threshold toward short clips.
Let the band logic put them on the 1-node/88-minute shape instead.

### 7.1 Cost

```
per sample-eval    = 191 s mean video x 1.95 xRT      = 0.103 GPU-h
stage 1  pass 1      1,350 sample-evals               = 139 GPU-h  -> 10.9 wall-h
stage 2  pass 2        675 sample-evals               =  70 GPU-h  ->  5.5 wall-h
stage 3  FULL 2,700  2,700 sample-evals               = 279 GPU-h  -> 21.8 wall-h
                                              TOTAL   = 488 GPU-h  -> 38.2 wall-h
on 16 GPUs at ~80 % per-generation efficiency (22-min windows)
22-min jobs                                           = ~104 jobs
```
Budget **~46-48 hours elapsed**: every 22-min job reloads the backbone on 16 ranks
(~65-90 s, ~7 %), each chained job waits on the scheduler after its `afterany`
predecessor, and `MaxJobsPerUser=1` means nothing else of yours runs for two days.

Stage 3's 21.8 wall-h is the band-weighted figure, not a flat 16-GPU figure: only
~44 % of OmniPro fits a 22-minute window at 4 nodes, so the tail runs at 8 and then
4 GPUs (§6). Measured precedent: phase A of `omni_s3_eval` took ~31 wall-h for the
same 2,700 samples, but that included ~2.5 h lost to a filesystem outage and ~9 h
wasted before its band-selection defect was fixed.

**Cut order if the allocation is short:** stage 3 is the last thing to cut, because
without it there is no paper number. Cut pass 2 first (the coarse grid already gives
a threshold), then reduce pass 1 from 10 thresholds to 5.

### 7.2 Chain hardening — learned the hard way on 2026-08-27

Copy these from the working harness; each one cost a real incident:

1. **Queue the successor FIRST**, before any work, with `--dependency=afterany`.
   The 22-min kill is a SIGKILL: nothing at the end of the script runs.
2. **`queue_next` giving up is a silent death.** It retries 5 times over 50 s then
   stops permanently for that generation. During a filesystem stall every `sbatch`
   failed and a generation completed *childless* — the dead-chain guard cannot catch
   that, because it only fires on two consecutive **completed** zero-progress
   generations. Run an external guard (`~/omni_s3_guard.sh`) that keeps one successor
   queued and restarts the chain if the queue empties.
3. **Put the guard on `$HOME`, not `/iopsstor` and not `/tmp`.** `/iopsstor` flapped
   repeatedly (transport endpoint shutdown); `/tmp` is cleared on session teardown.
4. **Wrap every `squeue`/`sbatch`/read in `timeout`** — `sbatch` hung indefinitely
   while `squeue` stayed responsive.
5. **`MaxSubmitJobsPerUser=2` counts running + pending**, so cancel before
   submitting a replacement, never after.
6. **Dead-chain guard**: abort after two zero-progress generations rather than
   burning the allocation.
7. A `QOSMaxSubmitJobPerUserLimit` rejection means the job queued its own successor —
   not a failure.

### 7.3 Launcher sketch

```bash
cd $ROOT
$PY - <<'PY'
import json, itertools, os
TASKS = ["instant_event_alert","semantic_condition_alert","explicit_target_grounding",
         "snapshot_counting","cumulative_counting","dedup_counting",
         "realtime_state_monitor","event_narration","sequential_step_instruction"]
THR_P1 = [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]
NSHARD = 4
rows = [f"p1\t{t}\t{thr:.4f}\t{s}" for t, thr in itertools.product(TASKS, THR_P1)
        for s in range(NSHARD)]
open("worklist_p1.tsv","w").write("\n".join(rows) + "\n")
print(len(rows), "work units")          # expect 360
PY
```

Per lane (`worker.sh`), the only per-cell state is two env vars:

```bash
IFS=$'\t' read -r PASS TASK THR SH <<<"$(sed -n "${i}p" $WORK)"
CELL=$ROOT/results/$PASS/$TASK/thr_$THR
mkdir -p "$CELL/lane$SH"
CUDA_VISIBLE_DEVICES=$SLURM_LOCALID \
OMNIPRO_HIT_THRESHOLD=$THR \
OMNIPRO_JUDGE_BACKEND=gemini GEMINI_MODEL=$PINNED_JUDGE \
OMNIPRO_GATE_MODE=controller OMNIPRO_DECODE_MODE=schema \
PROSYNC_DIR=$REPO/async_omni_v2 OMNIPRO_BACKEND=qwen2_5_omni \
  $PY -u evaluate.py --tasks "$TASK" --audio all \
      --subset_every 20 --subset_off 0 \
      --benchmark_json $DD/benchmark.json --dataset_dir $DD \
      --shard "$SH" --nshards 4 --resume \
      --out "$CELL/lane$SH" --done_glob "$CELL/lane*/online_pred.jsonl"
```

`worker.sh` must **not** unset the judge keys the way `omni_s3_eval/worker.sh` does —
that harness deliberately disables in-loop judging; here the joint metric is the
selection criterion. Budget the judge latency inside the window, or judge offline
between passes and select on cached verdicts.

---

## 8. Scoring and the two-pass driver

```
$ROOT/score_cells.py   # cell -> {time_p,time_r,time_f1,content_acc,joint_f1,n_*,xRT}
                       # writes results/<pass>/CELLS.json + CELLS.csv
$ROOT/pick.py          # applies the §3 selection rule; pass 1 -> best + 2nd best
                       #   -> emits worklist_p2.tsv via the §4 formula
                       # pass 2 -> FINAL_THRESHOLDS.json
$ROOT/overall.py       # pools the 9 winning cells (in-sample ceiling) and the
                       # held-out cells; emits the 3-strata table of §6
```

Reuse `system_3/omniprofast/metrics.py` scoring verbatim — greedy 1-to-1 ±3 s
temporal match, then content-gated joint (`METHODOLOGY_FOR_PAPER.md §7.3`). Do not
reimplement the scorer; a second implementation is a second set of bugs.

The chain decides its pass from disk, exactly as `run_chain.sbatch` recomputes its
phase: if `results/p1/CELLS.json` is incomplete run pass 1; else if
`worklist_p2.tsv` is absent generate it; else run pass 2; else run held-out; else
build the PDF and exit.

---

## 9. Deliverable — 12-page ECCV / Springer LNCS PDF

Springer LNCS (`llncs.cls`), ECCV camera-ready style, **12 pages** excluding
references. Build with a script modelled on `system_3/build_pdf.py`. Figures are
vector **PDF** from matplotlib (never PNG — LNCS is print), embedded at final size so
no scaling occurs and font sizes stay true.

### 9.1 Page budget

| pp. | content |
|--:|---|
| 1 | Title, abstract, Fig. 0 teaser (final thresholds vs shipped, one bar pair per task) |
| 2 | Introduction — proactive streaming, "when to speak" as a gate, the claim |
| 3 | Related work — OmniPro, MiniCPM-o 4.5 / MMDuet2 / LiveStar, threshold-free vs gated |
| 4 | Method — the three-thread system, where `p_hit` comes from, what the gate does |
| 5 | Protocol — 5 % all-audio subset, two-pass grid, judge, ±3 s joint matching |
| 6 | **Table 1** (final thresholds) + **Fig. 1** (F1 vs threshold, 3x3 small multiples) |
| 7 | **Fig. 2** (ROC + AUC per task, 3x3) + AUC-vs-F1 discussion |
| 8 | **Fig. 3** (per-sample precision/recall mean ± s.d. bands vs threshold, 3x3) |
| 9 | **Table 2** (complete 2,700 eval, 3 strata) + **Table 3** (per-task complete eval) |
| 10 | **Fig. 4** (emission calibration, emit/gt vs threshold) + **Fig. 5** (pass-1 -> pass-2 refinement) |
| 11 | Discussion + **Limitations** (all of §1.1 and §6.2) |
| 12 | Conclusion, broader impact, references begin |

### 9.2 Tables

**Table 1 — Finalized per-task thresholds.** One row per task, nine rows:
`task | shipped thr | pass-1 best | pass-1 2nd | pass-2 grid | FINAL thr | Δ vs shipped |
time-F1 | content-acc | joint-F1 | n | n_gt | n_emit | emit/gt`.
Bold the FINAL column. Never a bare F1 — `n`, `n_gt`, `n_emit` travel with every
number (§6.2).

**Table 2 — Complete OmniPro Online, 2,700 samples.** Rows = `gross`,
`audio_not_required`, `audio_required`; columns = `n | n_gt | n_emit | time-P |
time-R | time-F1 | content-acc | joint-F1`. Three blocks: **(a) full 2,700 with final
thresholds (the claim)**, (b) 5 % in-sample ceiling, (c) fit-disjoint rescore. Then a
baseline block: MiniCPM-o 4.5 20.9, MMDuet2 11.3, LiveStar 3.6, and Qwen2.5-Omni
Probe 20.1 — each footnoted with *different subset / different scoring*, per
`METHODOLOGY_FOR_PAPER.md §7.1`.

**Table 3 — Per-task complete eval.** The nine tasks at their final thresholds on all
2,700, with the shipped-threshold column beside them so the delta from fitting is
visible per task.

**Table 4 (compact) — Ablation of the fit.** Global 0.5 / best single global / shipped
per-task / fitted per-task, pooled time-F1 and joint-F1. Mirrors the progression
`METHODOLOGY_FOR_PAPER.md` already reports for the vision-only model
(0.190 -> 0.254 -> 0.316), making the omni result readable against it.

### 9.3 Figures

**Fig. 0 — Teaser.** Horizontal paired bars, one pair per task: shipped vs final
threshold, with the joint-F1 delta annotated. Sorted by |delta| so the biggest moves
read first. Single panel, wide, top of page 1.

**Fig. 1 — F1 vs threshold (3x3 small multiples).** x = threshold (0-1), y = F1.
Three series per panel: **time-F1**, **joint-F1**, and a faint **shipped-threshold
vertical rule**. Pass-1 points as markers, pass-2 points denser in the refined
interval, connected. Mark the chosen threshold with a filled marker + direct label.
**Nine tasks means nine panels, not nine colours on one axis** — a 9-series
categorical palette is not resolvable and the method forbids inventing a 9th hue.

**Fig. 2 — ROC and AUC per task (3x3 small multiples).** ROC of per-tick `p_hit`
against the ±3 s positive label — exactly what `omniprofast/auc.py` already computes;
reuse it rather than reimplementing. Plot the curve, the chance diagonal (dotted,
muted), the operating point of the final threshold (filled marker), and the shipped
operating point (hollow marker). Annotate **AUC ± bootstrap 95 % CI** in-panel.
This figure carries the paper's most defensible claim: `METHODOLOGY_FOR_PAPER.md`
argues **AUC measures whether perception is right, F1 measures whether the gate is
tuned** — two systems can be indistinguishable in F1 and separable in AUC. State
that AUC is threshold-free and therefore unaffected by this fitting, so it is the
honest measure of the backbone while F1 is the measure of the gate.

**Fig. 3 — Per-sample precision / recall distribution vs threshold (3x3).** For each
(task, threshold) compute precision and recall **per sample**, then plot mean with a
**± 1 s.d. band** across the task's 15 samples: precision and recall as two series,
bands semi-transparent, threshold on x.

> **Honesty requirement, and it must appear in the caption.** Per-sample precision and
> recall are bounded on [0,1] and, at ~2.4 GT events per sample, are frequently exactly
> 0 or 1 — so the distribution is **not** Normal and a mean ± s.d. band will extend
> outside [0,1]. Clip the band to [0,1] and **overlay the actual per-sample values as a
> jittered strip** (or a violin) so the reader sees the true spread rather than a
> Gaussian that does not exist. Report the s.d. as a dispersion statistic, not as a
> confidence interval; if a CI is wanted use a bootstrap over samples.

**Fig. 4 — Emission calibration.** x = threshold, y = `emit/gt` on a log axis, one
line per task (here nine lines *is* defensible: it is one series per panel-free
magnitude comparison, so use a 3x3 small-multiple again, or a single panel with the
nine as faint grey lines and 2-3 highlighted). Horizontal reference line at
`emit/gt = 1`. This is the figure that shows the two failure modes directly:
`instant_event_alert` at 16.3x (fires constantly) versus `dedup_counting` at 0.03x
(never fires), and where each crosses 1.

**Fig. 5 — Refinement.** For each task a narrow horizontal panel showing the pass-1
grid (10 open markers), the two selected points, the pass-2 interval shaded, the 5
refined points (filled), and the final choice. Makes the two-pass procedure legible
at a glance and shows honestly where the curve was flat.

### 9.4 Chart specification — ECCV grade

**Colour, computed not eyeballed.** Three-series palette, from the validated
categorical order, checked for CVD separation in OKLab (ΔE ×100) and for contrast on
white:

| role | hex | vs white | grayscale luminance |
|---|---|--:|--:|
| series 1 — time-F1 / precision | `#4a3aa7` violet | 8.56 | 0.073 |
| series 2 — joint-F1 / recall | `#2a78d6` blue | 4.42 | 0.187 |
| series 3 — third series if needed | `#eb6834` orange | 3.20 | 0.278 |

Pairwise CVD separation (all pass the ΔE ≥ 8 target; ΔE ≥ 15 normal-vision floor):

```
violet vs blue    normal 16.3  deut 13.0  prot 15.9  trit 17.3   PASS
violet vs orange  normal 37.6  deut 38.2  prot 29.4  trit 37.2   PASS
blue   vs orange  normal 33.6  deut 31.6  prot 24.9  trit 32.7   PASS
```

This ordering is deliberate: the three luminances (0.073 / 0.187 / 0.278) form a
monotone ladder, so the figures **survive grayscale printing** — a real risk for a
camera-ready. The default slot-3 aqua `#1baf7a` was rejected: it passes CVD but sits
at contrast 2.82 on white and luminance 0.322, too close to orange in grayscale.

**Identity is never colour alone.** Every series also carries a linestyle
(solid / dashed / dotted) and a marker shape (circle / square / triangle). A
colourblind reader, a grayscale printer, and a projector all resolve the figure.

**Marks and anatomy.**
- 1.5-2 pt lines; markers ≥ 4 pt (≥ 8 px equivalent); no marker on every point where
  points are dense — mark the grid points only.
- Recessive axes and grid: grid `#e8e8e6` hairline, axis `#52514e`, **no top/right
  spines**, no background fill, no gridlines on the ROC diagonal panel.
- Text in ink tokens, never in the series colour: `#0b0b0b` primary, `#52514e`
  secondary. Values and labels are text, not decoration.
- **One axis. Never a dual y-axis** — F1 and emit/gt are different scales and get
  different figures, which is why Fig. 4 exists separately from Fig. 1.
- Sequential magnitude (if a heatmap is added) = one hue light→dark, never a rainbow;
  diverging (delta vs shipped) = two hues with a **neutral grey** midpoint.
- 2 pt surface gap between adjacent fills; a 2 pt white ring where markers overlap.
- Direct-label selectively — the chosen threshold, the endpoints, the AUC. Never a
  number on every point.
- Legend present in every multi-series panel; in a 3x3 grid put **one** shared legend
  outside the axes (figure-level), not nine copies.

**Typography and sizing.** LNCS body is 10 pt Times; figure text must not be smaller
than **8 pt** after placement. Author each figure at its final column width
(single column 3.3 in, full width 6.9 in) with `savefig(..., bbox_inches='tight',
pad_inches=0.02)` and no post-hoc scaling in LaTeX. Set matplotlib to
`pdf.fonttype: 42` so fonts embed as TrueType rather than Type-3 (Type-3 fails many
camera-ready checks).

```python
mpl.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "grid.color": "#e8e8e6", "grid.linewidth": 0.5,
    "axes.grid": True, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "lines.markersize": 4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
```

**Every caption states `n`.** A 15-sample panel and a 300-sample panel must not look
alike; put the sample count and GT-event count in the caption, and mark any panel
whose cell was flagged `reliable=False` with a hatched background.

**Final pass: render and look at it.** The colour checks are computable and were
computed; layout is not. Open every figure at 100 % and at print size and check for
label collisions, tick overlap, and clipped legends before the PDF is called done.

---

## 10. Execution order

```
0.  Worktree of system3_qwem_omni; copy .env                       §0
1.  DECIDE §1.1(a): wire task_gate_modes + task_refractory_s, or
    document that the fit is for a no-refractory gate               §1.1
2.  Pin the judge: probe Gemini models, export the one that answers §5
3.  Freeze splits_thr_fit/<task>.json  (subset_off 0 and 1)         §2
4.  Smoke: 1 task x 2 thresholds x 2 samples, end to end            —
5.  Generate worklist_p1.tsv (360 units)                            §7.3
6.  sbatch the chain (4 nodes / 22 min), guard armed                §7.2
7.  score_cells.py + pick.py -> best/2nd per task                   §8
8.  Generate worklist_p2.tsv (180 units); chain continues           §4
9.  pick.py -> FINAL_THRESHOLDS.json                                §8
10. Write the 9 values into worktree async_omni_v2/config.py and
    assert config == FINAL_THRESHOLDS; assert no env override        §6
11. STAGE 3: full 2,700 eval, 4 nodes / 16 GPUs, chained, band
    step-down for the long tail, --nshards 16, per-sample resume     §6
12. overall.py -> 3-strata tables x {in-sample, full, fit-disjoint}  §6.1
13. auc.py -> per-task ROC/AUC + bootstrap CIs                       §9.3
14. figs.py -> Fig 0-5 as vector PDF at final column width           §9.4
15. build_pdf.py -> 12-page ECCV/LNCS PDF; render-and-look pass       §9
```

Step 4 is not optional: a smoke test that exercises one full cell end to end costs
minutes and catches a wrong env var before it is baked into 540 work units.
