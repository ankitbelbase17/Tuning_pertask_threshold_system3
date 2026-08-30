# CLUSTER_SCALING — how to actually get 16 GPUs on this cluster

**Written 2026-07-31.** Everything here was verified by *running it*, not by reading
documentation. Companion to `LEARNINGS.md`.

---

## 1. The limits that actually bind

```
partition debug    MaxNodes=4        MaxTime=01:30:00     (4 GPUs per node)
debug-qos          MaxSubmitJobsPerUser=2   MaxJobsPerUser=1   MaxTRESMins node=90
normal             12h, unlimited nodes  -- NO ACCESS for this account
```

Read `MaxTRESMins node=90` carefully, because it is the one that surprises people:

**A job may consume at most 90 NODE-MINUTES, however you slice it.**

| shape | node-minutes | verdict |
|---|---|---|
| 1 node × 90 min | 90 | OK |
| 2 nodes × 45 min | 90 | OK |
| 4 nodes × 22 min | 88 | OK |
| **4 nodes × 90 min** | **360** | **REJECTED** |

A rejected job does not error at submit time. It sits `PENDING` forever with reason
`QOSMaxNodeMinutesPerJob`. We lost time to exactly this.

**Consequence: per job the compute is fixed at ~6 GPU-hours regardless of shape.**
4 GPUs × 90 min and 16 GPUs × 22.5 min are the same budget.

**So why is 16 GPUs still 4× faster?** Because `MaxJobsPerUser=1` — jobs run
back-to-back, so burning the same 6 GPU-hours in 22 minutes instead of 90 gives four
times the throughput per wall-hour. The win is in the *rate*, not the *budget*.

Measured effect on the remaining 425 samples / 19.9 video-hours ≈ 51 GPU-hours:
**~17 h at 4 GPUs → ~3.5–4 h at 16 GPUs.**

---

## 2. `srun` is mandatory — a bash loop silently wastes 3 of 4 nodes

Proof, from test job `2968290` (`--nodes=4 --ntasks-per-node=1 --gres=gpu:4`):

```
2968290        nodetest  COMPLETED 0:0  AllocNodes=4  gres/gpu=16
2968290.batch  batch     COMPLETED 0:0  node=1        gres/gpu=4     <-- batch script
2968290.0      bash      COMPLETED 0:0  node=4        gres/gpu=16    <-- srun step
```

**The batch script body runs on the head node only.** The old launcher did:

```bash
for k in 0 1 2 3; do ( CUDA_VISIBLE_DEVICES=$k bash run_fast.sh ... ) & done
```

With `--nodes=4` that allocates (and bills) four nodes while three sit idle. Work must
go through `srun`:

```bash
srun --ntasks=16 --ntasks-per-node=4 bash -c '
  G=$(printf "r%02d" $SLURM_PROCID)      # PROCID picks the shard
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID   # LOCALID picks the GPU on this node
  ...'
```

---

## 3. Chaining, because 90 node-minutes is never enough

The established pattern (survives SIGKILL, verified across 16+ generations):

- `#SBATCH --signal=B:USR1@120` + `trap on_timeout USR1` — get warned before the wall.
- **Pre-submit the successor at job START**, not in tail code:
  `sbatch --dependency=afterany:$SLURM_JOB_ID ...`. `afterany` fires on *any* terminal
  state, so TIMEOUT/CANCELLED/FAILED all keep the chain alive.
- A plain `wait` is interrupted by the USR1 trap and falls through with lanes still
  running. Loop instead: `while kill -0 $PID 2>/dev/null; do wait $PID || true; done`.
- 1 running + 1 pending is exactly `MaxSubmitJobsPerUser=2`, so **a third job cannot be
  submitted while the chain is alive.** Anything extra must be triggered *by* the chain.

Why pre-submit rather than resubmit at the end: generation 1 of the earlier 4-GPU run
(job 2932056) put the resubmit *after* `wait`, which never returns inside the wall clock,
so the tail was unreachable and the chain died after one link.

---

## 4. Sharding — two mistakes that cost whole generations

Both were made and fixed on 2026-07-31. `reshard_remaining.py` now handles both.

### 4a. Reshard the REMAINING work, not the original corpus

The original split was 16-way over all 932 samples. Once 8 of those shards finished,
re-running that split on 16 GPUs would have left **half the workers idle**. Shards must
be rebuilt from what is actually undone — read from `online_pred.jsonl`, the same source
`evaluate.py --resume` uses, so the two can never disagree.

### 4b. Order samples SHORTEST-FIRST inside a shard

Duration-balancing packs longest-first. Writing shards in that order means every shard
*opens with its own worst case*:

- `evaluate.py` walks a shard in file order and resumes per sample.
- A sample killed by the wall clock banks **nothing**.
- Generation 1 of the 16-GPU run banked **6 samples in 22 minutes across 16 GPUs** —
  essentially a write-off, because all 16 workers were grinding their longest video.

Shortest-first means each job completes as many whole samples as it can and only the
single in-flight sample is ever lost.

### 4c. Quarantine samples that can never finish

At 2.56 wall-seconds per video-second, a 22-minute job (minus ~60 s model load) can only
finish videos up to **~492 s**. Longer ones loop forever — and because `evaluate.py`
walks in file order, **one unfinishable sample at the head of a shard blocks every sample
behind it, permanently.**

11 of 425 remaining samples (495–526 s) are quarantined to `splits_r2/toolong.json` and
run separately as **2 nodes × 45 min** (= 90 node-minutes, 8 GPUs), which affords ~2,640 s
of compute — enough for a 526 s video at 2.56×.

---

## 5. Throughput reference

| | |
|---|---|
| wall per second of video | **2.56 s** (0.39× realtime per GPU) |
| of which controller `gen` | 94% |
| per sample (median 150 s video) | ~250 s |
| per-generation efficiency, 4 GPU × 90 min | ~74% (model reload + lost in-flight sample) |
| full 932-sample corpus | ~49.6 video-hours ≈ 127 GPU-hours |

Model load is ~45–60 s per worker per generation. With 22-minute jobs that is ~4%
overhead, which is the price of the 4× rate gain — worth it, but it means shortening jobs
further has diminishing returns.

**Adding GPUs does not fix 2.56×.** See `CONTROLLER_DIAGNOSIS.md` — 94% of every tick is
one controller call, and the always-on controller needs that under 150 ms.

---

## 6. Recipe

```bash
# 1. rebuild shards from what is actually undone (quarantines the too-long ones)
python omniprofast/reshard_remaining.py --run-dir output_full9 --out splits_r2 --n 16

# 2. launch 16 GPUs, chained. Same out dir -> resume skips finished samples.
sbatch run_16gpu.sbatch output_full9 splits_r2 1 20

# 3. watch
squeue -u $USER -o "%.10i %.11j %.9T %.6M %.9L"
```

Sanity checks before trusting a run:
- `sacct -j <id> --format=JobID,State,AllocNodes,AllocTRES%50` — confirm the **srun step**
  shows all nodes and GPUs, not just the batch step.
- If a job is `PENDING` with `QOSMaxNodeMinutesPerJob`, your `nodes × time` exceeds 90.
- Confirm shards open with short videos: `python -c "import json;print([round(e['duration']) for e in json.load(open('omniprofast/splits_r2/r00.json'))[:5]])"`
