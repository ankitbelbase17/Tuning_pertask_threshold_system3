#!/bin/bash
#SBATCH --job-name=omnipro16
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=00:22:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --signal=B:USR1@120
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/slurm16_%j.log
#
# 16-GPU run of the REMAINING OmniPro work (audio_dependency != required).
#
# VERIFIED, NOT ASSUMED (job 2968290):
#   AllocNodes=4  gres/gpu=16   srun step exit 0:0
# debug-qos allows MaxNodes=4 and each node carries 4 GPUs, so 16 GPUs is the
# ceiling here. MaxJobsPerUser=1 / MaxSubmitJobsPerUser=2 still hold, so exactly
# one job runs at a time and we keep the pre-submitted afterany chain.
#
# WHY srun IS MANDATORY. In that same test the BATCH step reported node=1,
# gres/gpu=4 while the srun step reported node=4, gres/gpu=16. The batch script
# body runs on the head node only -- the previous launcher looped
# `CUDA_VISIBLE_DEVICES=$k bash run_fast.sh` in the shell, so with --nodes=4 it
# would have left three nodes allocated and idle. Work MUST go through srun.
#
# Mapping: one task per GPU. SLURM_PROCID (0..15) picks the shard,
# SLURM_LOCALID (0..3) picks the GPU within its node.
#
# Shards come from reshard_remaining.py, which splits ONLY unfinished samples --
# the original 16-way split has 8 complete shards, so reusing it would idle half
# the GPUs. Balanced by video duration (wall time is ~2.56x video duration).
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python

OUT_NAME=${1:-output_full9}      # SAME dir as the 4-GPU run: resume skips its 507 done samples
SPLIT_NAME=${2:-splits_r2}
GEN=${3:-1}
MAX_GEN=${4:-8}
NSHARD=16

OUT=$REPO/omniprofast/$OUT_NAME
SPLIT=$REPO/omniprofast/$SPLIT_NAME
mkdir -p "$OUT"
cd "$REPO/omniprofast"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} GEN=$GEN/$MAX_GEN"
echo "[job] nodes=${SLURM_JOB_NUM_NODES:-?} nodelist=${SLURM_JOB_NODELIST:-?}"
echo "[job] out=$OUT split=$SPLIT"

remaining () {   # samples still unscored IN r00..r15, straight from the predictions
  # r[0-9][0-9].json, NOT *.json. splits_r2 also holds toolong.json, the 11
  # quarantined 495-526 s samples that this job shape can never finish. Globbing
  # *.json floored this count at the number of undone long samples, so the
  # "NOTHING LEFT" branch below could never fire and the hand-off to
  # run_long.sbatch could never happen.
  "$PY" - <<'EOF' 2>/dev/null || echo 999
import glob, json, os
out = os.environ["OUT"]; split = os.environ["SPLIT"]
done = set()
for p in glob.glob(os.path.join(out, "**", "online_pred.jsonl"), recursive=True):
    for line in open(p, errors="replace"):
        line = line.strip()
        if line:
            try: done.add(json.loads(line)["id"])
            except Exception: pass
todo = 0
for f in glob.glob(os.path.join(split, "r[0-9][0-9].json")):
    todo += sum(1 for e in json.load(open(f)) if e["id"] not in done)
print(todo)
EOF
}
export OUT SPLIT
LEFT=$(remaining)
echo "[job] samples remaining at start: $LEFT"

# ---- chain: pre-submit the next generation before doing any work ------------
# afterany fires on ANY terminal state, so a wall-clock kill cannot break it.
CHAINED=0
chain () {
  [ "$CHAINED" = "1" ] && return
  [ "$GEN" -ge "$MAX_GEN" ] && { echo "[chain] MAX_GEN reached"; return; }
  cd "$REPO" || return
  if sbatch --dependency=afterany:${SLURM_JOB_ID} run_16gpu.sbatch \
       "$OUT_NAME" "$SPLIT_NAME" $((GEN + 1)) "$MAX_GEN"; then
    CHAINED=1; echo "[chain] gen $((GEN + 1)) PRE-QUEUED (afterany:${SLURM_JOB_ID})"
  else
    echo "[chain] pre-submit refused; USR1 will retry"
  fi
  cd "$REPO/omniprofast" || return
}
on_timeout () {
  echo "[chain] SIGUSR1 — 180s to the wall clock; saving scores."
  "$PY" dump_scores.py "$OUT" 2>&1 | tail -3 || true
  chain
}
trap on_timeout USR1

# ---- hand-off: the 11 quarantined long videos -------------------------------
# splits_r2/toolong.json holds 495-526 s samples that CANNOT finish inside this
# 22-minute shape. They get their own job geometry (2 nodes x 44 min = 88
# node-minutes, still under the debug-qos MaxTRESMins node=90 cap), which only
# becomes submittable once this chain stops occupying both queue slots.
#
# CAP PROOF. debug-qos is MaxJobsPerUser=1 / MaxSubmitJobsPerUser=2. Steady state
# of this chain is {gen N running, gen N+1 pending} = 2 = the cap, so nothing else
# can be submitted mid-chain. chain() queues gen N+1 only when LEFT > 0 AND
# GEN < MAX_GEN. The hand-off below runs only when LEFT <= 0 OR GEN >= MAX_GEN --
# the exact negation -- so at that point chain() queued nothing, this job is the
# ONLY job this user has in the system, and exactly one submit slot is free.
# Exactly one job is what we submit. afterany on this job means it also cannot
# start while this one runs (MaxJobsPerUser=1 would have blocked it anyway).
LONGED=0
submit_long () {
  [ "$LONGED" = "1" ] && return
  # belt and braces: never submit while a successor of ours is already queued.
  [ "$CHAINED" = "1" ] && { echo "[long] successor queued — not submitting"; return; }
  cd "$REPO" || return
  # Retry: the only realistic refusal is the predecessor still sitting in
  # COMPLETING, which momentarily keeps the submit count at 2. 10 x 60 s stays
  # well inside this job's 22-minute wall clock.
  for _try in 1 2 3 4 5 6 7 8 9 10; do
    if sbatch --dependency=afterany:${SLURM_JOB_ID} run_long.sbatch "$OUT_NAME" 1 6; then
      LONGED=1
      echo "[long] run_long.sbatch QUEUED (afterany:${SLURM_JOB_ID}) — long videos will run"
      break
    fi
    echo "[long] submit refused (try $_try/10); retrying in 60s"
    sleep 60
  done
  [ "$LONGED" = "1" ] || echo "[long] FAILED to queue run_long.sbatch — long videos NOT scheduled"
  cd "$REPO/omniprofast" || return
}

if [ "$LEFT" -gt 0 ]; then chain; fi

# Hand off only when the chain is genuinely OVER, not merely stalled: no r-shard
# work left, or the generation cap reached. Both imply chain() queued nothing, so
# CHAINED=0 and the slot is free. Deliberately NOT "CHAINED != 1": a transient
# sbatch refusal must stay recoverable by the USR1 retry of chain(), not silently
# replace the main chain with the long-video chain.
if [ "$LEFT" -le 0 ] || [ "$GEN" -ge "$MAX_GEN" ]; then submit_long; fi

if [ "$LEFT" -le 0 ]; then
  echo "[job] NOTHING LEFT in r00..r15 — main chain finished."
  "$PY" dump_scores.py "$OUT" 2>&1 | tail -5 || true
  exit 0
fi

# ---- the work: 16 tasks, one GPU each --------------------------------------
srun --ntasks=$NSHARD --ntasks-per-node=4 --kill-on-bad-exit=0 bash -c '
  G=$(printf "r%02d" $SLURM_PROCID)
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export OMNIPRO_GATE_MODE=controller
  export OMNIPRO_DECODE_MODE=schema
  echo "[task $SLURM_PROCID] $(hostname) gpu=$SLURM_LOCALID shard=$G"
  bash run_fast.sh \
    --audio none_helpful \
    --benchmark_json "'"$SPLIT"'/$G.json" \
    --dataset_dir "'"$DS"'" \
    --out "'"$OUT"'/$G" \
    >> "'"$OUT"'/${G}_launch.log" 2>&1
  echo "[task $SLURM_PROCID] $G exit=$?"
' &
SRUN_PID=$!
# plain `wait` is interrupted by the USR1 trap; loop until the step really ends
while kill -0 $SRUN_PID 2>/dev/null; do wait $SRUN_PID 2>/dev/null || true; done

echo "[job] gen $GEN done; samples remaining: $(remaining)"
"$PY" dump_scores.py "$OUT" 2>&1 | tail -5
chain
echo "[job] DONE gen $GEN"
