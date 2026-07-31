#!/bin/bash
#SBATCH --job-name=omniprolong
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=00:44:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --signal=B:USR1@120
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/slurmlong_%j.log
#
# The QUARANTINED long videos: splits_r2/toolong.json, 11 samples of 495-526 s.
#
# WHY A SEPARATE JOB SHAPE. reshard_remaining.py pulls anything longer than
# (22*60 - 60)/2.56 ~= 500 video-seconds out of r00..r15, because evaluate.py walks
# a shard in file order and resumes per sample: one sample that can never finish
# inside the 22-minute window blocks everything behind it forever. These need a
# longer wall clock, and on debug-qos a longer wall clock means fewer nodes.
#
# NODE-MINUTE ARITHMETIC (debug-qos MaxTRESMins node=90, verified with sacctmgr):
#     run_16gpu.sbatch : 4 nodes x 22 min = 88 node-min   <= 90  OK (proven to run)
#     this job         : 2 nodes x 44 min = 88 node-min   <= 90  OK
#     tempting but dead: 4 nodes x 90 min = 360 node-min  >  90  QOSMaxNodeMinutesPerJob
# 44 rather than 45 on purpose: 2 x 45 = 90 sits EXACTLY on the cap, and whether
# SLURM rejects at ">" or ">=" is not something we can test without submitting.
# 88 is the value already proven to pass. The minute buys nothing we need: at the
# worst rate ever measured on these videos (2.48 s wall per s of video, job 2968312)
# a 526 s video costs 526*2.48/60 + 1 min of model load = 22.7 min, so a 44-minute
# job clears it with ~19 minutes of headroom.
#
# WHY srun IS MANDATORY. With --nodes=N the batch script body runs on the HEAD NODE
# ONLY -- job 2968290 showed the batch step at node=1,gpu=4 while its srun step got
# node=4,gpu=16. A bash loop here would leave the second node allocated and idle.
# SLURM_PROCID (0..7) picks the bucket, SLURM_LOCALID (0..3) picks the GPU.
#
# Output goes to the SAME output_full9 as the main run, so evaluate.py --resume and
# the final aggregate see one corpus in one place.
#
# Self-chains via afterany, exactly like run_16gpu.sbatch, because 8 workers cannot
# be relied on to clear up to 11 long samples in one generation.
#
#   sbatch run_long.sbatch [OUT_NAME] [GEN] [MAX_GEN]
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python

OUT_NAME=${1:-output_full9}      # MUST match the main run: shared resume + aggregate
GEN=${2:-1}
MAX_GEN=${3:-6}
NSHARD=8                         # 2 nodes x 4 GPUs

OUT=$REPO/omniprofast/$OUT_NAME
SPLIT=$REPO/omniprofast/splits_long
mkdir -p "$OUT"
cd "$REPO/omniprofast"

echo "[long] host=$(hostname) jobid=${SLURM_JOB_ID:-none} GEN=$GEN/$MAX_GEN"
echo "[long] nodes=${SLURM_JOB_NUM_NODES:-?} nodelist=${SLURM_JOB_NODELIST:-?}"
echo "[long] out=$OUT split=$SPLIT"

# ---- rebuild the buckets from what is ACTUALLY still unpredicted --------------
# Reads done-ness from output_full9/**/online_pred.jsonl, the same source
# evaluate.py --resume uses, so this can never disagree with it. Rebuilding every
# generation means samples the r00..r15 chain finished on its own are skipped, and
# the tail generations are not 8 workers idling behind one straggler.
SPLIT_OUT=$("$PY" split_long.py --run-dir "$OUT_NAME" --out splits_long --n $NSHARD 2>&1)
echo "$SPLIT_OUT"
LEFT=$(printf '%s\n' "$SPLIT_OUT" | sed -n 's/^REMAINING=//p' | tail -1)
case "${LEFT:-}" in ''|*[!0-9]*) LEFT=999 ;; esac   # unreadable => assume work left
echo "[long] long samples remaining at start: $LEFT"

# ---- chain: pre-submit the next generation before doing any work --------------
# afterany fires on ANY terminal state, so a wall-clock kill cannot break the chain.
# Cap bookkeeping: while this job runs the queue holds {this, next} = 2 = exactly
# MaxSubmitJobsPerUser, which is the same steady state run_16gpu.sbatch keeps.
CHAINED=0
chain () {
  [ "$CHAINED" = "1" ] && return
  [ "$GEN" -ge "$MAX_GEN" ] && { echo "[long] MAX_GEN reached — chain ends here"; return; }
  cd "$REPO" || return
  if sbatch --dependency=afterany:${SLURM_JOB_ID} run_long.sbatch \
       "$OUT_NAME" $((GEN + 1)) "$MAX_GEN"; then
    CHAINED=1; echo "[long] gen $((GEN + 1)) PRE-QUEUED (afterany:${SLURM_JOB_ID})"
  else
    echo "[long] pre-submit refused; USR1 will retry"
  fi
  cd "$REPO/omniprofast" || return
}
on_timeout () {
  echo "[long] SIGUSR1 — 120s to the wall clock; saving scores."
  "$PY" dump_scores.py "$OUT" 2>&1 | tail -3 || true
  chain
}
trap on_timeout USR1

if [ "$LEFT" -le 0 ]; then
  echo "[long] NOTHING LEFT — every quarantined long video is predicted. Chain ends."
  "$PY" dump_scores.py "$OUT" 2>&1 | tail -5 || true
  exit 0
fi
chain

# ---- the work: 8 tasks, one GPU each ------------------------------------------
srun --ntasks=$NSHARD --ntasks-per-node=4 --kill-on-bad-exit=0 bash -c '
  G="L$SLURM_PROCID"
  export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
  export OMNIPRO_GATE_MODE=controller
  export OMNIPRO_DECODE_MODE=schema
  echo "[task $SLURM_PROCID] $(hostname) gpu=$SLURM_LOCALID bucket=$G"
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

echo "[long] gen $GEN done."
"$PY" split_long.py --run-dir "$OUT_NAME" --out splits_long --n $NSHARD 2>&1 | tail -3
"$PY" dump_scores.py "$OUT" 2>&1 | tail -5
chain
echo "[long] DONE gen $GEN"
