#!/bin/bash
#SBATCH --job-name=omnipro_all
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --signal=B:USR1@180
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/slurm_%j.log
#
# FULL RUN — every OmniPro sample that does NOT require audio.
#   932 samples / 699 videos (audio_dependency in {none, helpful})
#   16 duration-balanced shards, 4 GPUs, ~36.5 video-hours total
#
# SELF-CHAINING. The debug partition caps jobs at 1:30 AND enforces a per-user
# job limit (QOSMaxSubmitJobPerUserLimit), so a dependency chain cannot be
# pre-submitted — sbatch refuses. Instead this job RESUBMITS ITSELF at the end
# if work remains: the resubmit lands as the current job releases its slot, so
# the per-user limit is never violated. evaluate.py resumes by default, so each
# generation picks up exactly where the previous one stopped.
#
# Stops when all 16 shards have written online_metrics.json, or at MAX_GEN.
#
# TWO THINGS THIS FIXES vs every previous eval:
#  1. DATASET DIR — run_fast.sh hardcodes omni_pro/dataset, which holds only 34
#     of the 932 videos, so every historical result was silently capped by what
#     happened to be downloaded (the "11 SCA videos" is just all there was).
#     omnipro_data has all 2700. We pass --dataset_dir explicitly.
#  2. DECODE MODE — schema-walk, which passed its gate 647/647 and runs ~3x
#     faster with ~4x fewer tokens than the free decode.
#
# Per-tick p_hit is dumped to scores.jsonl every generation, so ANY firing
# threshold or gate strategy can be re-derived offline afterwards with no GPU.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
SPLIT=$REPO/omniprofast/splits_all
# OUT_NAME lets a config change start a CLEAN arm instead of resuming into results
# produced by different settings (resume is per-sample, so mixing would silently
# blend two configs in one number).
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python

# ⚠️ ARGUMENTS, NOT --export. SLURM's --export splits on COMMAS to separate
# NAME=VALUE pairs, so `--export=ALL,TASKS=a,b,c` silently became TASKS=a plus
# three bogus bare names -- and it dropped OUT_NAME as well. No error, no warning:
# job 2936893 ran with ONE task instead of five and wrote to the wrong directory.
# Positional args cannot be mangled that way.
#   sbatch run_all_audio_ok.sbatch <out_name> <task,list> [gen] [max_gen]
OUT_NAME=${1:-${OUT_NAME:-output_all}}
TASKS=${2:-${TASKS:-}}
GEN=${3:-${GEN:-1}}
MAX_GEN=${4:-${MAX_GEN:-12}}
NSHARD=16
OUT=$REPO/omniprofast/$OUT_NAME
mkdir -p "$OUT"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} GEN=$GEN/$MAX_GEN commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader || true
cd "$REPO/omniprofast"

done_count () {   # how many shards have finished (wrote their metrics)
  local n=0
  for i in $(seq 0 $((NSHARD - 1))); do
    [ -f "$OUT/$(printf 'g%02d' $i)/online_metrics.json" ] && n=$((n + 1))
  done
  echo $n
}

echo "[job] shards complete at start: $(done_count)/$NSHARD"

# ---------------------------------------------------------------------------
# CHAIN REPAIR (2026-07-30). Generation 1 (job 2932056) TIMED OUT and never
# resubmitted, so the chain died after one link with 20/932 samples done.
#
# Why the old design could never work: `wait` below blocks until all four GPU
# lanes finish their shards, and a shard is ~58 samples -- far more than fits in
# 1:30. So the tail of this script (including the resubmit) was UNREACHABLE; the
# job was always killed at the wall clock first.
#
# Fix: --signal=B:USR1@180 makes SLURM signal the batch script 180s BEFORE the
# limit. We resubmit from the handler, while still alive but with the slot about
# to free. Retry in a loop because the debug QOS caps submitted jobs per user --
# as this job winds down a retry gets in.
# ---------------------------------------------------------------------------
CHAINED=0                 # guard: resubmit at most once per generation
resubmit () {
  [ "$CHAINED" = "1" ] && { echo "[chain] already chained — skip."; return; }
  if [ "$GEN" -ge "$MAX_GEN" ]; then
    echo "[chain] hit MAX_GEN=$MAX_GEN — not resubmitting."; return
  fi
  cd "$REPO" || return
  for attempt in $(seq 1 30); do
    if sbatch run_all_audio_ok.sbatch "$OUT_NAME" "$TASKS" $((GEN + 1)) "$MAX_GEN"; then
      CHAINED=1
      echo "[chain] generation $((GEN + 1)) queued (attempt $attempt)"; return
    fi
    echo "[chain] sbatch refused (attempt $attempt) — retrying in 5s"; sleep 5
  done
  echo "[chain] FAILED to queue generation $((GEN + 1)) after 30 attempts"
}

on_timeout () {
  echo "[chain] SIGUSR1: ~180s to the wall clock. Saving scores + chaining."
  "$PY" dump_scores.py "$OUT" 2>&1 | tail -5 || true
  resubmit
  # let the shards keep working with the remaining time; SLURM kills us shortly.
}
trap on_timeout USR1

# 4 GPUs; GPU k takes shards k, k+4, k+8, k+12 in order. Resume makes an already
# finished shard cost seconds, so re-walking the list each generation is cheap.
for k in 0 1 2 3; do
(
  for i in $(seq $k 4 $((NSHARD - 1))); do
    G=$(printf "g%02d" $i)
    if [ -f "$OUT/$G/online_metrics.json" ]; then
      echo "[job] GPU$k  $G already complete — skip"
      continue
    fi
    echo "[job] GPU$k  -> $G"
    CUDA_VISIBLE_DEVICES=$k \
    OMNIPRO_GATE_MODE=controller \
    OMNIPRO_DECODE_MODE=schema \
      bash run_fast.sh \
        --audio none_helpful \
        ${TASKS:+--tasks "$TASKS"} \
        ${LIMIT:+--limit $LIMIT} ${SHORTEST:+--shortest} \
        ${MAXSEC:+--max_seconds $MAXSEC} \
        --benchmark_json "$SPLIT/$G.json" \
        --dataset_dir "$DS" \
        --out "$OUT/$G" \
        >> "$OUT/${G}_launch.log" 2>&1
  done
) &
done
# A plain `wait` is INTERRUPTED when the USR1 trap fires and would fall through
# with the lanes still running, so loop until no background lane remains.
while [ -n "$(jobs -pr)" ]; do
  wait -n 2>/dev/null || true
done
echo "[job] generation $GEN finished; shards complete: $(done_count)/$NSHARD"

# Persist per-tick p_hit -> any threshold becomes a free, offline decision.
"$PY" dump_scores.py "$OUT" 2>&1 | tail -20

# Normal-exit path. If the USR1 handler already chained, `resubmit` is a no-op.
if [ "$(done_count)" -ge "$NSHARD" ]; then
  echo "[job] ALL $NSHARD SHARDS COMPLETE — chain finished."
else
  echo "[job] work remains ($(done_count)/$NSHARD) — chaining generation $((GEN + 1))"
  resubmit
fi
echo "[job] DONE gen $GEN"
