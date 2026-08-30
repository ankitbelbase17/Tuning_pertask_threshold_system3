#!/bin/bash
#SBATCH --job-name=mvcc18
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_mvcc18/slurm_%j.log
#
# run_mvcc18.sbatch -- does dropping the MVCC snapshot cost ACCURACY?
#
# 2 videos from each of the 9 OmniPro tasks (18 samples, 563 video-seconds),
# chosen as the shortest per task from those output_full9 already covered.
#
# BOTH arms run NOW, at this commit, with the production config (sampling ON).
# Comparing a new run against the output_full9 log would confound the cache
# change with six commits of scorer/judge/gate drift landed since 07-30, so the
# old numbers are context, never the control. The control is the snapshot arm here.
#
#   GPU0 snapshot s0    GPU1 snapshot s1
#   GPU2 inplace  s0    GPU3 inplace  s1
#
# No judge during the GPU run (content is marked UNJUDGED; timing metrics are
# unaffected). judge_offline.py scores both arms afterwards from one shared cache,
# so identical predictions cost zero API calls and both arms are judged by the
# SAME backend -- otherwise the comparison is between two judges, not two systems.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
OUT=$REPO/omniprofast/output_mvcc18
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
mkdir -p "$OUT"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader || true
cd "$REPO/omniprofast"

export HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMNIPRO_GATE_MODE=controller
export OMNIPRO_DECODE_MODE=schema
export PROSYNC_DIR="$REPO/async_omni_v2"

k=0
for MODE in snapshot inplace; do
  for S in 0 1; do
    D="$OUT/$MODE/s$S"
    if [ -e "$D" ]; then echo "[job] FATAL: $D exists, refusing to reuse"; exit 1; fi
    mkdir -p "$D"
    echo "[job] GPU$k -> arm=$MODE shard=s$S"
    CUDA_VISIBLE_DEVICES=$k \
    OMNIPRO_CACHE_MODE=$MODE \
    OMNIPRO_DATASET_DIR="$DS" \
    OMNIPRO_BENCHMARK_JSON="$REPO/omniprofast/splits_mvcc/s$S.json" \
    OMNIPRO_OUTPUT_DIR="$D" \
    PYTHONPATH="$PROSYNC_DIR:$REPO/omniprofast" \
      "$PY" evaluate.py --audio all --no_resume --max_seconds 0 \
        --benchmark_json "$REPO/omniprofast/splits_mvcc/s$S.json" \
        --dataset_dir "$DS" --out "$D" \
        > "$D/launch.log" 2>&1 &
    k=$((k+1))
  done
done
wait
echo "[job] all 4 arms finished"

# An arm that crashed must NOT be scored: empty predictions compare equal to
# empty predictions, which is how a dead run turns into a green result.
FAIL=0
for MODE in snapshot inplace; do
  for S in 0 1; do
    D="$OUT/$MODE/s$S"
    N=$(wc -l < "$D/online_pred.jsonl" 2>/dev/null || echo 0)
    echo "[job] $MODE/s$S -> $N predictions"
    if [ "$N" -eq 0 ]; then echo "[job] FATAL: $MODE/s$S produced NOTHING"; FAIL=1; fi
    grep -m1 "cache_mode=" "$D/launch.log" || { echo "[job] no cache_mode line in $MODE/s$S"; FAIL=1; }
  done
done
[ $FAIL -eq 0 ] || { echo "[job] ABORTING before scoring -- an arm did not run"; exit 1; }

# merge the two shards of each arm into one prediction file
for MODE in snapshot inplace; do
  cat "$OUT/$MODE"/s*/online_pred.jsonl > "$OUT/$MODE/online_pred.jsonl"
  echo "[job] $MODE merged: $(wc -l < "$OUT/$MODE/online_pred.jsonl") predictions"
done
echo "[job] DONE -- score with: $PY judge_offline.py $OUT/snapshot (then inplace)"
