#!/bin/bash
#SBATCH --job-name=omnipro_all
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --array=0-3
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_all/slurm_%A_%a.log
#
# FULL RUN — every OmniPro sample that does NOT require audio.
#
#   932 samples / 699 videos  (audio_dependency in {none, helpful})
#   16 duration-balanced shards (splits_all/g00..g15), 2.28 video-hours each
#   4 array tasks x 4 GPUs = 16 workers, one shard per GPU
#   ~36.5 video-hours total -> ~3h wall at the measured 1.3 s/tick
#
# TWO THINGS THIS RUN FIXES vs every previous eval:
#
#  1. DATASET DIR. run_fast.sh hardcodes omni_pro/dataset, which holds only 34 of
#     the 932 videos -- so every historical result was silently capped by what
#     happened to be downloaded (the famous "11 SCA videos" is just all there was).
#     omnipro_data has all 932. We pass --dataset_dir explicitly.
#
#  2. DECODE MODE. schema-walk, which passed its gate 157/157 and runs 2.8x faster
#     with 3.5x fewer tokens than the free decode.
#
# Every tick logs p_hit (the continuous confidence from the logit read), which is
# the raw material for the dense AUC metric (ROADMAP 1.4) -- so this run doubles as
# the dataset for offline threshold sweeps that need no GPU.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data     # <-- has ALL 932 videos
SPLIT=$REPO/omniprofast/splits_all
OUT=$REPO/omniprofast/output_all
mkdir -p "$OUT"

A=${SLURM_ARRAY_TASK_ID:-0}
echo "[job] host=$(hostname) array=$A jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

cd "$REPO/omniprofast"

# array task A drives GPUs 0..3 -> shards 4A .. 4A+3
for k in 0 1 2 3; do
  G=$(printf "g%02d" $((A * 4 + k)))
  echo "[job] GPU$k -> shard $G ($(python -c "import json;print(len(json.load(open('$SPLIT/$G.json'))))" 2>/dev/null || echo '?') samples)"
  CUDA_VISIBLE_DEVICES=$k \
  OMNIPRO_GATE_MODE=controller \
  OMNIPRO_DECODE_MODE=schema \
    bash run_fast.sh \
      --audio none_helpful \
      --benchmark_json "$SPLIT/$G.json" \
      --dataset_dir "$DS" \
      --out "$OUT/$G" \
      > "$OUT/${G}_launch.log" 2>&1 &
done

wait
echo "[job] array task $A: all 4 shards finished"
for k in 0 1 2 3; do
  G=$(printf "g%02d" $((A * 4 + k)))
  m="$OUT/$G/online_metrics.json"
  echo "[job]   $G: $([ -f "$m" ] && echo present || echo MISSING)"
done
echo "[job] DONE array task $A"

# Persist per-tick p_hit so ANY threshold / gate strategy can be re-derived later
# offline, with no GPU. This is what makes the firing threshold a free decision.
/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python \
  dump_scores.py "$OUT" 2>&1 | tail -20
