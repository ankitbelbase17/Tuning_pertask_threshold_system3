#!/usr/bin/env bash
# Run the async-omni v2 (Qwen3-VL-8B) soccer demo ON YOUR GPU SERVER.
# Edit PY / VIDEO below for that machine, then:
#   ./run.sh                 # realtime, single GPU (everything on cuda:0)
#   ./run.sh batch           # as fast as the GPU allows, single GPU
#   ./run.sh realtime 2      # + writer on cuda:1 (minimal trigger->speech lag)
#   ./run.sh realtime 3      # + writer on cuda:1, encoder on cuda:2 (max parallel)
#   ./run.sh batch 3         # batch, 3 GPUs
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

export PYTHONNOUSERSITE=1                    # ignore ~/.local site so the env's torch is used
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python   # Qwen3-VL env
MODEL="Qwen/Qwen3-VL-8B-Instruct"           # HF id (downloads on first run)
VIDEO="/iopsstor/scratch/cscs/dbartaula/system_3/Highlights ｜ France 3-1 Senegal ｜ FIFA World Cup 2026™ [n3JDGlOwMJ4].webm"    # <-- set this on the server

MODE="${1:-batch}"
GPUS="${2:-3}"        # 1 = all on cuda:0; 2 = + writer on cuda:1; 3 = + encoder on cuda:2
# realtime = ~1x wall clock (speed 1.0); batch = as fast as the GPU allows.
if [ "$MODE" = "batch" ]; then CLOCK="--no_realtime"; else CLOCK="--realtime --speed 1.0"; fi

# GPU placement. Orchestrator (the primary KV cache) always lives on cuda:0.
#   1 GPU: encoder + orchestrator + writer all share the cuda:0 model.
#   2 GPU: writer replica on cuda:1 (parallel commentary).
#   3 GPU: + encoder replica on cuda:2 (vision encode no longer competes).
case "$GPUS" in
  3) GPUFLAGS="--device cuda:0 --writer_device cuda:1 --encoder_device cuda:2" ;;
  2) GPUFLAGS="--device cuda:0 --writer_device cuda:1" ;;
  *) GPUFLAGS="--device cuda:0" ;;
esac

# Reproducible eval needs --deterministic + a fixed --seed AND batch mode
# (realtime pacing is wall-clock, inherently nondeterministic). Drop --deterministic
# for max speed (non-reproducible). PRUNE toggles VisionZip token pruning.
REPRO="--deterministic --seed 0"
PRUNE=""    # e.g. PRUNE="--prune_img_tokens --prune_dominant_frac 0.60 --prune_contextual_frac 0.05"

PYTHONPATH="$HERE" "$PY" run.py \
  --model_id "$MODEL" \
  --video_path "$VIDEO" \
  --dtype float16 \
  --fps 8.0 \
  --max_seconds 120 \
  --kv_budget 262144 \
  --goal_threshold 0.5 \
  --log_gate_every 1 \
  --timestamp_tokens \
  $REPRO $PRUNE $GPUFLAGS $CLOCK
