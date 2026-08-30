#!/usr/bin/env bash
# Run the icl_ingester_writer pipeline (Qwen3-VL-8B) on a video, single GPU.
# The controller self-paces in VIDEO time, so we always run realtime.
#   ./run.sh                      # default video
#   ./run.sh /path/to/video.mp4   # a specific video
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

export PYTHONNOUSERSITE=1
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
MODEL="Qwen/Qwen3-VL-8B-Instruct"
VIDEO="${1:-/iopsstor/scratch/cscs/dbartaula/system_3/Highlights ｜ France 3-1 Senegal ｜ FIFA World Cup 2026™ [n3JDGlOwMJ4].webm}"

PYTHONPATH="$HERE" "$PY" run.py \
  --model_id "$MODEL" \
  --video_path "$VIDEO" \
  --dtype bfloat16 \
  --fps 8.0 \
  --max_seconds 120 \
  --kv_budget 262144 \
  --realtime --speed 1.0
