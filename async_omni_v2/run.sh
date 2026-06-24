#!/usr/bin/env bash
# Run the async-omni v2 (Qwen3-VL-8B) soccer demo ON YOUR GPU SERVER.
# Edit PY / VIDEO below for that machine, then: ./run.sh   (or ./run.sh batch)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

export PYTHONNOUSERSITE=1                    # ignore ~/.local site so the env's torch is used
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/async_omni/bin/python   # Qwen3-VL env
MODEL="Qwen/Qwen3-VL-8B-Instruct"           # HF id (downloads on first run)
VIDEO="/iopsstor/scratch/cscs/dbartaula/system_3/Highlights ｜ France 3-1 Senegal ｜ FIFA World Cup 2026™ [n3JDGlOwMJ4].webm"    # <-- set this on the server

MODE="${1:-realtime}"
if [ "$MODE" = "batch" ]; then CLOCK="--no_realtime"; else CLOCK="--realtime --speed 2.0"; fi

PYTHONPATH="$HERE" "$PY" run.py \
  --model_id "$MODEL" \
  --video_path "$VIDEO" \
  --dtype float16 \
  --fps 1.5 \
  --max_seconds 120 \
  --kv_budget 4096 \
  --goal_threshold 0.5 \
  --log_gate_every 1 \
  $CLOCK
