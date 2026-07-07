#!/usr/bin/env bash
# run_fast.sh — FAST OmniPro eval: system_5, ONLINE mode, bundled 27-video subset
#               (9 tasks x 3 shortest, audio_dependency == none).
#
# Zero-config: just `bash run_fast.sh`. Dataset / python / model paths are
# hardcoded below (same machine), and the LLM-judge credentials are read from
# the nearest .env found by walking up from this script (your project root).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- 1. LLM-as-judge credentials: source the nearest .env (project root) ----
# Picks up GEMINI_API_KEY (and GEMINI_API_BASE / GEMINI_MODEL if you set them).
__d="$HERE"
while [ "$__d" != "/" ] && [ ! -f "$__d/.env" ]; do __d="$(dirname "$__d")"; done
if [ -f "$__d/.env" ]; then
  set -a; . "$__d/.env"; set +a
  echo "[run_fast] sourced env: $__d/.env"
else
  echo "[run_fast] WARNING: no .env found above $HERE — judge will use lexical fallback"
fi

# --- 2. Hardcoded machine paths (dataset, python, model pipeline, HF cache) --
# These live on this system, so the dir is drop-in from any repo on this box.
export SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/dbartaula}"
export PY="/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python"
export PROSYNC_DIR="/iopsstor/scratch/cscs/dbartaula/system_3/async_omni_v2"   # this repo's system_5 pipeline (Qwen3-VL)
export OMNIPRO_DATASET_DIR="/iopsstor/scratch/cscs/dbartaula/omni_pro/dataset"  # raw_videos/*.mp4
export HF_HOME="/iopsstor/scratch/cscs/dbartaula/hf_cache"       # Qwen3-VL weights (system_5)
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# subset + output stay INSIDE this copied dir (portable, no $SCRATCH/eval writes)
export OMNIPRO_BENCHMARK_JSON="$HERE/benchmark_mini.json"
export OMNIPRO_OUTPUT_DIR="$HERE/output"

# --- 3. Judge sanity check --------------------------------------------------
if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "[run_fast] GEMINI_API_KEY found -> LLM judge (event_narration, sequential_step_instruction)"
  if [ -z "${GEMINI_API_BASE:-}" ]; then
    echo "[run_fast] WARNING: GEMINI_API_KEY set but GEMINI_API_BASE is NOT."
    echo "[run_fast]          Gemini calls will fail and silently degrade to lexical scoring."
    echo "[run_fast]          Add GEMINI_API_BASE (and optionally GEMINI_MODEL) to your .env."
  else
    echo "[run_fast] GEMINI_API_BASE=$GEMINI_API_BASE  GEMINI_MODEL=${GEMINI_MODEL:-gemini-3-flash-preview}"
  fi
else
  echo "[run_fast] no GEMINI_API_KEY -> free-text tasks scored by lexical fallback"
fi

# --- 4. Run: system_5, online, all 9 tasks x 3 shortest, one prompt variant --
VARIANT="${VARIANT:-v01_baseline_direct}"   # set VARIANT="" to sweep all 10
echo "[run_fast] running system_5 ONLINE on 27 videos (variant=${VARIANT:-ALL})"
"$PY" evaluate.py \
  --mode online \
  --system system5 \
  --variants "$VARIANT" \
  --fps "${FPS:-1.0}" \
  --tolerance "${TOLERANCE:-3.0}" \
  --out "$OMNIPRO_OUTPUT_DIR" \
  --no_wandb \
  "$@"

echo
echo "== online_metrics.json =="
find "$OMNIPRO_OUTPUT_DIR" -name online_metrics.json -print
