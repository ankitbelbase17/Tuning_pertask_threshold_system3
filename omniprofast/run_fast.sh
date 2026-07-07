#!/usr/bin/env bash
# run_fast.sh — FAST OmniPro eval: icl_ingester_writer, ONLINE mode, bundled
#               27-video subset (9 tasks x 3 shortest, audio_dependency == none).
#
# Zero-config: just `bash run_fast.sh`. Dataset / python / model paths are
# hardcoded below, and the Gemini LLM-judge key is read from the nearest .env.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- 1. LLM-as-judge credentials: source the nearest .env (project root) ----
__d="$HERE"
while [ "$__d" != "/" ] && [ ! -f "$__d/.env" ]; do __d="$(dirname "$__d")"; done
if [ -f "$__d/.env" ]; then
  set -a; . "$__d/.env"; set +a
  echo "[run_fast] sourced env: $__d/.env"
else
  echo "[run_fast] WARNING: no .env found above $HERE — judge will use lexical fallback"
fi

# --- 2. Hardcoded machine paths (dataset, python, model pipeline, HF cache) --
export SCRATCH="${SCRATCH:-/iopsstor/scratch/cscs/dbartaula}"
export PY="/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python"
export PROSYNC_DIR="/iopsstor/scratch/cscs/dbartaula/system_3/async_omni_v2"
export OMNIPRO_DATASET_DIR="/iopsstor/scratch/cscs/dbartaula/omni_pro/dataset"
export HF_HOME="/iopsstor/scratch/cscs/dbartaula/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export OMNIPRO_BENCHMARK_JSON="$HERE/benchmark_mini.json"
export OMNIPRO_OUTPUT_DIR="$HERE/output"

if [ -n "${GEMINI_API_KEY:-}" ]; then
  echo "[run_fast] GEMINI_API_KEY found -> Gemini LLM judge (model=${GEMINI_MODEL:-gemini-3.5-flash})"
else
  echo "[run_fast] no GEMINI_API_KEY -> free-text tasks scored by lexical fallback"
fi

# --- 3. Run: icl_ingester_writer, online, all 9 tasks x 3 shortest -----------
VARIANT="${VARIANT:-v01_baseline_direct}"
echo "[run_fast] running icl_ingester_writer ONLINE on 27 videos (variant=$VARIANT)"
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
