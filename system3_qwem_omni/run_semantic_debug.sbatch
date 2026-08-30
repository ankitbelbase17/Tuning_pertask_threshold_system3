#!/bin/bash
#SBATCH --job-name=sca_eval
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_semantic/slurm_%j.log
# Run the 11 semantic_condition_alert (audio=none) videos on a DEDICATED debug node.
# Uses all 4 GH200s: controller (cuda:1) + encoder (cuda:2) on their own GPUs so the
# controller's decode doesn't time-share with frame encode/ingest (config.py devices).
# Rebuilds the conda env only if it's broken.
set -uo pipefail

PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
REPO=/iopsstor/scratch/cscs/dbartaula/system_3

echo "[job] host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true

# --- env sanity: rebuild ONLY if the interpreter/imports are broken ---
if "$PY" -c "import torch,transformers,av,requests,PIL; from google import genai; print('[job] env OK', torch.__version__, 'cuda:', torch.cuda.is_available())"; then
  echo "[job] env is healthy, skipping recreate"
else
  echo "[job] env BROKEN -> running recreate_env.sh"
  bash "$REPO/recreate_env.sh"
fi

# --- run: 11 videos, dedicated GPU (SLURM already isolated one to us) ---
cd "$REPO/omniprofast"
bash run_fast.sh \
  --benchmark_json /iopsstor/scratch/cscs/dbartaula/omni_pro/dataset/benchmark.json \
  --tasks semantic_condition_alert \
  --audio none \
  --no_resume \
  --out "$REPO/omniprofast/output_semantic"

echo "[job] DONE rc=$?"
