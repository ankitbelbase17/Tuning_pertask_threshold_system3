#!/usr/bin/env bash
# recreate_env.sh — build a FRESH conda env for the async_omni_v2 (prosync) pipeline
# + OmniPro eval harness. The scratch miniforge periodically loses its stdlib .py
# files (env corruption); this script rebuilds a clean env from the INTACT home
# miniforge bootstrap. Everything is installed via the new env's OWN interpreter
# (full paths, `python -m pip`), so nothing depends on shell activation.
#
# Usage:  bash recreate_env.sh
# Then point run.sh / run_fast.sh PY at the printed interpreter path.
set -euo pipefail

# --- bootstrap: the home miniforge survives when the scratch one corrupts ---
BOOT_CONDA=/users/dbartaula/miniforge3/bin/conda
ENV=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env
PY="$ENV/bin/python"
PYVER=3.12.13

[ -x "$BOOT_CONDA" ] || { echo "FATAL: bootstrap conda missing at $BOOT_CONDA"; exit 1; }

echo "[recreate_env] creating fresh env: $ENV (python $PYVER)"
"$BOOT_CONDA" create -y -p "$ENV" "python=$PYVER"

echo "[recreate_env] interpreter: $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"
"$PY" -m pip install --upgrade pip

# --- torch: GH200 (aarch64) + CUDA 13.0, matching the previous working env ---
echo "[recreate_env] installing torch 2.12.1 + torchvision 0.27.1 (cu130, aarch64)"
# torchvision is REQUIRED: the Qwen3-VL AutoProcessor pulls in Qwen3VLVideoProcessor
# which needs the torchvision backend, or from_pretrained raises ImportError.
"$PY" -m pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130

# --- model + video + eval + judge deps (versions matched to the working env) ---
echo "[recreate_env] installing model/eval deps"
"$PY" -m pip install --no-cache-dir \
  "transformers==5.12.1" \
  "accelerate==1.14.0" \
  "huggingface_hub==1.20.1" \
  "safetensors==0.8.0" \
  "av==17.1.0" \
  "pillow" \
  "numpy" \
  "requests" \
  "google-genai"

# --- verify: every import the pipeline + judge needs, and CUDA visibility ---
echo "[recreate_env] verifying imports..."
"$PY" - <<'PYEOF'
import torch, transformers, accelerate, av, requests, safetensors, huggingface_hub, PIL
from google import genai
print("torch        ", torch.__version__, "| cuda_available:", torch.cuda.is_available(),
      "| device_count:", (torch.cuda.device_count() if torch.cuda.is_available() else 0))
print("transformers ", transformers.__version__)
print("accelerate   ", accelerate.__version__)
print("av           ", av.__version__)
print("google-genai : import OK")
print("ALL IMPORTS OK")
PYEOF

echo
echo "[recreate_env] DONE. New interpreter:"
echo "    $PY"
echo "[recreate_env] Update PY in async_omni_v2/run.sh and omniprofast/run_fast.sh to that path."
