#!/usr/bin/env bash
# Build the `mobileo` env + download the Mobile-O-0.5B checkpoint.
# Run on the hala LOGIN node (needs internet; CPU only). Idempotent-ish.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

ENV=mobileo
echo "[1/4] create env $ENV (python 3.12)"
micromamba create -y -n "$ENV" python=3.12 || echo "env may already exist"

run() { micromamba run -n "$ENV" "$@"; }

echo "[2/4] install torch 2.3.0 (cu121) + deps"
run pip install --no-input torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
# image-understanding only: no flash-attn / deepspeed / diffusers / gradio needed
# Version pins matter: the mobileo code imports Qwen2_5_VLConfig (needs transformers>=4.49)
# and timm.layers (needs timm>=0.9); latest transformers (5.x) has an import bug, so pin 4.49.
run pip install --no-input \
    "transformers==4.49.0" "diffusers==0.35.2" "timm==0.9.16" accelerate \
    einops einops-exts sentencepiece shortuuid ftfy pillow \
    sentence-transformers scikit-learn rouge-score
echo "[3/4] install mobileo package (editable)"
run pip install --no-input -e "$ROOT/Mobile-O"

echo "[4/4] download Mobile-O-0.5B checkpoint + MiniLM (for prompt similarity)"
# NOTE: checkpoint files are at the repo ROOT (the README's allow_patterns is wrong for this repo)
run python -c "from huggingface_hub import snapshot_download; p=snapshot_download(repo_id='Amshaker/Mobile-O-0.5B', local_dir='$ROOT/checkpoints/Mobile-O-0.5B'); print('CKPT:',p)"
run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); print('MiniLM cached')"

echo "DONE. checkpoint at: $ROOT/checkpoints/Mobile-O-0.5B"
