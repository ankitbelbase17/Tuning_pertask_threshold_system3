#!/usr/bin/env python
"""Fetch Qwen2.5-Omni-7B into THIS user's HF cache.

dbartaula/hf_cache is mode 700, so the weights the fork was smoke-tested against
are unreadable from the dthapa account. Everything else in that scratch (the repo,
omnipro_data, miniforge3) is group-readable; only the model cache is not.
Compute nodes have no outbound network, so this must run on a LOGIN node.
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/iopsstor/scratch/cscs/dthapa/hf_cache")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# prosync_env lacks hf_xet and is read-only to this account, so force the classic
# HTTP download path instead of Xet storage (ValueError: need hf_xet otherwise).
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2.5-Omni-7B"
t0 = time.time()
p = snapshot_download(
    REPO,
    max_workers=16,
    # .bin duplicates the safetensors shards; skip it and the 30 GB of extras
    ignore_patterns=["*.bin", "*.pth", "*.msgpack", "*.h5", "*.onnx"],
)
print(f"OK {REPO} -> {p}  ({time.time()-t0:.0f}s)", flush=True)
