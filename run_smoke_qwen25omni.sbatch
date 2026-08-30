#!/bin/bash
#SBATCH --job-name=smoke_q25o
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system3_qwem_omni/smoke_qwen25omni/slurm_%j.log
#
# FIRST SMOKE TEST of the Qwen2.5-Omni-7B backend + Option-A audio ingestion +
# TMRoPE positions (see FORK_QWEN25OMNI.md). This is deliberately tiny:
#   - a short real clip (~12s) WITH a real audio track (confirmed via av
#     directly: HdbBkWjBQWw.mp4 has 1 audio + 1 video stream, non-silent)
#   - max_seconds=12 -> exercises seed, several frame ingests, ~6 audio-chunk
#     boundary crossings (12s / 2s per chunk), several controller ticks
#   - the ONE thing OMNI_EXTENSION.md's own recommended sequence calls the
#     single most important open question: does `from_pretrained` complete
#     under transformers 5.12.1 at all. Weights are already fully cached
#     (5/5 safetensor shards, no .incomplete) -- confirmed before submitting
#     this job, so this run tests LOADING + FORWARD, not the download.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system3_qwem_omni
OUT=$REPO/smoke_qwen25omni
mkdir -p "$OUT"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

export HF_HOME=/iopsstor/scratch/cscs/dbartaula/hf_cache
# WORKAROUND (documented in FORK_QWEN25OMNI.md sec 4 point 4): prosync_env's
# `av` is currently missing libXau on its link path (a pre-existing,
# environment-level issue -- affects vision_stream.py too, not introduced by
# this fork). Confirmed fix: point the linker at pillow's bundled copy.
export LD_LIBRARY_PATH="/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/lib/python3.12/site-packages/pillow.libs:${LD_LIBRARY_PATH:-}"

PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
VIDEO=/iopsstor/scratch/cscs/dbartaula/omnipro_data/raw_videos/HdbBkWjBQWw.mp4

echo "[job] env check:"
CUDA_VISIBLE_DEVICES=0 "$PY" -c "
import torch, transformers, av, PIL
print('  torch', torch.__version__, 'cuda avail', torch.cuda.is_available())
print('  transformers', transformers.__version__)
print('  av', av.library_versions['libavformat'])
from transformers import Qwen2_5OmniForConditionalGeneration
print('  Qwen2_5OmniForConditionalGeneration import OK (native, no trust_remote_code)')
" 2>&1 | tee "$OUT/env_check_${SLURM_JOB_ID:-local}.log"

echo "[job] launching async_omni_v2/run.py (Qwen2.5-Omni-7B, audio ON, TMRoPE ON), GPU 0, 12s clip"
cd "$REPO/async_omni_v2"
CUDA_VISIBLE_DEVICES=0 "$PY" -u run.py \
  --video_path "$VIDEO" \
  --backend qwen2_5_omni \
  --use_audio \
  --tmrope_positions \
  --max_seconds 12 \
  --deterministic \
  --profile \
  > "$OUT/smoke_${SLURM_JOB_ID:-local}.log" 2>&1
rc=$?
echo "[job] run.py exit code: $rc"

echo; echo "===== KEY LINES ====="
LOG="$OUT/smoke_${SLURM_JOB_ID:-local}.log"
echo "--- backend load ---";        grep -m5 "\[backend\]"    "$LOG"
echo "--- ingester seed/audio-on ---"; grep -m5 "] ingester "  "$LOG"
echo "--- audio chunks ingested ---";  grep "ingester.audio "  "$LOG" | head -10
echo "--- any WARNINGs (RoPE-horizon guard, audio_io env/file issues) ---"
grep -i "WARNING" "$LOG" || echo "  (none -- clean run)"
echo "--- controller ticks ---";    grep -c "] ctrl.gate "     "$LOG"
echo "--- any fires? ---";          grep "] ctrl.gate " "$LOG" | grep "fire=True" | head -5
echo "--- any Traceback? ---";      grep -c "Traceback"        "$LOG"
echo "--- tail of raw log ---";     tail -40 "$LOG"
echo "[job] DONE rc=$rc"
exit $rc
