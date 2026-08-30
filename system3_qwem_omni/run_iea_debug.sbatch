#!/bin/bash
#SBATCH --job-name=iea_eval
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --array=1-3
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_iea_debug/slurm_%A_%a.log
# instant_event_alert ICL (controller) eval on the debug partition.
# The debug time limit is 1:30, and one full 17-video controller run is ~3h, so we
# SPLIT the 17 videos across the node's 4 GPUs (4 balanced subsets in iea_split/)
# and run them in PARALLEL -> wall-clock ~= the heaviest single-GPU group (~65 min).
# --array=1-3 gives 3 independent runs (different nodes) so we can average (the eval
# is non-deterministic; report the mean, not one run).
set -uo pipefail

PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data     # dataset dir that HAS all 17 IEA videos
SPLIT=$REPO/omniprofast/iea_split
R=${SLURM_ARRAY_TASK_ID:-1}

echo "[job] host=$(hostname) run=$R CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

# env sanity (warn only; don't auto-rebuild)
"$PY" -c "import torch,transformers,av,PIL; from google import genai; print('[job] env OK', torch.__version__, 'cuda:', torch.cuda.is_available())" \
  || echo "[job] WARNING: env import check failed — run may error"

cd "$REPO/omniprofast"
mkdir -p "$REPO/omniprofast/output_iea_debug"

# 4 parallel eval processes, one per GPU, each on its video subset
for k in 0 1 2 3; do
  echo "[job] launching group $k on GPU $k ($(basename $SPLIT/iea_g${k}.json))"
  CUDA_VISIBLE_DEVICES=$k OMNIPRO_GATE_MODE=controller \
    bash run_fast.sh \
      --tasks instant_event_alert \
      --audio none \
      --no_resume \
      --benchmark_json "$SPLIT/iea_g${k}.json" \
      --dataset_dir "$DS" \
      --out "$REPO/omniprofast/output_iea_debug/r${R}_g${k}" \
      > "$REPO/omniprofast/output_iea_debug/r${R}_g${k}_launch.log" 2>&1 &
done
wait
echo "[job] run $R: all 4 groups finished rc=$?"
for k in 0 1 2 3; do
  m="$REPO/omniprofast/output_iea_debug/r${R}_g${k}/online_metrics.json"
  echo "[job]   g$k metrics: $([ -f "$m" ] && echo present || echo MISSING)"
done
echo "[job] DONE run $R"
