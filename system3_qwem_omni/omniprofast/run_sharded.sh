#!/bin/bash
# run_sharded.sh — shard the OmniPro × system_5 sweep across local GPUs.
# Each shard runs the ONLINE phase then the PROBE phase on one GPU; a final
# global MERGE (with wandb) aggregates all shards. Resumable: re-run to continue.
#
# Env knobs (with defaults):
#   NSHARDS  number of GPUs/shards          (default 4)
#   TASKS    comma-sep OmniPro tasks         (default: all 9)
#   AUDIO    none|helpful|required|none_helpful|all   (default none_helpful)
#   LIMIT    samples per task, 0=all         (default 0)
#   MAXS     max_seconds per video, 0=full   (default 0)
#   RUNNAME  wandb run-name prefix
set -u
# --- load centralized config from repo .env (env python, HF cache, datasets: all in $SCRATCH) ---
__d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; while [ "$__d" != "/" ] && [ ! -f "$__d/.env" ]; do __d="$(dirname "$__d")"; done
[ -f "$__d/.env" ] && { set -a; . "$__d/.env"; set +a; }
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# system5 (Qwen3-VL) lives in iopsstor hf_cache; system4 (Qwen2.5-Omni) in capstor.
SYSTEM=${SYSTEM:-system5}
if [ "$SYSTEM" = "system4" ]; then
  export HF_HOME=${HF_HOME:-/capstor/scratch/cscs/dbartaula/hf_cache}
  OUT=${OUT:-output_system4}
  RESULTS_NAME=${RESULTS_NAME:-result_system4.md}
  AUDIO=${AUDIO:-all}            # system4 has audio: full dataset is fair
else
  export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}
  OUT=${OUT:-output}
  RESULTS_NAME=${RESULTS_NAME:-results.md}
  AUDIO=${AUDIO:-none_helpful}   # vision-only: required samples excluded
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_SILENT=true

NSHARDS=${NSHARDS:-4}
TASKS=${TASKS:-instant_event_alert,semantic_condition_alert,explicit_target_grounding,snapshot_counting,cumulative_counting,dedup_counting,realtime_state_monitor,event_narration,sequential_step_instruction}
LIMIT=${LIMIT:-0}
MAXS=${MAXS:-0}
RUNNAME=${RUNNAME:-$SYSTEM-$(date +%m%d-%H%M)}

COMMON="--system $SYSTEM --results_name $RESULTS_NAME --out $OUT \
        --tasks $TASKS --audio $AUDIO --limit $LIMIT --max_seconds $MAXS --run_name $RUNNAME"

echo "[launcher] NSHARDS=$NSHARDS AUDIO=$AUDIO LIMIT=$LIMIT MAXS=$MAXS"
pids=()
for k in $(seq 0 $((NSHARDS-1))); do
  ( CUDA_VISIBLE_DEVICES=$k python evaluate.py --mode online --shard $k --nshards $NSHARDS $COMMON --no_wandb \
    && CUDA_VISIBLE_DEVICES=$k python evaluate.py --mode probe  --shard $k --nshards $NSHARDS $COMMON --no_wandb \
  ) > "output/shard_${k}.log" 2>&1 &
  pids+=($!)
  echo "[launcher] shard $k -> GPU $k (pid ${pids[-1]}, log output/shard_${k}.log)"
done

echo "[launcher] waiting for ${#pids[@]} shards ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done

echo "[launcher] all shards done (fail=$fail); global MERGE + wandb ..."
python evaluate.py --mode merge $COMMON
echo "[launcher] DONE -> output/results.md , output/summary.json"
