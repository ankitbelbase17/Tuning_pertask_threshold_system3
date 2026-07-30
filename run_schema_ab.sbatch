#!/bin/bash
#SBATCH --job-name=schema_ab
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_schema_ab/slurm_%j.log
#
# PHASE 1.1 GATE — schema-walk decoder vs the legacy free decode, matched A/B.
#
# Same commit, same weights, same videos, same seed; the ONLY difference is
# OMNIPRO_DECODE_MODE. One arm per GPU so all four run concurrently:
#
#   GPU 0  schema  semantic_condition_alert     GPU 1  free  semantic_condition_alert
#   GPU 2  schema  instant_event_alert          GPU 3  free  instant_event_alert
#
# What we are looking for (in priority order):
#   1. AGREEMENT — does the logit-read `hit` match what a free decode would have
#      produced? Logged every tick as agree=/argmax= by verify_logit_read.
#      If these disagree, the whole schema path is built on sand. THIS IS THE GATE.
#   2. COMPLIANCE — valid-JSON rate should go 85.8% -> 100%, and fps/next_check_s
#      should vanish from the default path (was 1.0% / 4.1% compliant).
#   3. LATENCY — ntok and gen= per tick; expect ~35 decodes -> ~6.
#   4. ACCURACY — F1 must not regress. Fewer tokens is worthless if it costs recall.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
OUT=$REPO/omniprofast/output_schema_ab
mkdir -p "$OUT"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

cd "$REPO/omniprofast"

# 3 shortest videos per task per arm: enough ticks for the agreement + compliance
# statistics to converge, small enough to finish well inside the 1:30 debug limit.
LIMIT=3

launch () {  # $1=gpu  $2=schema|free  $3=task
  echo "[job] GPU$1  decode_mode=$2  task=$3"
  CUDA_VISIBLE_DEVICES=$1 \
  OMNIPRO_GATE_MODE=controller \
  OMNIPRO_DECODE_MODE=$2 \
    bash run_fast.sh \
      --tasks "$3" \
      --audio none \
      --limit $LIMIT \
      --shortest \
      --no_resume \
      --out "$OUT/${3}_${2}" \
      > "$OUT/${3}_${2}_launch.log" 2>&1 &
}

launch 0 schema semantic_condition_alert
launch 1 free   semantic_condition_alert
launch 2 schema instant_event_alert
launch 3 free   instant_event_alert

wait
echo "[job] all four arms finished"

for t in semantic_condition_alert instant_event_alert; do
  for m in schema free; do
    d="$OUT/${t}_${m}"
    echo "[job] $t/$m metrics: $([ -f "$d/online_metrics.json" ] && echo present || echo MISSING)"
  done
done

# Compliance is computable straight from the logs, no GPU and no judge needed.
PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
echo; echo "===== COMPLIANCE ====="
for t in semantic_condition_alert instant_event_alert; do
  for m in schema free; do
    "$PY" compliance.py "$OUT/${t}_${m}" 2>&1 | sed "s|^|[${t:0:3}/$m] |"
  done
done

echo; echo "===== LOGIT-READ AGREEMENT (the Phase 1.1 gate) ====="
for t in semantic_condition_alert instant_event_alert; do
  f=$(ls -t "$OUT/${t}_schema"/*.log 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "[$t] no log"; continue; }
  tot=$(grep -c "agree=" "$f" 2>/dev/null || echo 0)
  agr=$(grep -c "agree=True" "$f" 2>/dev/null || echo 0)
  nb=$(grep -o "argmax='[^']*'" "$f" 2>/dev/null | grep -cv "true\|false" || echo 0)
  echo "[$t] ticks_with_read=$tot  agree=$agr  argmax_not_a_bool=$nb"
done
echo "[job] DONE"
