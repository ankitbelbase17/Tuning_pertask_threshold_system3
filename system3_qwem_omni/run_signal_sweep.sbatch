#!/bin/bash
#SBATCH --job-name=sigsweep
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_sigsweep/slurm_%j.log
#
# SIGNAL-QUALITY SWEEP.
#
# The gate is NOT the bottleneck. gates.py screened 15 firing strategies offline
# over saved p_hit and the ceiling barely moved (best pooled F1 0.286, recall
# 0.195 -- 33 of 41 events missed). AUC 0.53-0.70 says p_hit only weakly
# separates event-moments from the rest. No thresholding scheme fixes a weak
# signal, so this sweep attacks what SHAPES p_hit instead.
#
# Unlike gate strategies, every variant here changes the forward pass, so none
# can be screened offline -- they need GPU.
#
# VARIANTS (one variable each; run on the SAME 3 videos per task):
#   A  seen_mode=before   (control -- current behaviour, ~1.3 s/tick)
#   B  seen_mode=off      read the level immediately, ZERO decodes (~0.15 s/tick)
#                         -> if this holds up, the always-on trigger is affordable
#                            and ROADMAP Priority 1 unblocks
#   C  seen_mode=after    level first, then describe
#                         -> separates "perception step helps" from "its ORDER helps"
#   D  seen tokens 6      (vs the default 12) -- is a shorter observation enough?
#
# A/B/C together answer the Priority-1 question directly. Compare with auc.py
# (AUC is threshold-free, so it measures SIGNAL quality rather than gate tuning).
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
OUT=$REPO/omniprofast/output_sigsweep
mkdir -p "$OUT"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
cd "$REPO/omniprofast"

# 2 tasks with the strongest signal so far (dedup 0.699 AUC, state_monitor 0.636)
# -- a weak-signal task cannot show a difference between variants.
TASKS=(dedup_counting realtime_state_monitor)
run () {   # $1=gpu  $2=variant  $3=SEEN_MODE  $4=SEEN_TOKENS  $5=task
  echo "[job] GPU$1  $2  seen_mode=$3 seen_tokens=$4  task=$5"
  CUDA_VISIBLE_DEVICES=$1 \
  OMNIPRO_GATE_MODE=controller OMNIPRO_DECODE_MODE=schema \
  OMNIPRO_SEEN_MODE=$3 OMNIPRO_SEEN_TOKENS=$4 \
    bash run_fast.sh --tasks "$5" --audio none_helpful --limit 3 --shortest \
      --no_resume --dataset_dir "$DS" --out "$OUT/${5}__$2" \
      > "$OUT/${5}__$2_launch.log" 2>&1
}

# 4 variants x 2 tasks = 8 runs; 4 GPUs, 2 sequential rounds (~15 min each)
for T in "${TASKS[@]}"; do
  run 0 A_before "before" 12 "$T" &
  run 1 B_off    "off"    12 "$T" &
  run 2 C_after  "after"  12 "$T" &
  run 3 D_seen6  "before"  6 "$T" &
  wait
  echo "[job] finished all variants for $T"
done

PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python
echo; echo "===== time_f1 (gate-dependent) ====="
for T in "${TASKS[@]}"; do for V in A_before B_off C_after D_seen6; do
  f=$(ls -t "$OUT/${T}__$V"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] && echo "[$T/$V] $(tail -1 "$f")" || echo "[$T/$V] NO LOG"
done; done

echo; echo "===== AUC (threshold-free -> SIGNAL quality) ====="
for T in "${TASKS[@]}"; do for V in A_before B_off C_after D_seen6; do
  echo -n "[$T/$V] "
  "$PY" auc.py "$OUT/${T}__$V" --task "$T" 2>/dev/null \
    | grep -E "AUC =|AP  =" | tr '\n' ' '; echo
done; done

echo; echo "===== LATENCY (the 10x question) ====="
for T in "${TASKS[@]}"; do for V in A_before B_off C_after D_seen6; do
  f=$(ls -t "$OUT/${T}__$V"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  echo -n "[$T/$V] "
  grep -o "gen=[0-9.]*s ntok=[0-9]*" "$f" | "$PY" -c "
import sys,statistics as st
g=[];n=[]
for l in sys.stdin:
    p=l.split(); g.append(float(p[0][4:-1])); n.append(int(p[1][5:]))
print(f'ticks={len(g)} gen_med={st.median(g):.2f}s ntok_med={st.median(n):.0f}' if g else 'no ticks')
"
done; done
echo "[job] DONE"
