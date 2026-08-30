#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --partition=debug
#SBATCH --account=a168
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/dbartaula/system_3/omniprofast/output_smoke_thr/slurm_%j.log
#
# THRESHOLD VALIDATION RUN -- same 12 videos, per-task hit_threshold from auc.py.
#
# Checks, in order of what would waste the most time if wrong:
#   1. Does it run at all with the new MEMORY (seen-trace + already-told-user)?
#   2. Are emits REASONABLE -- not zero (silent), not spamming?
#   3. Are the answers actually correct (judge content_acc)?
#   4. Is it fast enough (gen_s per tick)?
#   5. Do the 6 NEW task prompts work at all? Two of them have NEVER been run.
#
# 4 tasks x 3 shortest videos, one task per GPU. Uses omnipro_data, which holds
# all 932 videos -- the dir run_fast.sh hardcodes has only 34, which is why every
# past eval was silently capped.
set -uo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
DS=/iopsstor/scratch/cscs/dbartaula/omnipro_data
OUT=$REPO/omniprofast/output_smoke
OUT2=$REPO/omniprofast/output_smoke_thr
mkdir -p "$OUT2"

echo "[job] host=$(hostname) jobid=${SLURM_JOB_ID:-none} commit=$(git -C $REPO rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader || true
cd "$REPO/omniprofast"

# 2 proven prompts + 2 NEVER-RUN new ones (dedup_counting is the flagship task:
# 283 of our 932 samples, and only 17 of its 300 need audio).
TASKS=(semantic_condition_alert instant_event_alert dedup_counting realtime_state_monitor)

for k in 0 1 2 3; do
  T=${TASKS[$k]}
  echo "[job] GPU$k -> $T"
  CUDA_VISIBLE_DEVICES=$k \
  OMNIPRO_GATE_MODE=controller \
  OMNIPRO_DECODE_MODE=schema \
    bash run_fast.sh \
      --tasks "$T" \
      --audio none_helpful \
      --limit 3 \
      --shortest \
      --no_resume \
      --dataset_dir "$DS" \
      --out "$OUT2/$T" \
      > "$OUT2/${T}_launch.log" 2>&1 &
done
wait
echo "[job] all 4 smoke arms finished"

PY=/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python

echo; echo "===== 1. RESULTS ====="
for T in "${TASKS[@]}"; do
  f=$(ls -t "$OUT2/$T"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || { echo "[$T] NO LOG"; continue; }
  echo "[$T] $(tail -1 "$f")"
done

echo; echo "===== 2. EMIT SANITY (silent? spamming?) ====="
for T in "${TASKS[@]}"; do
  f=$(ls -t "$OUT2/$T"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  echo "[$T]"; grep -o "video=\S* emits=[0-9]* gt=[0-9]* tp=[0-9]* fp=[0-9]* fn=[0-9]*" "$f" | sed 's/^/    /'
done

echo; echo "===== 3. LATENCY ====="
for T in "${TASKS[@]}"; do
  f=$(ls -t "$OUT2/$T"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  echo -n "[$T] "
  grep -o "gen=[0-9.]*s ntok=[0-9]*" "$f" | "$PY" -c "
import sys,statistics as st
g=[];n=[]
for l in sys.stdin:
    p=l.split(); g.append(float(p[0][4:-1])); n.append(int(p[1][5:]))
print(f'ticks={len(g)} gen_med={st.median(g):.2f}s gen_max={max(g):.2f}s ntok_med={st.median(n):.0f}' if g else 'no ticks')
"
done

echo; echo "===== 4. MEMORY IS BEING FED BACK ====="
for T in "${TASKS[@]}"; do
  f=$(ls -t "$OUT2/$T"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  echo "[$T] distinct 'seen' observations logged: $(grep -o '"seen":"[^"]*"' "$f" | sort -u | wc -l)"
done

echo; echo "===== 5. COMPLIANCE ====="
for T in "${TASKS[@]}"; do
  "$PY" compliance.py "$OUT2/$T" 2>&1 | sed "s|^|[$T] |"
done

echo; echo "===== 6. LOGIT-READ AGREEMENT ====="
for T in "${TASKS[@]}"; do
  f=$(ls -t "$OUT2/$T"/run_*.log 2>/dev/null | head -1)
  [ -n "$f" ] || continue
  echo "[$T] reads=$(grep -c 'agree=' "$f") agree=$(grep -c 'agree=True' "$f")"
done
echo "[job] DONE"
