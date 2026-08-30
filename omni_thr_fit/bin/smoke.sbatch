#!/bin/bash
#SBATCH --job-name=thrsmoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit/logs/smoke_%j.log
#
# THRESHOLD_FIT_v2.md sec.10 step 4 on a DEDICATED GPU.
#
# The same smoke on the ln003 login node stalled: CPU ticking, GPU at 0%, no log
# output for 7 minutes. That node carries 18 other users, load average 34, and
# /dev/shm at 89% -- so a stall there does not distinguish "our bug" from "shared
# node". This job runs the identical work on an exclusive debug node, which is
# where the real fleet runs anyway, and so also proves out the SLURM path.
#
# faulthandler is armed: if it stalls HERE too, we get every thread's stack
# instead of having to guess.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source $THR_ROOT/env.sh

echo "[smoke] host=$(hostname) job=${SLURM_JOB_ID:-none} $(date)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo "[smoke] pids.max=$(cat /sys/fs/cgroup/pids.max 2>/dev/null || echo n/a) nproc=$(nproc) OMP=$OMP_NUM_THREADS"

unset GEMINI_API_KEY OPENAI_API_KEY GEMINI_API_BASE OPENAI_API_BASE
cd "$REPO/omniprofast" || exit 2

TASK=${TASK:-semantic_condition_alert}
for THR in 0.05 0.95; do
  D=$THR_ROOT/smoke_debug/thr_$THR
  mkdir -p "$D"
  echo "[smoke] === thr=$THR ==="
  # PYTHONFAULTHANDLER + a watchdog timeout: dump_traceback_later fires at 900 s
  # and prints EVERY thread's stack, so a hang is diagnosable from the log.
  PYTHONFAULTHANDLER=1 OMNIPRO_HIT_THRESHOLD=$THR \
  timeout 1200 "$PY" -u -X faulthandler -c "
import faulthandler, sys, runpy
faulthandler.dump_traceback_later(900, exit=True)
sys.argv = ['evaluate.py',
            '--tasks', '$TASK', '--audio', 'all',
            '--limit', '2', '--shortest', '--max_seconds', '45',
            '--benchmark_json', '$OMNIPRO_BENCHMARK_JSON',
            '--dataset_dir', '$OMNIPRO_DATASET_DIR',
            '--no_resume', '--out', '$D']
runpy.run_path('evaluate.py', run_name='__main__')
" > "$D/run.log" 2>&1
  rc=$?
  echo "[smoke] thr=$THR rc=$rc"
  echo "[smoke]   ticks=$(grep -c 'ctrl.gate' "$D/run.log" 2>/dev/null || echo 0)" \
       "fires=$(grep -c 'fire=True' "$D/run.log" 2>/dev/null || echo 0)" \
       "emits=$(grep -c 'CONTROLLER' "$D/run.log" 2>/dev/null || echo 0)"
  tail -4 "$D/run.log"
done

echo; echo "[smoke] ===== VERDICT ====="
"$PY" - "$THR_ROOT/smoke_debug" <<'PY'
import json, os, sys
root = sys.argv[1]; res = {}
for thr in ("0.05", "0.95"):
    fp = os.path.join(root, f"thr_{thr}", "online_pred.jsonl")
    if not os.path.exists(fp):
        print(f"  thr={thr}  NO PREDICTIONS -- FAILED"); continue
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    res[thr] = sum(len(r.get("predictions", [])) for r in rows)
    xrt = [r["realtime_factor"] for r in rows if r.get("realtime_factor")]
    print(f"  thr={thr}  n_sample={len(rows)}  n_emit={res[thr]}"
          + (f"  xRT={sum(xrt)/len(xrt):.2f}" if xrt else ""))
if len(res) == 2:
    lo, hi = res["0.05"], res["0.95"]
    print(f"\n  emits at 0.05 ({lo}) vs at 0.95 ({hi})")
    print("  THRESHOLD IS LIVE -- safe to launch the sweep" if lo > hi else
          "  *** THRESHOLD INERT OR TIED -- do NOT launch ***")
PY
