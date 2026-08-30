#!/bin/bash
#SBATCH --job-name=thrfit
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --time=00:22:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --signal=B:USR1@120
#SBATCH --output=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit/logs/thrfit_%j.log
#
# Self-chaining threshold-fit sweep on the debug partition.
#
# WHY CHAINED. Verified on this cluster 2026-08-28:
#   sacctmgr show qos debug-qos -> MaxTRESMins node=90, MaxTRES node=4,
#                                  MaxJobsPU=1, MaxSubmitPU=2
# 4 nodes therefore buys a 22-minute window, one job runs at a time and exactly
# one dependent may be queued. The sweep needs ~145 GPU-h, so the job re-submits
# itself until the worklist is empty.
#
# WHY THE SHAPE IS RECOMPUTED, NOT PASSED. Each generation asks chain_state.py
# what is actually finished on disk. A generation that dies, is preempted, or
# empties a duration band mid-window cannot leave the chain pointing at a shape
# that can no longer clear the tail.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source $THR_ROOT/env.sh

GEN=${1:-1}
MAX_GEN=${2:-400}
NODES=${SLURM_JOB_NUM_NODES:-4}
NLANE=$(( NODES * 4 ))
mkdir -p "$THR_ROOT/logs"

# chain_state must never be able to kill the chain. A transient filesystem hiccup
# used to be fatal upstream: the traceback goes to stderr, $() yields "", eval
# sets nothing, and `set -u` aborts the script BEFORE the successor is queued --
# so one bad read at 3am silently ends a multi-day run. Retry, then fall back to
# values that keep the chain moving (assume work remains, keep the current shape).
st () {
  local r=""
  for _try in 1 2 3; do
    r=$(timeout 300 "$PY" "$THR_ROOT/lib/chain_state.py" --nodes "$NODES" \
          2>>"$THR_ROOT/logs/chain_state.err")
    if [ -n "$r" ] && printf '%s' "$r" | grep -q '^COMPLETE='; then
      printf '%s\n' "$r"; return 0
    fi
    sleep 5
  done
  echo "[chain] WARNING chain_state failed 3x -- assuming work remains" >&2
  printf 'TARGET=0\nDONE=0\nREMAIN=1\nCELLS_TOTAL=0\nCELLS_DONE=0\nCOMPLETE=0\n'
  printf 'LONGEST_REMAIN_S=0.0\nXRT_MEAN=\nXRT_P95=\nREC_NODES=%s\nMAX_DUR=0\n' "$NODES"
  printf 'FIT_N=0\nREMAIN_GPUH=0.0\n'
}

eval "$(st)"
if [ "${COMPLETE:-0}" = "1" ]; then
  echo "[chain] WORKLIST COMPLETE at gen $GEN -- nothing to do"
  exit 0
fi

echo "[chain] gen=$GEN/$MAX_GEN nodes=$NODES lanes=$NLANE"
echo "[chain] progress=$DONE/$TARGET samples, cells=$CELLS_DONE/$CELLS_TOTAL"
echo "[chain] xRT_mean=${XRT_MEAN:-n/a} p95=${XRT_P95:-n/a} longest_remaining=${LONGEST_REMAIN_S}s rec_nodes=$REC_NODES"

# ---- WINDOW-FIT CAP for THIS generation ----------------------------------
# chain_state's MAX_DUR is computed for the shape it RECOMMENDS. This job is
# already running at $NODES, which may differ, so re-ask for this shape and use
# that answer. Handing a 22-min/4-node generation a 595 s video is exactly the
# retry-forever failure the band selection exists to prevent.
MYCAP=$(timeout 300 "$PY" "$THR_ROOT/lib/chain_state.py" --nodes "$NODES" \
          --max_nodes "$NODES" 2>/dev/null | sed -n 's/^MAX_DUR=//p')
export MAX_DUR=${MYCAP:-0}
echo "[chain] window-fit cap for this shape: max_dur=${MAX_DUR}s"

# ---- pre-queue the successor BEFORE any work -----------------------------
# afterany, so the SIGKILL at the wall clock cannot break the chain: nothing at
# the END of this script is guaranteed to run. MaxSubmitPU=2 is why exactly one
# dependent is queued at a time.
CHAINED=0
NEXT_NODES=${REC_NODES:-$NODES}
NEXT_TIME=$(awk -v n="$NEXT_NODES" 'BEGIN{m=int(88/n); printf "%02d:%02d:00", int(m/60), m%60}')
queue_next () {
  [ "$CHAINED" = "1" ] && return 0
  [ "$GEN" -ge "$MAX_GEN" ] && { echo "[chain] MAX_GEN reached"; return 0; }
  for i in 1 2 3 4 5; do
    if timeout 120 sbatch --account="$SLURM_ACCOUNT_USE" \
              --dependency=afterany:"${SLURM_JOB_ID}" \
              --nodes="$NEXT_NODES" --time="$NEXT_TIME" \
              "$THR_ROOT/bin/run_chain.sbatch" $(( GEN + 1 )) "$MAX_GEN" >/dev/null 2>&1; then
      CHAINED=1
      echo "[chain] gen $(( GEN + 1 )) queued: nodes=$NEXT_NODES time=$NEXT_TIME"
      return 0
    fi
    sleep 10
  done
  echo "[chain] WARNING failed to queue gen $(( GEN + 1 )) -- the external guard must rescue"
}
trap 'echo "[chain] caught USR1 (preemption/wall)"; queue_next' USR1
queue_next

# ---- run the lanes -------------------------------------------------------
# srun is mandatory: the batch body runs on the head node only.
# Lanes 0..NLANE-1 are the debug fleet; the ln003 login chain uses lane ids
# 100+, so the two fleets never collide in the log namespace.
# The real wall, not a hardcoded 22 min: this generation may be running at 2 or 1
# node (44 / 88 min) after a reshape, and stopping at 22 would discard the extra
# window the reshape existed to obtain. See bin/job_deadline.sh.
DEADLINE=$(bash "$THR_ROOT/bin/job_deadline.sh" "$NODES")
srun --export=ALL --ntasks="$NLANE" --ntasks-per-node=4 --gpus-per-task=1 \
     bash "$THR_ROOT/bin/srun_lane.sh" "$DEADLINE" "$MAX_DUR" &
while [ -n "$(jobs -pr)" ]; do wait -n 2>/dev/null || true; done

# ---- report progress; reshape the queued successor if the tail demands it --
eval "$(st | sed 's/^/E_/')"
echo "[chain] gen $GEN end: $E_DONE/$E_TARGET samples, cells $E_CELLS_DONE/$E_CELLS_TOTAL"

# ---- DEAD-CHAIN GUARD ----------------------------------------------------
# A chain that re-submits forever while completing nothing burns the whole
# allocation silently. Two consecutive generations banking ZERO new samples means
# the run is broken (model fails to load, dataset unreadable, or the surviving
# tail cannot fit even a 90-minute window), so stop and let a human look.
STALL_F=$THR_ROOT/.stall
PREV_DONE=$(cat "$THR_ROOT/.progress" 2>/dev/null || echo -1)
echo "$E_DONE" > "$THR_ROOT/.progress"
if [ "$E_DONE" -le "$PREV_DONE" ] && [ "$PREV_DONE" -ge 0 ]; then
  STALL=$(( $(cat "$STALL_F" 2>/dev/null || echo 0) + 1 ))
  echo "$STALL" > "$STALL_F"
  echo "[chain] WARNING generation banked no new samples (stall $STALL/2)"
  if [ "$STALL" -ge 2 ]; then
    echo "[chain] ABORT: two consecutive generations with zero progress."
    echo "[chain] inspect $THR_ROOT/logs/lane_*.log -- chain STOPPED, successor cancelled."
    DEAD=$(timeout 60 squeue -u "$USER" -h -o "%i %E" 2>/dev/null \
             | awk -v j="$SLURM_JOB_ID" '$2 ~ j {print $1}' | head -1)
    [ -n "$DEAD" ] && timeout 60 scancel "$DEAD" 2>/dev/null
    exit 1
  fi
else
  rm -f "$STALL_F"
fi

echo "[chain]   next shape: nodes=${E_REC_NODES:-?} max_dur=${E_MAX_DUR:-?}s fit_n=${E_FIT_N:-?}"
if [ "$CHAINED" = "1" ] && [ -n "${E_REC_NODES:-}" ] && [ "$E_REC_NODES" != "$NEXT_NODES" ]; then
  DEP=$(timeout 60 squeue -u "$USER" -h -o "%i %E" 2>/dev/null \
          | awk -v j="$SLURM_JOB_ID" '$2 ~ j {print $1}' | head -1)
  if [ -n "$DEP" ]; then
    NT=$(awk -v n="$E_REC_NODES" 'BEGIN{m=int(88/n); printf "%02d:%02d:00", int(m/60), m%60}')
    echo "[chain] tail needs a longer window -> reshaping successor to nodes=$E_REC_NODES time=$NT"
    # MaxSubmitPU=2 counts running + pending, so CANCEL BEFORE submitting the
    # replacement, never after.
    timeout 60 scancel "$DEP" 2>/dev/null
    timeout 120 sbatch --account="$SLURM_ACCOUNT_USE" \
        --dependency=afterany:"${SLURM_JOB_ID}" --nodes="$E_REC_NODES" --time="$NT" \
        "$THR_ROOT/bin/run_chain.sbatch" $(( GEN + 1 )) "$MAX_GEN" >/dev/null 2>&1 \
      && echo "[chain] successor reshaped" \
      || echo "[chain] WARNING reshape failed; original successor cancelled"
  fi
fi
echo "[chain] gen $GEN exit $(date)"
