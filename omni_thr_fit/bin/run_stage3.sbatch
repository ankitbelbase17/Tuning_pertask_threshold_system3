#!/bin/bash
#SBATCH --job-name=s3full
#SBATCH --partition=debug
#SBATCH --nodes=4
#SBATCH --time=00:22:00
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --signal=B:USR1@120
#SBATCH --output=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit/logs/s3full_%j.log
#
# Stage 3 (sec.6): the complete 2,700-sample OmniPro Online eval with the fitted
# thresholds read from config.py -- no OMNIPRO_HIT_THRESHOLD override.
#
# Structurally identical to run_chain.sbatch (same QoS arithmetic, same
# pre-queue-before-work rule, same dead-chain guard) but over a different unit of
# work, so it asks stage3_state.py what remains. The two chains must not run at
# once: debug-qos MaxJobsPU=1. preflight_stage3.sh warns if one is still queued.
#
# NOTE ON THE BAND. sec.6 forbids capping long videos OUT of stage 3, and nothing
# here does: --max_dur only defers a sample to a generation with a longer window,
# and stage3_state.py keeps counting it in REMAIN until it is actually banked. The
# chain steps 4 -> 2 -> 1 nodes as each band empties, ending at the 90-minute
# shape that can clear the 594.6 s tail.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source $THR_ROOT/env.sh

GEN=${1:-1}
MAX_GEN=${2:-600}
# The ARM is positional, NOT inherited from the submitting environment. A chain
# re-submits itself for a day; an env var set at the first submit is one dropped
# --export away from vanishing, and the fallback would be "fitted" -- which finds
# the fitted arm already complete and stands the global arm down looking finished.
# Same failure the pass-1/2 chain had with its worklist; same fix.
export S3_ARM=${3:-${S3_ARM:-fitted}}
NODES=${SLURM_JOB_NUM_NODES:-4}
NLANE=$(( NODES * 4 ))

eval "$("$PY" "$THR_ROOT/lib/arms.py")" || exit 1
mkdir -p "$THR_ROOT/logs" "$S3_RESULTS"
echo "[s3] arm=$S3_ARM ($S3_LABEL) -> $S3_RESULTS override=${S3_OVERRIDE:-<none>}"

# THE OVERRIDE MUST NOT SURVIVE INTO THE CHAIN from an interactive shell. A
# generation inherits the submitting environment through sbatch, so a stale
# `export OMNIPRO_HIT_THRESHOLD=0.5` from debugging pass 1 would propagate to
# every generation and quietly void the number. Cleared here unconditionally;
# the arm that WANTS an override re-exports it per lane in stage3_worker.sh,
# from STAGE3_ARMS.json, where it is paired with the output directory.
unset OMNIPRO_HIT_THRESHOLD

# Same retry-and-keep-moving contract as run_chain.sbatch: chain_state failing
# must never end the chain before the successor is queued.
st () {
  local r=""
  for _try in 1 2 3; do
    # KEEP ONLY KEY=VALUE LINES. stage3_state.py imports the eval module, which
    # logs its banner to stdout -- "[02:19:24] [eval] loaded 2700 samples |
    # tasks=['cumulative_counting', 'dedup_counting', ...]". That text reaches
    # `eval` below, where bash tries to RUN it: gen 1 logged
    # "dedup_counting,: command not found" five times. Harmless there only
    # because the banner happens to contain nothing dangerous -- eval'ing
    # arbitrary log text is not a thing to leave working by luck.
    r=$(timeout 600 "$PY" "$THR_ROOT/lib/stage3_state.py" --nodes "$NODES" \
          2>>"$THR_ROOT/logs/stage3_state.err" | grep -E '^[A-Z_]+=')
    if [ -n "$r" ] && printf '%s' "$r" | grep -q '^COMPLETE='; then
      printf '%s\n' "$r"; return 0
    fi
    sleep 5
  done
  echo "[s3] WARNING stage3_state failed 3x -- assuming work remains" >&2
  printf 'TARGET=0\nDONE=0\nREMAIN=1\nCELLS_TOTAL=1\nCELLS_DONE=0\nCOMPLETE=0\n'
  printf 'LONGEST_REMAIN_S=0.0\nXRT_MEAN=\nXRT_P95=\nREC_NODES=%s\nMAX_DUR=0\n' "$NODES"
  printf 'FIT_N=0\nREMAIN_GPUH=0.0\n'
}

eval "$(st)"
if [ "${COMPLETE:-0}" = "1" ]; then
  echo "[s3] STAGE 3 COMPLETE at gen $GEN -- all $TARGET samples banked"
  exit 0
fi

echo "[s3] gen=$GEN/$MAX_GEN nodes=$NODES lanes=$NLANE"
echo "[s3] progress=$DONE/$TARGET samples remaining=$REMAIN (${REMAIN_GPUH} GPU-h)"
echo "[s3] xRT_mean=${XRT_MEAN:-n/a} p95=${XRT_P95:-n/a} longest_remaining=${LONGEST_REMAIN_S}s"

MYCAP=$(timeout 600 "$PY" "$THR_ROOT/lib/stage3_state.py" --nodes "$NODES" \
          --max_nodes "$NODES" 2>/dev/null | sed -n 's/^MAX_DUR=//p')
export MAX_DUR=${MYCAP:-0}
echo "[s3] window-fit cap for this shape: max_dur=${MAX_DUR}s (0 = take everything)"

CHAINED=0
NEXT_NODES=${REC_NODES:-$NODES}
NEXT_TIME=$(awk -v n="$NEXT_NODES" 'BEGIN{m=int(88/n); printf "%02d:%02d:00", int(m/60), m%60}')
queue_next () {
  [ "$CHAINED" = "1" ] && return 0
  [ "$GEN" -ge "$MAX_GEN" ] && { echo "[s3] MAX_GEN reached"; return 0; }
  for i in 1 2 3 4 5; do
    if timeout 120 sbatch --account="$SLURM_ACCOUNT_USE" \
              --dependency=afterany:"${SLURM_JOB_ID}" \
              --nodes="$NEXT_NODES" --time="$NEXT_TIME" \
              "$THR_ROOT/bin/run_stage3.sbatch" $(( GEN + 1 )) "$MAX_GEN" "$S3_ARM" >/dev/null 2>&1; then
      CHAINED=1
      echo "[s3] gen $(( GEN + 1 )) queued: nodes=$NEXT_NODES time=$NEXT_TIME"
      return 0
    fi
    sleep 10
  done
  echo "[s3] WARNING failed to queue gen $(( GEN + 1 )) -- the external guard must rescue"
}
trap 'echo "[s3] caught USR1 (preemption/wall)"; queue_next' USR1
queue_next

# ---- run the lanes -------------------------------------------------------
# nshards == this generation's lane count. Resume is global (--done_glob over
# every lane file), which is what makes the shard split a work-distribution hint
# rather than an ownership claim -- so the 16 -> 8 -> 4 reshape cannot orphan or
# duplicate a sample.
DEADLINE=$(bash "$THR_ROOT/bin/job_deadline.sh" "$NODES")
srun --export=ALL --ntasks="$NLANE" --ntasks-per-node=4 --gpus-per-task=1 \
     bash "$THR_ROOT/bin/srun_stage3_lane.sh" "$NLANE" "$DEADLINE" "$MAX_DUR" &
while [ -n "$(jobs -pr)" ]; do wait -n 2>/dev/null || true; done

eval "$(st | sed 's/^/E_/')"
echo "[s3] gen $GEN end: $E_DONE/$E_TARGET samples banked, $E_REMAIN left"

# ---- dead-chain guard ----------------------------------------------------
STALL_F=$THR_ROOT/.stall_s3
PREV_DONE=$(cat "$THR_ROOT/.progress_s3" 2>/dev/null || echo -1)
echo "$E_DONE" > "$THR_ROOT/.progress_s3"
if [ "$E_DONE" -le "$PREV_DONE" ] && [ "$PREV_DONE" -ge 0 ]; then
  STALL=$(( $(cat "$STALL_F" 2>/dev/null || echo 0) + 1 ))
  echo "$STALL" > "$STALL_F"
  echo "[s3] WARNING generation banked no new samples (stall $STALL/2)"
  if [ "$STALL" -ge 2 ]; then
    echo "[s3] ABORT: two consecutive generations with zero progress."
    echo "[s3] inspect $THR_ROOT/logs/stage3_lane_*.log -- chain STOPPED."
    DEAD=$(timeout 60 squeue -u "$USER" -h -o "%i %E" 2>/dev/null \
             | awk -v j="$SLURM_JOB_ID" '$2 ~ j {print $1}' | head -1)
    [ -n "$DEAD" ] && timeout 60 scancel "$DEAD" 2>/dev/null
    exit 1
  fi
else
  rm -f "$STALL_F"
fi

# ---- reshape the successor if the surviving tail needs a longer window ----
echo "[s3]   next shape: nodes=${E_REC_NODES:-?} max_dur=${E_MAX_DUR:-?}s fit_n=${E_FIT_N:-?}"
if [ "$CHAINED" = "1" ] && [ -n "${E_REC_NODES:-}" ] && [ "$E_REC_NODES" != "$NEXT_NODES" ]; then
  DEP=$(timeout 60 squeue -u "$USER" -h -o "%i %E" 2>/dev/null \
          | awk -v j="$SLURM_JOB_ID" '$2 ~ j {print $1}' | head -1)
  if [ -n "$DEP" ]; then
    NT=$(awk -v n="$E_REC_NODES" 'BEGIN{m=int(88/n); printf "%02d:%02d:00", int(m/60), m%60}')
    echo "[s3] tail needs a longer window -> reshaping successor to nodes=$E_REC_NODES time=$NT"
    # MaxSubmitPU=2 counts running + pending: cancel BEFORE submitting.
    timeout 60 scancel "$DEP" 2>/dev/null
    timeout 120 sbatch --account="$SLURM_ACCOUNT_USE" \
        --dependency=afterany:"${SLURM_JOB_ID}" --nodes="$E_REC_NODES" --time="$NT" \
        "$THR_ROOT/bin/run_stage3.sbatch" $(( GEN + 1 )) "$MAX_GEN" "$S3_ARM" >/dev/null 2>&1 \
      && echo "[s3] successor reshaped" \
      || echo "[s3] WARNING reshape failed; original successor cancelled"
  fi
fi
echo "[s3] gen $GEN exit $(date)"
