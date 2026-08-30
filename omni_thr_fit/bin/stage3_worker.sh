#!/usr/bin/env bash
# stage3_worker.sh -- one lane of the FULL 2,700-sample eval (sec.6).
#
# DIFFERENT FROM worker.sh, deliberately. worker.sh pulls (task, threshold, shard)
# units from a worklist through an atomic claim, because pass 1/2 are 94 small
# independent cells. Stage 3 is ONE run over the whole benchmark, so the work
# split is just the round-robin shard and the claim machinery would add a lock
# with nothing to protect. Resume is still global (--done_glob over every lane),
# which is what makes the 16 -> 8 -> 4 lane reshape safe.
#
# Usage: stage3_worker.sh <lane> <nshards> <deadline_epoch> [max_dur]
set -uo pipefail
source "${THR_ROOT:?}/env.sh"

LANE=${1:?lane}
NSHARDS=${2:?nshards}
DEADLINE=${3:?deadline epoch}
MAX_DUR=${4:-0}

OUT=$THR_ROOT/results/full2700/lane$LANE
LOG=$THR_ROOT/logs/stage3_lane_${LANE}.log
mkdir -p "$OUT"

# Judge offline, exactly as in pass 1/2 (RUNBOOK sec.3): a live judge puts a
# network round-trip on the critical path of a bounded window, and the verdicts
# are backfilled from the banked predictions afterwards.
unset GEMINI_API_KEY OPENAI_API_KEY GEMINI_API_BASE OPENAI_API_BASE

# THE POINT OF STAGE 3 (sec.6): no threshold override. The run must exercise the
# per-task lookup in config.py exactly as a deployed system would, so a stale
# export from an interactive shell would silently invalidate the headline number.
# preflight_stage3.sh asserts this too; belt and braces, because this is the one
# mistake that produces a plausible-looking wrong result.
if [ -n "${OMNIPRO_HIT_THRESHOLD:-}" ]; then
  echo "[lane $LANE] ABORT: OMNIPRO_HIT_THRESHOLD=$OMNIPRO_HIT_THRESHOLD is set" >> "$LOG"
  exit 2
fi

left=$(( DEADLINE - $(date +%s) ))
if [ "$left" -lt "${MIN_SLICE:-240}" ]; then
  echo "[lane $LANE] only ${left}s left -- not starting" >> "$LOG"
  exit 0
fi

cd "$REPO/omniprofast" || exit 2
echo "[lane $LANE] $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} shard=$LANE/$NSHARDS" \
     "max_dur=$MAX_DUR start=$(date +%H:%M:%S) ${left}s left" >> "$LOG"

# No --tasks / --subset_every: the whole benchmark. --audio all: sec.6 wants all
# three strata, and 65% of OmniPro is audio_dependency=required.
timeout $(( left - 30 )) "$PY" -u evaluate.py \
    --audio all \
    --benchmark_json "$OMNIPRO_BENCHMARK_JSON" --dataset_dir "$OMNIPRO_DATASET_DIR" \
    --shard "$LANE" --nshards "$NSHARDS" --max_dur "$MAX_DUR" --resume \
    --out "$OUT" \
    --done_glob "$THR_ROOT/results/full2700/lane*/online_pred.jsonl" \
    >> "$OUT/run.log" 2>&1
echo "[lane $LANE] exit rc=$? $(date +%H:%M:%S)" >> "$LOG"
