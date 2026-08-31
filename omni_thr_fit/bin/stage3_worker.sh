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

# Resolve this lane's ARM. Unset $S3_ARM gives results/full2700 with no override,
# which is what this script did before arms existed.
eval "$("$PY" "$THR_ROOT/lib/arms.py")" || exit 2
OUT=$S3_RESULTS/lane$LANE
LOG=$THR_ROOT/logs/stage3_lane_${S3_ARM}_${LANE}.log
mkdir -p "$OUT"

# Judge offline, exactly as in pass 1/2 (RUNBOOK sec.3): a live judge puts a
# network round-trip on the critical path of a bounded window, and the verdicts
# are backfilled from the banked predictions afterwards.
unset GEMINI_API_KEY OPENAI_API_KEY GEMINI_API_BASE OPENAI_API_BASE

# THE ARM AND ITS THRESHOLD SOURCE MUST AGREE, asserted here and not merely
# arranged by the caller. The two arms differ ONLY in a directory and this one
# variable, and every way of getting the pairing wrong produces a run that
# completes and banks well-formed predictions under the other arm's name --
# undetectable downstream, because nothing in a prediction records which
# threshold produced it.
#
#   fitted (override empty): sec.6 requires NO override, so the run exercises
#     config.py's per-task lookup exactly as a deployed system would. A stale
#     export from an interactive shell would silently void the headline number.
#   g015 (override set): the flat global threshold must actually be forced, or
#     the "single global" control is really a second copy of the fitted arm.
if [ -z "${S3_OVERRIDE:-}" ]; then
  if [ -n "${OMNIPRO_HIT_THRESHOLD:-}" ]; then
    echo "[lane $LANE] ABORT arm=$S3_ARM: OMNIPRO_HIT_THRESHOLD=$OMNIPRO_HIT_THRESHOLD is set but this arm forbids an override" >> "$LOG"
    exit 2
  fi
else
  export OMNIPRO_HIT_THRESHOLD="$S3_OVERRIDE"
  if [ "${OMNIPRO_HIT_THRESHOLD}" != "$S3_OVERRIDE" ]; then
    echo "[lane $LANE] ABORT arm=$S3_ARM: override did not take" >> "$LOG"
    exit 2
  fi
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
    --done_glob "$S3_RESULTS/lane*/online_pred.jsonl" \
    >> "$OUT/run.log" 2>&1
echo "[lane $LANE] exit rc=$? $(date +%H:%M:%S)" >> "$LOG"
