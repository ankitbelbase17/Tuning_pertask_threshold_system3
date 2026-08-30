#!/usr/bin/env bash
# worker.sh -- one lane = one GPU. Pulls work units until its window runs out.
#
# Used UNCHANGED by both fleets: the debug-partition chain (SLURM_PROCID picks the
# lane, 22-minute window) and the ln003 login chain (lane passed on the command
# line, killable at any instant). The only difference is who computes DEADLINE.
#
# Usage: worker.sh <lane> <deadline_epoch> [max_dur]
set -uo pipefail
source "${THR_ROOT:-/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit}/env.sh"

LANE=${1:?lane}
DEADLINE=${2:?deadline epoch}
MAX_DUR=${3:-0}
# Which pass is ACTIVE comes from a FILE (see chain_state.default_worklist).
# Both fleets and every lane must agree on this, and a lane inherits its
# environment from whichever chain generation happened to spawn it -- so the file
# is the single source of truth and the env var is only an override for one-offs.
#   echo worklist_p2.tsv > $THR_ROOT/ACTIVE_WORKLIST
WORK=${WORKLIST:-}
if [ -z "$WORK" ]; then
  WORK=$(cat "$THR_ROOT/ACTIVE_WORKLIST" 2>/dev/null | tr -d ' \n')
  WORK=${WORK:-worklist_p1.tsv}
  case "$WORK" in /*) ;; *) WORK=$THR_ROOT/$WORK ;; esac
fi
LOG=$THR_ROOT/logs/lane_${LANE}.log

# NO IN-LOOP JUDGING (RUNBOOK.md sec.3). evaluate.py scores each sample as it
# finishes, so a live ContentJudge puts a network round-trip on the critical path
# of a 22-minute window -- and Gemini is unusable on this key. Unsetting the keys
# makes the judge report "unavailable" and return None instantly: content metrics
# are WITHHELD, never guessed; predictions stay on disk; the offline judge fills
# verdicts in afterwards. Selection runs on time_f1, which never touches a judge.
unset GEMINI_API_KEY OPENAI_API_KEY GEMINI_API_BASE OPENAI_API_BASE

# The shortest sample in the frozen split is ~24 s of video; at the measured omni
# p95 xRT of 3.34 that is ~80 s, plus the model load already paid. Do not START a
# unit this window cannot plausibly bank: a unit begun and SIGKILLed banks nothing
# and simply gets re-claimed, so the time is pure loss.
MIN_SLICE=${MIN_SLICE:-240}

cd "$REPO/omniprofast" || exit 2
echo "[lane $LANE] $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} start=$(date +%H:%M:%S)" \
     "deadline=$(date -d @"$DEADLINE" +%H:%M:%S) max_dur=$MAX_DUR" >> "$LOG"

banked=0
# Units this lane has already attempted THIS GENERATION and banked nothing from.
# Without this a lane can re-claim the same barren shard for the whole window:
# next_unit returns the first incomplete claimable unit in worklist order, and a
# shard whose samples are all longer than $MAX_DUR is incomplete forever. On
# 2026-08-28 that produced 26,277 claims across seven generations, every one for
# instant_event_alert, while 84 of 94 cells were never touched. The scheduler now
# filters barren CELLS by --max_dur; this filters barren SHARDS of a live cell,
# which the cell-level check cannot see.
SKIP=""
while :; do
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt "$MIN_SLICE" ]; then
    echo "[lane $LANE] ${left}s left (< $MIN_SLICE) -- stopping cleanly" >> "$LOG"
    break
  fi

  UNIT=$(timeout 120 "$PY" "$THR_ROOT/lib/claim_cli.py" next \
           --worklist "$WORK" --lane "$LANE" \
           --max_dur "$MAX_DUR" --skip "$SKIP" 2>>"$LOG")
  rc=$?
  if [ "$rc" = "3" ] || [ -z "${UNIT:-}" ]; then
    echo "[lane $LANE] no claimable units left" >> "$LOG"
    break
  fi
  IFS=$'\t' read -r PASS TASK THR SH <<<"$UNIT"
  CELL=$THR_ROOT/results/$PASS/$TASK/thr_$THR
  mkdir -p "$CELL/lane$SH"
  BEFORE=$(cat "$CELL"/lane*/online_pred.jsonl 2>/dev/null | wc -l)
  echo "[lane $LANE] --> $PASS $TASK thr=$THR shard=$SH ($(date +%H:%M:%S), ${left}s left)" >> "$LOG"

  # Keep the claim fresh so a long unit is not stolen by another lane's TTL sweep.
  # Invoked via `bash`, not directly: this tree is created without the execute bit
  # and a non-executable heartbeat fails silently into "Permission denied", which
  # costs nothing visible now and lets a long unit's claim go stale at 30 min.
  bash "$THR_ROOT/bin/heartbeat.sh" "$UNIT" & HB=$!

  # THE SWEEP KNOB. OMNIPRO_HIT_THRESHOLD forces one value across all tasks, which
  # IS a per-task knob because the run is restricted to a single --tasks. With the
  # fitted gate wired (system5_adapter.py) this threshold is now the actual firing
  # rail, not merely the answer-decode branch it used to be.
  OMNIPRO_HIT_THRESHOLD=$THR \
  timeout $(( left - 30 )) "$PY" -u evaluate.py \
      --tasks "$TASK" --audio all \
      --subset_every 20 --subset_off 0 \
      --benchmark_json "$OMNIPRO_BENCHMARK_JSON" --dataset_dir "$OMNIPRO_DATASET_DIR" \
      --shard "$SH" --nshards 4 --max_dur "$MAX_DUR" --resume \
      --out "$CELL/lane$SH" \
      --done_glob "$CELL/lane*/online_pred.jsonl" >> "$CELL/lane$SH/run.log" 2>&1
  erc=$?
  kill "$HB" 2>/dev/null
  timeout 60 "$PY" "$THR_ROOT/lib/claim_cli.py" release --unit "$UNIT" 2>/dev/null
  banked=$(( banked + 1 ))

  # Did this unit actually bank anything? Count the cell's rows before and after
  # rather than trusting the exit code: evaluate.py exits 0 both when it finishes
  # samples and when --max_dur filtered every one of them away.
  AFTER=$(cat "$CELL"/lane*/online_pred.jsonl 2>/dev/null | wc -l)
  if [ "$AFTER" -le "$BEFORE" ]; then
    SKIP="${SKIP:+$SKIP,}$PASS:$TASK:$THR:$SH"
    echo "[lane $LANE] <-- $TASK thr=$THR shard=$SH rc=$erc BARREN (nothing under ${MAX_DUR}s) ($(date +%H:%M:%S))" >> "$LOG"
  else
    echo "[lane $LANE] <-- $TASK thr=$THR shard=$SH rc=$erc +$(( AFTER - BEFORE )) samples ($(date +%H:%M:%S))" >> "$LOG"
  fi
done
echo "[lane $LANE] done, $banked units attempted $(date +%H:%M:%S)" >> "$LOG"
