#!/usr/bin/env bash
# ab_inplace.sh — END-TO-END proof that dropping the KV-cache snapshot changes
#                 nothing except memory and time.
#
# test_inplace.py proves the MECHANISM (borrow == snapshot, bit for bit, on a
# synthetic cache). This proves the SYSTEM: run the real pipeline over real
# videos twice -- once with controller_cache_mode=snapshot, once inplace -- and
# diff everything the controller produced.
#
# What must match, and why each one is checked separately:
#   ctrl.raw      the emitted control JSON, every tick. If these differ the model
#                 saw a different cache -- the strongest possible signal.
#   ctrl.gate     p_hit / p_more / fire / level. The gate consumes p_hit, so an
#                 identical raw with a drifting p_hit would still change emissions.
#   CONTROLLER    the actual emissions (time + text) that get scored.
#   online_pred   the scored artefact, minus per-run wall-clock fields.
# Timing fields (gen=, wall_s) are EXPECTED to differ and are stripped before
# diffing -- that difference is the point of the change.
#
# Usage:
#   bash ab_inplace.sh [SPLIT_JSON] [OUT_DIR]
# Defaults to splits_ab/ab.json, which make_ab_split.py writes.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SPLIT="${1:-$HERE/splits_ab/ab.json}"
OUT="${2:-$HERE/output_ab_inplace}"
# seconds of each clip to run. Both arms get the same cap, so truncating costs
# nothing in validity -- it only bounds how long the proof takes.
MAXS="${MAXS:-0}"
DS="${OMNIPRO_DATASET_DIR:-/iopsstor/scratch/cscs/dbartaula/omnipro_data}"
PY="${PY:-/iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env/bin/python}"

export HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/dbartaula/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# deterministic cuBLAS must be set BEFORE the first cuBLAS handle exists, or the
# two arms are not bit-comparable and this whole test proves nothing.
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export OMNIPRO_GATE_MODE=controller
export OMNIPRO_DECODE_MODE=schema
export PROSYNC_DIR="${PROSYNC_DIR:-$(cd "$HERE/../async_omni_v2" && pwd)}"

[ -f "$SPLIT" ] || { echo "no split at $SPLIT — run: $PY make_ab_split.py"; exit 1; }
echo "[ab] split=$SPLIT  dataset=$DS  out=$OUT"
echo "[ab] samples: $($PY -c "import json,sys;print(len(json.load(open('$SPLIT'))))")"

# THREE arms, not two. `snapshot2` is a NULL CONTROL: the same mode run twice.
# Without it a divergence is unattributable -- if the pipeline is not bit-
# reproducible run-to-run, snapshot-vs-inplace differing proves nothing about
# inplace. snapshot-vs-snapshot2 IS the noise floor, and the inplace result can
# only be read against it. MISSION §6 claims lockstep is bit-reproducible; this
# arm is what actually tests that claim.
for MODE in snapshot snapshot2 inplace; do
  D="$OUT/$MODE"
  ARM="${MODE%2}"                      # snapshot2 -> snapshot
  # NEVER silently reuse a previous arm: a stale directory is how a killed run
  # leaks into a comparison and produces a confident wrong answer. Refuse.
  if [ -e "$D" ]; then
    echo "[ab] FATAL: $D already exists. Refusing to reuse it -- delete it and"
    echo "[ab]        re-run, or the diff may compare against a stale arm."
    exit 1
  fi
  mkdir -p "$D"
  echo "[ab] ===== running arm: $MODE (cache_mode=$ARM) ====="
  OMNIPRO_CACHE_MODE=$ARM \
  OMNIPRO_DATASET_DIR="$DS" \
  OMNIPRO_BENCHMARK_JSON="$SPLIT" \
  OMNIPRO_OUTPUT_DIR="$D" \
  PYTHONPATH="$PROSYNC_DIR:$HERE" \
    "$PY" evaluate.py --audio all --no_resume --max_seconds "$MAXS" \
      --benchmark_json "$SPLIT" --dataset_dir "$DS" --out "$D" \
      > "$D/launch.log" 2>&1
  echo "[ab] $MODE exit=$? -> $D"
  # evaluate.py logs to stdout, which lands in launch.log; run_fast.sh would have
  # made a run_<stamp>.log. Check whichever exists.
  grep -hm1 "cache_mode=" "$D"/launch.log "$D"/run_*.log 2>/dev/null \
    || echo "[ab] WARNING: no cache_mode line in $D!"
done

echo
echo "[ab] ========== NULL CONTROL: snapshot vs snapshot =========="
echo "[ab] Any difference here is run-to-run noise, NOT caused by inplace."
"$PY" ab_inplace_diff.py "$OUT/snapshot" "$OUT/snapshot2"
CTRL_RC=$?

echo
echo "[ab] ========== THE ACTUAL TEST: snapshot vs inplace =========="
"$PY" ab_inplace_diff.py "$OUT/snapshot" "$OUT/inplace"
TEST_RC=$?

echo
echo "[ab] ========== VERDICT =========="
if [ $CTRL_RC -ne 0 ]; then
  echo "[ab] The NULL CONTROL diverged: this pipeline is NOT bit-reproducible"
  echo "[ab] run-to-run, so this harness CANNOT prove or disprove inplace"
  echo "[ab] equivalence. Fix reproducibility first. (test_inplace.py still holds:"
  echo "[ab] it compares both modes inside ONE process, where the model is fixed.)"
elif [ $TEST_RC -ne 0 ]; then
  echo "[ab] Control is clean but inplace diverged => the divergence IS inplace."
else
  echo "[ab] Control clean AND inplace identical => inplace is equivalent."
fi
