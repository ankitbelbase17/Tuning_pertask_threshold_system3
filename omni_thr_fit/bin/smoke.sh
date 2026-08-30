#!/usr/bin/env bash
# smoke.sh -- THRESHOLD_FIT_v2.md sec.10 step 4, and it is not optional:
# one full cell end to end costs minutes and catches a wrong env var before it is
# baked into 376 work units.
#
# Runs ONE task at TWO thresholds on the two shortest frozen samples, on one GPU,
# through the same evaluate.py invocation worker.sh uses. What it proves:
#   1. the backend loads from THIS account's HF cache (dbartaula's is mode 700)
#   2. the fitted gate is actually in force (grep for gate_strategy in the log)
#   3. two different thresholds produce DIFFERENT emission counts -- the whole
#      point of the rewiring, and the thing that was silently false before
#   4. predictions land where score_cells.py expects them
#
# Usage: bash bin/smoke.sh [gpu] [task]
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source "$THR_ROOT/env.sh"

GPU=${1:-3}
TASK=${2:-semantic_condition_alert}
OUT=$THR_ROOT/smoke
unset GEMINI_API_KEY OPENAI_API_KEY GEMINI_API_BASE OPENAI_API_BASE

cd "$REPO/omniprofast" || exit 2
echo "[smoke] task=$TASK gpu=$GPU repo=$REPO"
echo "[smoke] $(nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader -i "$GPU")"

for THR in 0.05 0.95; do
  D=$OUT/thr_$THR
  mkdir -p "$D"
  echo "[smoke] === threshold $THR -> $D ==="
  CUDA_VISIBLE_DEVICES=$GPU OMNIPRO_HIT_THRESHOLD=$THR \
  timeout 3600 "$PY" -u evaluate.py \
      --tasks "$TASK" --audio all \
      --limit 2 --shortest --max_seconds 45 \
      --benchmark_json "$OMNIPRO_BENCHMARK_JSON" --dataset_dir "$OMNIPRO_DATASET_DIR" \
      --no_resume --out "$D" > "$D/run.log" 2>&1
  rc=$?
  n_emit=$("$PY" - "$D/online_pred.jsonl" <<'PY'
import json, sys
try:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError:
    print("NO-FILE"); raise SystemExit
print(sum(len(r.get("predictions", [])) for r in rows), "emits over",
      len(rows), "samples")
PY
)
  echo "[smoke] thr=$THR rc=$rc  $n_emit"
  echo "[smoke]   gate in force: $(grep -m1 -o 'gate_strategy=[a-z]*' "$D/run.log" || echo '(not logged)')"
  echo "[smoke]   fires: $(grep -c 'fire=True' "$D/run.log" 2>/dev/null || echo 0) / ticks: $(grep -c 'ctrl.gate' "$D/run.log" 2>/dev/null || echo 0)"
  grep -m3 'Traceback\|Error\|WARNING' "$D/run.log" 2>/dev/null | head -3
done

echo
echo "[smoke] ===== VERDICT ====="
"$PY" - "$OUT" <<'PY'
import json, os, sys
root = sys.argv[1]
res = {}
for thr in ("0.05", "0.95"):
    fp = os.path.join(root, f"thr_{thr}", "online_pred.jsonl")
    if not os.path.exists(fp):
        print(f"  thr={thr}  NO PREDICTIONS -- smoke FAILED"); continue
    rows = [json.loads(l) for l in open(fp) if l.strip()]
    res[thr] = sum(len(r.get("predictions", [])) for r in rows)
    xrt = [r.get("realtime_factor") for r in rows if r.get("realtime_factor")]
    print(f"  thr={thr}  n_sample={len(rows)}  n_emit={res[thr]}  "
          f"xRT={sum(xrt)/len(xrt):.2f}" if xrt else f"  thr={thr} n_emit={res[thr]}")
if len(res) == 2:
    lo, hi = res["0.05"], res["0.95"]
    ok = lo > hi
    print(f"\n  emissions at 0.05 ({lo}) {'>' if ok else '<='} at 0.95 ({hi})")
    print("  THRESHOLD IS LIVE -- the sweep will measure something" if ok else
          "  *** THRESHOLD IS INERT -- do NOT launch the sweep ***")
PY
