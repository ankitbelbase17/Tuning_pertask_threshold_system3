#!/usr/bin/env bash
# preflight_stage3.sh -- sec.6's sanity gate. Run this before launching stage 3.
#
# Stage 3 is the headline number. Every failure mode it has is SILENT: a stale
# threshold export, a config that was never updated, or a second debug job that
# starves the chain all produce a run that completes and reports a plausible
# wrong answer. So the gate is an assertion, not a checklist.
set -uo pipefail
export THR_ROOT=${THR_ROOT:-/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit}
source "$THR_ROOT/env.sh"
fail=0
note () { printf '  %-6s %s\n' "$1" "$2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

echo "== stage 3 preflight =="

[ -f "$THR_ROOT/FINAL_THRESHOLDS.json" ] \
  && note OK "FINAL_THRESHOLDS.json present" \
  || note FAIL "FINAL_THRESHOLDS.json missing -- pass 2 has not been picked yet"

# 1. the config the RUN will import must equal the fit's output
if [ -f "$THR_ROOT/FINAL_THRESHOLDS.json" ]; then
  "$PY" - <<'PY'
import json, os, sys
sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "repo", "async_omni_v2"))
from config import AsyncOmniConfig as C
want = json.load(open(os.path.join(os.environ["THR_ROOT"], "FINAL_THRESHOLDS.json")))
have = C().task_hit_thresholds
bad = {t: (have.get(t), v) for t, v in want.items() if abs(have.get(t, -1) - v) > 1e-9}
print("  OK     config.task_hit_thresholds == FINAL_THRESHOLDS" if not bad
      else f"  FAIL   config != FINAL_THRESHOLDS: {bad}")
sys.exit(1 if bad else 0)
PY
  [ $? -eq 0 ] || fail=1
fi

# 2. no override may reach the workers. sec.6: "no OMNIPRO_HIT_THRESHOLD override"
[ -z "${OMNIPRO_HIT_THRESHOLD:-}" ] \
  && note OK "no OMNIPRO_HIT_THRESHOLD in the environment" \
  || note FAIL "OMNIPRO_HIT_THRESHOLD=$OMNIPRO_HIT_THRESHOLD is exported -- unset it"

# 3. the fitted gate must still be the gate that was fitted. OMNIPRO_GATE_FIT=0
#    would silently restore the shipped 0.5 hysteresis rail and make the fitted
#    thresholds inert again -- the exact defect this experiment corrects.
case "${OMNIPRO_GATE_FIT:-1}" in
  0|false|False|"") note FAIL "OMNIPRO_GATE_FIT=${OMNIPRO_GATE_FIT} -- fitted gate disabled" ;;
  *)               note OK   "fitted gate active (OMNIPRO_GATE_FIT=${OMNIPRO_GATE_FIT:-1})" ;;
esac

# 4. debug-qos allows ONE running job per user: a pass-1/2 chain still in flight
#    would trade windows with stage 3 and both would crawl.
RUN=$(squeue -u "$USER" -h -p debug -t RUNNING,PENDING -o "%i %j" 2>/dev/null | grep -c thrfit || true)
[ "${RUN:-0}" = "0" ] \
  && note OK "no thrfit chain occupying the debug partition" \
  || note WARN "$RUN thrfit job(s) still queued/running -- stage 3 will contend for MaxJobsPU=1"

# 5. state, not an assertion: which mode/refractory stage 3 will actually run.
#    These were NOT refitted (RUNBOOK sec.2) and belong in the paper's limitations.
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "repo", "async_omni_v2"))
from config import AsyncOmniConfig as C
c = C()
print("  ----   gate in force (mode/refractory inherited from the vision-only fit):")
for t in sorted(c.task_hit_thresholds):
    print(f"         {t:<30} thr={c.task_hit_thresholds[t]:<7} "
          f"mode={c.task_gate_modes.get(t,'edge'):<6} refr={c.task_refractory_s.get(t,0.0)}s")
PY

echo
[ "$fail" = "0" ] && echo "PREFLIGHT PASS -- safe to launch stage 3" \
                  || { echo "PREFLIGHT FAIL -- do not launch"; exit 1; }
