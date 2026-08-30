#!/usr/bin/env bash
# status.sh -- one screen: both fleets, the worklist, and the shape the tail wants.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
source "$THR_ROOT/env.sh"

echo "===== worklist ====="
timeout 300 "$PY" "$THR_ROOT/lib/chain_state.py" --nodes 4 2>/dev/null \
  | sed 's/^/  /'

echo
echo "===== debug fleet (SLURM) ====="
timeout 60 squeue -u "$USER" -o "%.9i %.9P %.9T %.6D %.11l %.11M %.20R" 2>/dev/null \
  | sed 's/^/  /'

echo
# The node is read from the newest chain log, not hardcoded: the chain is
# relaunched from whichever login node the shell happens to be on (it has run on
# ln002 and ln003), and a status header naming the wrong one sends you to look at
# an idle node's GPUs and conclude the fleet is dead.
LN=$(sed -n 's/.*chain start .* on \([a-z0-9-]*\),.*/\1/p' \
     "$(ls -t "$THR_ROOT"/logs/login_chain*.out 2>/dev/null | head -1)" 2>/dev/null | tail -1)
echo "===== login fleet (${LN:-unknown node}) ====="
if [ -f "$THR_ROOT/STOP_LOGIN" ]; then
  echo "  STOP_LOGIN present -- login chain is standing down"
fi
n=$(pgrep -u "$USER" -f 'bin/worker.sh 1[0-9][0-9]' 2>/dev/null | wc -l)
echo "  login worker lanes alive: $n"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  gpu /'

echo
echo "===== recent lane activity ====="
for f in "$THR_ROOT"/logs/lane_*.log; do
  [ -f "$f" ] || continue
  printf '  %-22s %s\n' "$(basename "$f")" "$(tail -1 "$f")"
done | tail -20
