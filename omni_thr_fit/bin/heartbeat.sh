#!/usr/bin/env bash
# heartbeat.sh -- refresh one work unit's claim mtime every 5 minutes.
#
# Split out of worker.sh so the background subshell is a real process worker.sh
# can kill by pid without also killing the evaluate.py it is protecting.
set -uo pipefail
# Source the environment rather than relying on inheritance: `set -u` here would
# abort on an unset $PY, and an aborted heartbeat is invisible -- the claim simply
# goes stale 30 minutes later and another lane redoes the work.
source "${THR_ROOT:-/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit}/env.sh"
UNIT=${1:?unit}
while :; do
  sleep 300
  timeout 60 "$PY" "$THR_ROOT/lib/claim_cli.py" beat --unit "$UNIT" 2>/dev/null || exit 0
done
