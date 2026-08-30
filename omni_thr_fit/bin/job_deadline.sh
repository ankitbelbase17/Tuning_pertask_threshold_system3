#!/usr/bin/env bash
# job_deadline.sh -- echo the epoch at which this generation's lanes must stop.
#
# WHY THIS IS NOT `now + 22*60`. The chain reshapes 4 -> 2 -> 1 nodes to give the
# long tail a longer window (88/n minutes), but a hardcoded 22-minute deadline
# would stop the lanes 22 minutes into a 44- or 88-minute wall and throw away
# half to three quarters of the very window the reshape was performed to obtain.
#
# Ask SLURM for the real EndTime; the arithmetic fallback (88/n, matching what
# queue_next requested) only matters if scontrol is unreachable.
#
# Usage: DEADLINE=$(bash bin/job_deadline.sh <nodes> [slack_s])
set -uo pipefail
NODES=${1:-4}
SLACK=${2:-120}          # leave room to flush logs before the SIGKILL at the wall

END=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
  ET=$(timeout 30 scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
         | tr ' ' '\n' | sed -n 's/^EndTime=//p' | head -1)
  [ -n "$ET" ] && END=$(date -d "$ET" +%s 2>/dev/null)
fi
if [ -z "$END" ]; then
  END=$(( $(date +%s) + (88 / NODES) * 60 ))
fi
echo $(( END - SLACK ))
