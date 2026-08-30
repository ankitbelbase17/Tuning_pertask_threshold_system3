#!/usr/bin/env bash
# submit_chain.sh — run the FULL 932-sample eval on the debug partition by
# CHAINING jobs, because debug caps every job at 1:30.
#
# Each generation is an --array=0-3 job (4 nodes x 4 GPUs = 16 workers, one shard
# each) and depends on the previous with afterany, so the chain survives a
# generation that times out — which is exactly what we expect it to do.
#
# Progress carries across generations because evaluate.py resumes by default:
# samples already scored are skipped, so generation N+1 picks up where N stopped.
# Extra generations are therefore harmless no-ops once everything is done.
#
# Sizing: 36.5 video-hours / 16 workers = 2.28 video-hours each, ~3h of wall at
# the measured 1.3 s/tick -> ~2-3 generations. We submit more for slack.
#
# Usage:  bash submit_chain.sh [n_generations]     (default 5)
set -euo pipefail

REPO=/iopsstor/scratch/cscs/dbartaula/system_3
N=${1:-5}
cd "$REPO"

mkdir -p omniprofast/output_all
prev=""
echo "submitting $N chained generations of run_all_audio_ok.sbatch"
for i in $(seq 1 "$N"); do
  if [ -z "$prev" ]; then
    id=$(sbatch --parsable run_all_audio_ok.sbatch)
  else
    # afterany, not afterok: a generation that hits the 1:30 wall exits non-zero,
    # and we still want the next one to continue from the resume point.
    id=$(sbatch --parsable --dependency=afterany:"$prev" run_all_audio_ok.sbatch)
  fi
  echo "  generation $i: job $id${prev:+  (after $prev)}"
  prev=$id
done
echo
echo "watch:   squeue -u $USER"
echo "results: $REPO/omniprofast/output_all/*/online_metrics.json"
echo "scores:  python omniprofast/dump_scores.py output_all   # re-threshold offline"
