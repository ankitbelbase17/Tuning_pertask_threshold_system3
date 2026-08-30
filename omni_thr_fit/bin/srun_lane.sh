#!/usr/bin/env bash
# srun_lane.sh -- the per-task entry point of the debug fleet.
#
# One SLURM task = one GPU = one lane. Exists only to turn SLURM_PROCID into a
# lane id and hand off to worker.sh, which is shared verbatim with the ln003
# login fleet.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
DEADLINE=${1:?deadline}
MAX_DUR=${2:-0}
exec bash "$THR_ROOT/bin/worker.sh" "${SLURM_PROCID:-0}" "$DEADLINE" "$MAX_DUR"
