#!/usr/bin/env bash
# srun_stage3_lane.sh -- per-task entry point of the stage-3 fleet.
# One SLURM task = one GPU = one shard of --nshards. Mirrors srun_lane.sh.
set -uo pipefail
export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
NSHARDS=${1:?nshards}
DEADLINE=${2:?deadline}
MAX_DUR=${3:-0}
exec bash "$THR_ROOT/bin/stage3_worker.sh" "${SLURM_PROCID:-0}" "$NSHARDS" "$DEADLINE" "$MAX_DUR"
