#!/usr/bin/env bash
# Shared environment for the omni threshold-fit experiment (THRESHOLD_FIT_v2.md).
#
# ACCOUNT NOTE. This runs as `dthapa`, not `dbartaula` who authored the spec.
# Everything in dbartaula's scratch is group-readable EXCEPT hf_cache (mode 700),
# so the model weights are fetched into this account's own HF cache instead
# (bin/fetch_weights.py, login node only -- compute nodes have no egress).

export THR_ROOT=/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit
export REPO=$THR_ROOT/repo                 # isolated copy of system3_qwem_omni
# OUR OWN INTERPRETER, not the other account's.
# This experiment originally ran against dbartaula's prosync_env. On 2026-08-28
# at 19:34, mid-sweep, that account began a pip reinstall of torch: torch/bin/
# and torch/lib/libshm* were emptied, `_C._initExtension(_manager_path())` raised
# at import, and every lane died. A read-only environment owned by someone else
# who is actively changing it cannot underpin a multi-day run.
# bin/repair_env.sh made this private copy and rebuilt torch from the official
# 2.12.1+cu130 cp312 aarch64 wheel. Package set verified identical to the
# original (which also lacked torchaudio/decord/librosa/soundfile/scipy --
# the pipeline never used them); matplotlib added here for the figures.
export PY=/iopsstor/scratch/cscs/dthapa/envs/prosync_env/bin/python

# ---- dataset: OUR OWN COPY ----
# It was read straight out of dbartaula's scratch until 2026-08-28. After that
# account mutated its conda env mid-sweep (see PY above), every cross-account
# runtime dependency became a liability -- and this one is worse than the env
# was: a changed or removed video would not crash, it would silently produce
# different predictions. 31 GB, 1849 files, verified byte-identical per file.
export DD=/iopsstor/scratch/cscs/dthapa/omnipro_data
export OMNIPRO_DATASET_DIR=$DD
export OMNIPRO_BENCHMARK_JSON=$DD/benchmark.json

# ---- model weights: THIS account's cache ----
export HF_HOME=/iopsstor/scratch/cscs/dthapa/hf_home
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# PyAV links libXau via pillow's bundled libs; without this `import av` dies with
# "libXau-...so.6: cannot open shared object file" and every lane fails at startup.
export LD_LIBRARY_PATH="/iopsstor/scratch/cscs/dthapa/envs/prosync_env/lib/python3.12/site-packages/pillow.libs:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
# deterministic cuBLAS must be set before the first cuBLAS handle
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# THREAD CAP -- MEASURED, NOT PRECAUTIONARY. The first smoke run died with
#   libgomp: Thread creation failed: Resource temporarily unavailable
# AFTER the backend had loaded cleanly and the three pipeline threads had started.
# Cause: this login node's cgroup caps the user at 1000 pids
# (/sys/fs/cgroup/user.slice/user-1371.slice/pids.max) and ~440 were already in
# use, while torch sizes its OpenMP pool from the VISIBLE CORE COUNT -- 48 on a
# login node, 288 on a compute node. Four lanes at the default would ask for 192
# (login) or 1152 (compute) OMP threads on top of the pipeline threads and PyAV's
# decoders. The eval is GPU-bound and runs one sample at a time, so a small pool
# costs nothing measurable and this applies to BOTH fleets.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS

# ---- the pipeline under test ----
export PROSYNC_DIR=$REPO/async_omni_v2
export OMNIPRO_BACKEND=qwen2_5_omni
export OMNIPRO_GATE_MODE=controller
export OMNIPRO_DECODE_MODE=schema

# ---- judge: ONE model across every cell, or joint_f1 is not comparable ----
# metrics.py:353-360 shares a cache namespace between the Gemini and REST judges,
# so backend=auto would silently serve one model's verdicts as another's.
export OMNIPRO_JUDGE_BACKEND=${OMNIPRO_JUDGE_BACKEND:-gemini}
export OMNIPRO_JUDGE_CACHE=$THR_ROOT/judge_cache.json
export OMNIPRO_JUDGE_TRACE_PATH=$THR_ROOT/judge_trace.jsonl
[ -f "$THR_ROOT/PINNED_JUDGE" ] && export GEMINI_MODEL=$(cat "$THR_ROOT/PINNED_JUDGE")

# API keys
[ -f $REPO/.env ] && { set -a; . $REPO/.env; set +a; }

# ---- SLURM ----
# a168's association carries QoS `stop` (MaxTRESPU=0) so its jobs never start;
# a0264 carries `normal`. Verified 2026-08-28: a 4-node debug job under a0264
# schedules immediately, while a a168 job from 9 Jun is still pending.
export SLURM_ACCOUNT_USE=${SLURM_ACCOUNT_USE:-a0264}
