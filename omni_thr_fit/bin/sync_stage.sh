#!/bin/bash
# sync_stage.sh -- mirror the two live trees into the push staging repo.
#
# Staging is a COPY, not a symlink, so a commit taken from it is a frozen
# snapshot -- that is the point, but it also means staging goes stale silently:
# a pass appends samples to EXISTING lane files, so the file COUNT stays put
# while the contents move on. Never judge staging freshness by counting files.
#
# --delete mirrors removals too. The repo's own top-level files (.gitignore,
# README.md) and .git live ABOVE these two directories and are untouched.
#
# Lived in a session scratchpad until 2026-08-31, which was the wrong place for
# it: the exclude list below is the only thing standing between a 2.4 GB tree
# copy and the push, and an unversioned copy of that list is one lost scratchpad
# away from being reconstructed from memory.
#
#   PUSH_STAGE=/path/to/push_stage bash bin/sync_stage.sh
set -euo pipefail

BASE=${SYNC_BASE:-/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28}
S=${PUSH_STAGE:?PUSH_STAGE must name the staging repo}

# Excludes mirror .gitignore's intent. They are applied HERE as well as there so
# the 2.4 GB repo/ copy and the 444 MB of run.log never even get written into
# the scratchpad -- gitignore would keep them out of the commit but not off disk.
COMMON=(
  --exclude '.git/'
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '*.pyo'
  --exclude '.env'
  --exclude '*.bak_*'
  --exclude 'core'
  --exclude 'core.*'
  --exclude 'core_*'
  --exclude '*.pt'
  --exclude '*.safetensors'
  --exclude '*.webm'
  --exclude '*.mp4'
  --exclude '*.mkv'
)

echo "== omni_thr_fit =="
rsync -a --delete "${COMMON[@]}" \
  --exclude 'repo/' \
  --exclude 'run.log' \
  "$BASE/omni_thr_fit/" "$S/omni_thr_fit/"

echo "== system3_qwem_omni =="
rsync -a --delete "${COMMON[@]}" \
  --exclude 'omniprofast/output*/' \
  --exclude '.git.corrupt.bak/' \
  --exclude '.claude/' \
  "$BASE/system3_qwem_omni/" "$S/system3_qwem_omni/"

echo
echo "== staging size =="
du -sh "$S/omni_thr_fit" "$S/system3_qwem_omni"
echo
echo "== safety re-check: nothing that must never ship =="
for pat in '.env' 'run.log'; do
  n=$(find "$S" -name "$pat" -not -path '*/.git/*' | wc -l)
  echo "  $pat on disk in staging: $n"
done
echo -n "  repo/ present: "; [ -d "$S/omni_thr_fit/repo" ] && echo YES-PROBLEM || echo no
