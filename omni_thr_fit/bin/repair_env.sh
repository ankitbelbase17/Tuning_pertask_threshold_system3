#!/usr/bin/env bash
# repair_env.sh -- give this experiment its OWN python environment.
#
# WHY THIS EXISTS. The run was launched against
#   /iopsstor/scratch/cscs/dbartaula/miniforge3/envs/prosync_env
# which belongs to ANOTHER ACCOUNT and is read-only to us. On 2026-08-28 at 19:34,
# mid-sweep, that account started a `pip` reinstall of torch: 594 files changed,
# pip itself was upgraded to 26.2, and `torch/bin/` plus `torch/lib/libshm*` were
# emptied. `torch/__init__.py` calls
#     _C._initExtension(_manager_path())
# at IMPORT time, and _manager_path() raises if torch/bin/torch_shm_manager is
# missing -- so `import torch` failed outright and every lane died.
#
# A shared, mutating dependency owned by someone else is not a base to run a
# multi-day experiment on. This script makes a private copy and repairs it.
#
# THE TORCH PACKAGE IS REPLACED WHOLESALE FROM THE OFFICIAL WHEEL, not patched.
# The source tree was being rewritten WHILE we copied it, so the copy could be
# torn in ways no file-by-file check would notice. Unpacking the exact build the
# env reports (2.12.1+cu130, cp312, manylinux_2_28_aarch64) makes the one package
# that was in flux verifiably consistent.
#
#   bash bin/repair_env.sh <path/to/torch-...whl> [dest_env]
set -euo pipefail
WHEEL=${1:?path to the torch wheel}
DEST=${2:-/iopsstor/scratch/cscs/dthapa/envs/prosync_env}
SP=$DEST/lib/python3.12/site-packages

[ -d "$SP" ] || { echo "no site-packages at $SP"; exit 2; }
echo "== repairing $DEST =="

# Never delete: move the possibly-torn copy aside, keep it for comparison.
STAMP=$(date +%Y%m%d_%H%M%S)
if [ -d "$SP/torch" ]; then
  mv "$SP/torch" "$SP/.torch.torn_$STAMP"
  echo "  set aside the copied torch -> .torch.torn_$STAMP"
fi

TMP=$(mktemp -d "${TMPDIR:-/iopsstor/scratch/cscs/dthapa/tmp}/torchwhl.XXXXXX")
echo "  unpacking $(basename "$WHEEL") ..."
"${PY_BOOT:-python3}" -c "
import zipfile, sys
zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])
" "$WHEEL" "$TMP"
mv "$TMP/torch" "$SP/torch"

# The wheel stores no permission bits for scripts on some builds; the shm manager
# must be executable or torch raises the same way it did with the file missing.
chmod +x "$SP/torch/bin/torch_shm_manager" 2>/dev/null || true
find "$SP/torch/lib" -name "*.so*" -exec chmod +x {} + 2>/dev/null || true

echo "  torch/bin: $(ls "$SP/torch/bin" 2>/dev/null | tr '\n' ' ')"
echo "== verifying =="
"$DEST/bin/python" -c "
import torch, sys
print('  python  ', sys.version.split()[0])
print('  torch   ', torch.__version__)
print('  cuda    ', torch.cuda.is_available(), torch.cuda.device_count(), 'device(s)')
" || { echo "VERIFY FAILED"; exit 1; }
echo "repair OK"
