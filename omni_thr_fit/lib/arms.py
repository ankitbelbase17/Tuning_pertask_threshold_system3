#!/usr/bin/env python
"""arms.py -- resolve one stage-3 arm from STAGE3_ARMS.json.

  python lib/arms.py            # shell-eval'able: S3_DIR=... S3_OVERRIDE=...
  python lib/arms.py --field dir

The arm is read from $S3_ARM, defaulting to "fitted" -- so every existing caller
that knows nothing about arms keeps the exact behaviour it had before arms
existed: results/full2700 with no threshold override.

WHY A RESOLVER AND NOT TWO CONSTANTS IN EACH SCRIPT. The arms differ only in a
directory and an env var. Four scripts need both halves, and if any one of them
pairs the right directory with the wrong override the run still completes -- it
just banks the other arm's predictions under this arm's name. Nothing downstream
can detect that, because the predictions are individually well-formed. One
resolver means the pairing cannot be got wrong in only one place.
"""
from __future__ import annotations
import argparse, json, os, sys

ROOT = os.environ.get(
    "THR_ROOT", "/iopsstor/scratch/cscs/dthapa/system3_qwem_omni_8_28/omni_thr_fit")
ARMS = os.path.join(ROOT, "STAGE3_ARMS.json")


def load():
    with open(ARMS) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def resolve(name=None):
    name = name or os.environ.get("S3_ARM") or "fitted"
    arms = load()
    if name not in arms:
        sys.exit(f"ABORT: unknown S3_ARM={name!r}; known: {sorted(arms)}")
    a = dict(arms[name])
    a["name"] = name
    a["results"] = os.path.join(ROOT, "results", a["dir"])
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--field", default=None)
    a = ap.parse_args()
    arm = resolve(a.arm)
    if a.field:
        v = arm.get(a.field)
        print("" if v is None else v)
        return
    # Shell-eval'able, and EXPORTED. `eval "$(arms.py)"` on bare assignments
    # creates shell variables that child processes cannot see, so a python block
    # further down the caller reads them as empty and silently takes the wrong
    # branch -- which is how the control arm's preflight first printed the fitted
    # arm's thresholds as "in force". Exporting is what makes the resolver's
    # answer reach every child, which is the whole point of having one.
    #
    # An override of null prints EMPTY, not the string "None":
    # `OMNIPRO_HIT_THRESHOLD=None` would be read by the adapter as a value, and is
    # exactly the silent-wrong-answer path this file exists to prevent.
    ov = arm.get("override")
    print(f"export S3_ARM={arm['name']}")
    print(f"export S3_DIR={arm['dir']}")
    print(f"export S3_RESULTS={arm['results']}")
    print(f"export S3_OVERRIDE={'' if ov is None else ov}")
    print(f"export S3_LABEL={arm['label']!r}")


if __name__ == "__main__":
    main()
