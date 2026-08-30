#!/usr/bin/env python
"""apply_thresholds.py -- write FINAL_THRESHOLDS.json into the worktree config.py.

sec.6 requires stage 3 to run with NO OMNIPRO_HIT_THRESHOLD override, so the
fitted values must live in `task_hit_thresholds` and be reached through the
adapter's normal per-task lookup -- the same path a deployed system takes.

REWRITES ONLY THE NINE VALUES, IN PLACE. The dict is located by its literal
`task_hit_thresholds: dict = field(...)` line and only the `"task": value,` lines
inside its braces are touched, so the surrounding comments -- which carry the
"all three knobs are ONE fitted config" warning -- survive verbatim. A regenerated
block would drop them, and that warning is the reason this experiment exists.

  python bin/apply_thresholds.py            # apply, keeping a timestamped backup
  python bin/apply_thresholds.py --check    # verify only, change nothing
"""
from __future__ import annotations
import argparse, json, os, re, shutil, time

ROOT = os.environ["THR_ROOT"]
CFG = os.path.join(ROOT, "repo", "async_omni_v2", "config.py")
FINAL = os.path.join(ROOT, "FINAL_THRESHOLDS.json")
START = "task_hit_thresholds: dict = field("


def rewrite(text, want):
    lines = text.splitlines(keepends=True)
    try:
        i = next(k for k, l in enumerate(lines) if START in l)
    except StopIteration:
        raise SystemExit(f"ABORT: {START!r} not found in {CFG}")
    j = next(k for k in range(i, len(lines)) if lines[k].strip().startswith("})"))
    seen, out = set(), []
    for l in lines[i + 1:j]:
        m = re.match(r'(\s*)"([a-z_]+)":\s*([0-9.]+),\s*$', l)
        if m and m.group(2) in want:
            indent, task = m.group(1), m.group(2)
            seen.add(task)
            out.append(f'{indent}"{task}": {want[task]},\n')
        else:
            out.append(l)              # a comment inside the dict stays put
    missing = set(want) - seen
    if missing:
        raise SystemExit(f"ABORT: tasks not present in the dict: {sorted(missing)}")
    return "".join(lines[:i + 1] + out + lines[j:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    with open(FINAL) as f:
        want = json.load(f)
    with open(CFG) as f:
        text = f.read()
    new = rewrite(text, want)

    if a.check:
        print("MATCH" if new == text else "DIFFERS")
        raise SystemExit(0 if new == text else 1)
    if new == text:
        print("config.py already holds FINAL_THRESHOLDS -- nothing to do")
        return
    bak = f"{CFG}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(CFG, bak)             # never overwrite without a copy on disk
    with open(CFG, "w") as f:
        f.write(new)
    print(f"backup: {bak}")
    for t, v in sorted(want.items()):
        print(f"  {t:<30}{v}")


if __name__ == "__main__":
    main()
