#!/usr/bin/env python
"""Thin CLI over worklist.py so worker.sh can claim/release without inline python."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.environ["THR_ROOT"], "lib"))
import worklist as W                                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["next", "release", "beat", "status"])
ap.add_argument("--worklist",
                default=os.path.join(os.environ["THR_ROOT"], "worklist_p1.tsv"))
ap.add_argument("--lane", default="?")
ap.add_argument("--unit", default="")          # "pass<TAB>task<TAB>thr<TAB>shard"
# THE CURRENT GENERATION'S WINDOW. Without it `next` hands back cells whose only
# remaining samples are longer than this job shape can finish, and the worker
# re-claims the same barren unit forever -- see worklist.cell_actionable.
ap.add_argument("--max_dur", type=float, default=0.0)
# Units this lane already attempted this generation and banked nothing from,
# as "pass:task:thr:shard" separated by commas.
ap.add_argument("--skip", default="")
a = ap.parse_args()

if a.cmd == "next":
    skip = [tuple(x.split(":")) for x in a.skip.split(",") if x]
    u = W.next_unit(a.worklist, a.lane, max_dur=a.max_dur, skip=skip)
    if u:
        print(f"{u[0]}\t{u[1]}\t{u[2]:.4f}\t{u[3]}")
    sys.exit(0 if u else 3)                    # 3 = nothing left to claim

elif a.cmd in ("release", "beat"):
    p, t, thr, sh = a.unit.split("\t")
    (W.release if a.cmd == "release" else W.heartbeat)(p, t, float(thr), int(sh))

elif a.cmd == "status":
    units = W.read_worklist(a.worklist)
    cells = sorted({(p, t, thr) for p, t, thr, _ in units})
    done = sum(1 for c in cells if W.cell_complete(*c)[0])
    n_ids = sum(len(W.done_ids(W.cell_dir(*c))) for c in cells)
    print(f"CELLS_TOTAL={len(cells)}")
    print(f"CELLS_DONE={done}")
    print(f"SAMPLE_EVALS_DONE={n_ids}")
    print(f"SAMPLE_EVALS_TARGET={len(cells) * 15}")
    print(f"COMPLETE={1 if done == len(cells) else 0}")
