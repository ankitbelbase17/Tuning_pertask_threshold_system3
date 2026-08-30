#!/usr/bin/env python
"""Work-unit definition, generation and atomic claiming.

TWO FLEETS, ONE WORKLIST. A debug-partition chain (16 GPUs, 22-minute windows)
and a login-node chain (ln003, shared GPUs, killable at any moment) both pull
from this list. They must not duplicate work and neither may block the other, so:

  * COMPLETION is a property of the CELL, not of a lane: a cell is done when all
    of its frozen sample ids appear in ANY lane's online_pred.jsonl. This is the
    same `--done_glob` discipline evaluate.py already implements per sample, so a
    unit killed mid-window resumes in the next generation rather than restarting.
  * CLAIMS are advisory, not locks. `mkdir` is atomic, so a claim is a cheap way
    to keep two lanes off the same shard, but it carries a TTL: a lane SIGKILLed
    at the wall clock cannot release its claim, and a permanent lock would
    deadlock the run. Worst case after a TTL expiry is duplicated work, which
    per-sample resume then absorbs -- never corruption.
"""
from __future__ import annotations
import json, os, glob, time, itertools

TASKS = ["instant_event_alert", "semantic_condition_alert", "explicit_target_grounding",
         "snapshot_counting", "cumulative_counting", "dedup_counting",
         "realtime_state_monitor", "event_narration", "sequential_step_instruction"]

# sec.3 coarse grid, verbatim.
THR_P1 = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

# RAIL EXTENSION, and only where the evidence asks for it. The offline replay of
# the completed 2,700-sample phase-A run (resweep.py, 507,865 ticks) put the
# optimum for exactly these two tasks at 0.05 -- the bottom RAIL of resweep's own
# search range, which is also THR_P1's minimum. GATE_TUNING.md sec.3 rule 2: a
# winner sitting at the edge of the range means the grid is TRUNCATING the search,
# so widen it and re-fit. Adding the two points everywhere would cost 18 extra
# cells to answer a question only two tasks are asking, so it is targeted.
THR_P1_EXTRA = {
    "semantic_condition_alert":    [0.01, 0.02],
    "sequential_step_instruction": [0.01, 0.02],
}
NSHARD = 4

ROOT = os.environ["THR_ROOT"]
CLAIM_TTL = float(os.environ.get("THR_CLAIM_TTL", 1800))   # 30 min > any one window


def cell_dir(pass_, task, thr):
    return os.path.join(ROOT, "results", pass_, task, f"thr_{thr:.4f}")


def frozen_ids(task, off=0):
    with open(os.path.join(ROOT, "splits_thr_fit", f"{task}.off{off}.json")) as f:
        return set(json.load(f)["ids"])


def done_ids(cell):
    ids = set()
    for fp in glob.glob(os.path.join(cell, "lane*", "online_pred.jsonl")):
        try:
            with open(fp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ids.add(json.loads(line)["id"])
                    except (json.JSONDecodeError, KeyError):
                        pass          # a lane SIGKILLed mid-write leaves a torn line
        except OSError:
            pass                      # /iopsstor flaps; treat as "nothing known yet"
    return ids


def task_thresholds(task, base=None):
    """The grid for one task: the shared coarse grid plus any rail extension."""
    return sorted(set((base if base is not None else THR_P1)
                      + THR_P1_EXTRA.get(task, [])))


def gen_worklist(path, thresholds=None, tasks=TASKS, pass_="p1", nshard=NSHARD,
                 per_task=None):
    """per_task = {task: [thr, ...]} overrides the shared grid (used for pass 2)."""
    rows = []
    for t in tasks:
        grid = (per_task or {}).get(t) or task_thresholds(t, thresholds)
        for thr in grid:
            rows += [f"{pass_}\t{t}\t{thr:.4f}\t{s}" for s in range(nshard)]
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    return rows


def read_worklist(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p, t, thr, sh = line.rstrip("\n").split("\t")
                out.append((p, t, float(thr), int(sh)))
    return out


def cell_complete(pass_, task, thr):
    want = frozen_ids(task)
    return want.issubset(done_ids(cell_dir(pass_, task, thr))), want


def claim(pass_, task, thr, shard, lane):
    """Atomically claim one shard. -> True if this lane got it."""
    d = os.path.join(cell_dir(pass_, task, thr), ".claims")
    os.makedirs(d, exist_ok=True)
    c = os.path.join(d, f"shard{shard}")
    try:
        os.mkdir(c)                                   # atomic
    except FileExistsError:
        try:
            age = time.time() - os.path.getmtime(c)
        except OSError:
            return False
        if age < CLAIM_TTL:
            return False
        # STALE: the holder was killed at a wall clock without releasing. Take it.
        try:
            os.utime(c, None)
        except OSError:
            return False
    with open(os.path.join(c, "owner"), "w") as f:
        f.write(f"{lane} {os.uname().nodename} {int(time.time())}\n")
    return True


def heartbeat(pass_, task, thr, shard):
    c = os.path.join(cell_dir(pass_, task, thr), ".claims", f"shard{shard}")
    try:
        os.utime(c, None)
    except OSError:
        pass


def release(pass_, task, thr, shard):
    c = os.path.join(cell_dir(pass_, task, thr), ".claims", f"shard{shard}")
    for p in (os.path.join(c, "owner"), c):
        try:
            os.remove(p) if os.path.isfile(p) else os.rmdir(p)
        except OSError:
            pass


_DUR_CACHE = {}


def durations():
    """id -> duration, from the benchmark. Cached: next_unit is called per unit."""
    if not _DUR_CACHE:
        with open(os.environ["OMNIPRO_BENCHMARK_JSON"]) as f:
            for e in json.load(f):
                _DUR_CACHE[e["id"]] = float(e.get("duration", 0.0))
    return _DUR_CACHE


def cell_actionable(pass_, task, thr, max_dur):
    """Does this cell have work the CURRENT job shape can actually finish?

    THE BUG THIS EXISTS TO FIX (2026-08-28). `cell_complete` asks "are all 15
    frozen ids banked?", which is the right question for the pass but the WRONG
    one for a lane: a cell whose only remaining samples are longer than this
    generation's --max_dur is not complete, yet there is nothing a lane can do
    about it. worker.sh would claim such a unit, run evaluate.py (which filters
    every sample out and exits in ~12 s), release, and -- because next_unit
    returns the first incomplete claimable unit in worklist order -- IMMEDIATELY
    CLAIM THE SAME UNIT AGAIN. Measured: 26,277 claims in seven generations, all
    for instant_event_alert, while 84 of the 94 cells were never touched once.
    Generations 3-7 banked 3, 1, 1, 0, 0 samples and the dead-chain guard
    correctly stopped the run.

    max_dur == 0 means "no cap" (the 1-node shape), where every cell is
    actionable by definition.
    """
    if not max_dur or max_dur <= 0:
        return True
    dur = durations()
    remaining = frozen_ids(task) - done_ids(cell_dir(pass_, task, thr))
    return any(dur.get(i, 0.0) <= max_dur for i in remaining)


def next_unit(worklist_path, lane, max_dur=0.0, skip=()):
    """Claim and return the next ACTIONABLE unit, or None if there is none.

    Cells are consumed in worklist order so every threshold of a task advances
    together -- that way a run stopped early still yields a usable, if coarse,
    curve for every task instead of a complete curve for two tasks and nothing
    for the rest.

    Two filters keep a lane from spinning on work it cannot do:
      max_dur -- skip cells whose remaining samples all exceed this window
                 (see cell_actionable)
      skip    -- units this lane already attempted THIS GENERATION and banked
                 nothing from. The max_dur filter is per CELL, but a unit is a
                 SHARD of a cell: shard 0 can be empty while shards 1-3 are not,
                 which is a narrower version of the same infinite loop. The
                 caller passes what it has already found barren.
    """
    skip = set(skip)
    for (p, t, thr, sh) in read_worklist(worklist_path):
        if (p, t, f"{thr:.4f}", str(sh)) in skip:
            continue
        ok, _ = cell_complete(p, t, thr)
        if ok:
            continue
        if not cell_actionable(p, t, thr, max_dur):
            continue
        if claim(p, t, thr, sh, lane):
            return (p, t, thr, sh)
    return None
