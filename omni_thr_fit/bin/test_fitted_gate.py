#!/usr/bin/env python
"""Unit test for the fitted gate (edge|level + refractory). No GPU, no model.

Replays the exact expression from controller.py's `fitted` branch over synthetic
p_hit sequences. Cheap guard against the three ways this can be wrong:
  - refractory suppressing the FIRST fire (last_fire_vt initialised to 0, not -inf)
  - "edge" firing on every tick of a sustained crossing
  - "level" NOT firing on a sustained crossing
  - the FIRST tick counting as a rising edge (it has no predecessor, so it is not
    one) -- see bin/audit_first_tick.py for what that cost on the long-refractory
    tasks before it was fixed
"""
import os, sys, dataclasses
sys.path.insert(0, os.path.join(os.environ["REPO"], "async_omni_v2"))
from config import AsyncOmniConfig                                  # noqa: E402

def replay(p_hits, thr, mode, refrac, answers=None):
    """Mirror of controller.py's fitted branch. Returns the fire times."""
    answers = answers if answers is not None else [True] * len(p_hits)
    prev_above, last_fire_vt, fires = None, -1e9, []
    for vt, (p, ans) in enumerate(zip(p_hits, answers)):
        above = p >= thr
        # the answer decode is gated by the SAME threshold upstream, so an answer
        # can only exist on a tick that is above it
        ans = ans and above
        crossed = above if mode == "level" else (above and prev_above is False)
        if ans and crossed and (vt - last_fire_vt) >= refrac:
            fires.append(vt); last_fire_vt = vt
        prev_above = above
    return fires

fail = 0
def check(name, got, want):
    global fail
    ok = got == want
    fail += (not ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got} want {want}")

print("== config carries the new fields ==")
c = AsyncOmniConfig()
check("task_gate_mode default", c.task_gate_mode, "edge")
check("refractory_s default", c.refractory_s, 0.0)
c2 = dataclasses.replace(c, gate_strategy="fitted", task_gate_mode="level",
                         refractory_s=7.0, hit_threshold=0.33)
check("replace works", (c2.gate_strategy, c2.task_gate_mode, c2.refractory_s,
                        c2.hit_threshold), ("fitted", "level", 7.0, 0.33))

print("\n== edge vs level on a sustained crossing ==")
p = [0.1, 0.9, 0.9, 0.9, 0.1, 0.9]
check("edge fires on rises only", replay(p, 0.5, "edge", 0), [1, 5])
check("level fires every tick above", replay(p, 0.5, "level", 0), [1, 2, 3, 5])

print("\n== refractory ==")
# the re-rise is at t=5, the fire at t=1: a gap of 4 clears a refractory of 3
check("edge + refrac 3 lets the re-rise through", replay(p, 0.5, "edge", 3), [1, 5])
check("edge + refrac 5 suppresses the re-rise", replay(p, 0.5, "edge", 5), [1])
check("level + refrac 2", replay(p, 0.5, "level", 2), [1, 3, 5])
check("first fire is NOT suppressed by a 600s refractory",
      replay([0.1, 0.9, 0.9], 0.5, "edge", 600.0), [1])

print("\n== the first tick is not a rising edge ==")
# THE ARTIFACT. p_hit is high on tick 0 because the model answers from the prompt
# before it has seen any video. Under `edge` that must not fire: with a 600 s
# refractory the spurious fire consumes the video's only emission and the real
# event at t=2 is locked out.
check("edge does NOT fire on a high tick 0",
      replay([0.9, 0.1, 0.9], 0.5, "edge", 600.0), [2])
check("edge with no predecessor above stays silent",
      replay([0.9, 0.9, 0.9], 0.5, "edge", 0.0), [])
check("level DOES still fire on tick 0 (every tick above is a level hit)",
      replay([0.9, 0.1, 0.9], 0.5, "level", 0.0), [0, 2])
check("a genuine rise at tick 1 still fires",
      replay([0.1, 0.9], 0.5, "edge", 0.0), [1])

print("\n== the threshold is now the rail across the WHOLE grid ==")
# the defect this gate exists to fix: under hysteresis every thr <= 0.5 collapsed
# onto 0.5. Under `fitted` each grid point must give a distinct emission count.
ramp = [i / 100 for i in range(101)] * 3
counts = {thr: len(replay(ramp, thr, "level", 0))
          for thr in (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)}
print("  thr -> n_emit:", counts)
check("all 10 grid points distinct", len(set(counts.values())), 10)
check("monotone decreasing in threshold",
      list(counts.values()) == sorted(counts.values(), reverse=True), True)

print(f"\n{'ALL PASS' if not fail else str(fail) + ' FAILURES'}")
sys.exit(1 if fail else 0)
