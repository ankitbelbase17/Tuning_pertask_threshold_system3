# Invalidated by the edge-needs-a-predecessor fix (2026-08-28)

These cells were produced by the gate that seeded `prev_above = False`, so the
FIRST tick of every video counted as a rising edge. `instant_event_alert` is an
`edge`-mode task with a 600 s refractory, so that spurious tick-1 fire consumed
the only emission each video was allowed and the real event was locked out:

    vid  1.0s  p_hit=0.792  rise=True  fire=True    <- no context yet
    vid 24.0s  p_hit=0.958  rise=True  fire=False   <- the real event, suppressed

Every one of the 37 banked samples emitted exactly once, at vid 1.0 s, against
ground truth at 26-98 s -- so time-F1 was structurally 0 at every threshold and
the grid could not have chosen between them.

Kept, not deleted: they are the evidence for the fix, and THRESHOLD_FIT_RUNBOOK.md
cites them. Superseded by the cells under `../../results/p1/instant_event_alert`.

See `bin/audit_first_tick.py` and `FIRST_TICK_AUDIT.json` for the offline
measurement over the full 2,700-sample phase-A run that motivated the change,
and `repo/async_omni_v2/controller.py.bak_before_edge1` for the previous gate.

Only `edge`-mode tasks are affected; the four `level`-mode tasks are
bit-identical under the change and nothing of theirs was discarded (nothing of
theirs had banked yet either).
