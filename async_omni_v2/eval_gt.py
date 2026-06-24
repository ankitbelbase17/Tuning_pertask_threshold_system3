"""
eval_gt.py — OPTIONAL, video-specific accuracy check (NOT general).

This only activates for the "France 3-1 Senegal, FIFA World Cup 2026" highlight
(matched by its YouTube id in the video path). For any other video the
evaluator stays None and nothing is scored.

It answers: how close did the orchestrator's goal TRIGGERS — and the WRITER's
spoken lines — land relative to the real goal moments? It prints, per goal, the
signed video-time error of the trigger, the writer's wall-clock latency + line,
plus detection rate, mean |error|, and any false triggers.
"""
import threading

# Real goal-like moments (video seconds), from the ground truth provided.
# The 55s event was a France goal DISALLOWED for offside (score stayed 1-0) —
# the model still SEES a net/celebration there, so firing is reasonable; we tag
# it so the report can call it out rather than penalise it as a false positive.
WORLD_CUP_GT = [
    (46.0, "GOAL France (Mbappe)  -> 1-0", False),
    (55.0, "France goal DISALLOWED (offside, stays 1-0)", True),
    (68.0, "GOAL France (Barcola) -> 2-0", False),
    (83.0, "GOAL Senegal          -> 2-1", False),
    (91.0, "GOAL France (Mbappe)  -> 3-1", False),
]
_VIDEO_KEY = "n3JDGlOwMJ4"          # YouTube id in the highlight's filename


def make_evaluator(video_path, window=10.0, enabled=True):
    path = video_path or ""
    if not enabled:
        return None
    if _VIDEO_KEY in path or ("France" in path and "Senegal" in path):
        return GroundTruthEvaluator(WORLD_CUP_GT, window)
    return None


class GroundTruthEvaluator:
    def __init__(self, gt, window):
        self.gt = sorted(gt)
        self.window = window
        self._lock = threading.Lock()
        self.triggers = []          # (vt, yes_share)
        self.writes = []            # (vt, text, wall_latency_s)

    # ---- recording (any thread) ----
    def record_trigger(self, vt, share):
        with self._lock:
            self.triggers.append((float(vt), float(share)))

    def record_write(self, vt, text, wall_latency):
        with self._lock:
            self.writes.append((float(vt), text, float(wall_latency)))

    # ---- scoring ----
    def _match(self, triggers):
        """Greedy nearest-distance assignment: each goal <-> at most one trigger
        within the window, picking globally smallest |error| first."""
        pairs = []
        for ei, (t, _, _) in enumerate(self.gt):
            for ti, (vt, _) in enumerate(triggers):
                if abs(vt - t) <= self.window:
                    pairs.append((abs(vt - t), ei, ti))
        pairs.sort()
        ev_match, used = {}, set()
        for _, ei, ti in pairs:
            if ei in ev_match or ti in used:
                continue
            ev_match[ei] = ti
            used.add(ti)
        return ev_match, used

    def _nearest_write(self, vt):
        best = None
        for w in self.writes:
            if best is None or abs(w[0] - vt) < abs(best[0] - vt):
                best = w
        return best if (best and abs(best[0] - vt) <= 0.5) else None

    def report(self):
        with self._lock:
            triggers = sorted(self.triggers)
            n_writes = len(self.writes)
        ev_match, used = self._match(triggers)

        out = ["", "=" * 78,
               "GROUND-TRUTH EVAL — France 3-1 Senegal (World Cup 2026)",
               "=" * 78,
               f"match window = +/-{self.window:.0f}s video time | "
               f"{len(triggers)} triggers, {n_writes} writer lines", ""]
        deltas = []
        for ei, (t, label, disallowed) in enumerate(self.gt):
            if ei in ev_match:
                vt, share = triggers[ev_match[ei]]
                d = vt - t
                deltas.append(abs(d))
                w = self._nearest_write(vt)
                if w is not None:
                    wtxt = f'  writer +{w[2]:4.1f}s: "{w[1][:46]}"'
                else:
                    wtxt = "  writer: (none)"
                out.append(f"  {t:5.1f}s {label:44s} fired {vt:5.1f}s "
                           f"(d={d:+5.1f}s, share={share:.2f}){wtxt}")
            else:
                tag = "(disallowed; ok to skip)" if disallowed else "*** MISSED ***"
                out.append(f"  {t:5.1f}s {label:44s} {tag}")

        real = [g for g in self.gt if not g[2]]
        det_real = sum(1 for ei, g in enumerate(self.gt) if not g[2] and ei in ev_match)
        fp = [triggers[ti] for ti in range(len(triggers)) if ti not in used]
        out.append("")
        if deltas:
            out.append(f"  real goals detected: {det_real}/{len(real)}   "
                       f"(all events matched: {len(ev_match)}/{len(self.gt)})   "
                       f"mean |error| = {sum(deltas)/len(deltas):.1f}s")
        else:
            out.append(f"  real goals detected: 0/{len(real)}")
        if fp:
            out.append("  false triggers: " + ", ".join(
                f"{vt:.0f}s(share={s:.2f})" for vt, s in fp))
        out.append("=" * 78)
        return "\n".join(out)
