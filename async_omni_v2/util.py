"""
util.py — tiny shared helpers: a wall-clock-relative logger + a profiler.

Log format:  [  wall | vid  ] who  message
  wall = seconds since process start, vid = video timestamp of the event.
Reading the gap between `wall` and `vid` is how you see the async decoupling
(e.g. the writer lagging behind the orchestrator).

Profiler: a thread-safe accumulator answering the operational questions —
  * which GPU op is heaviest (per-op compute time),
  * is anyone blocked/starved (per-op QUEUE WAIT: time a request sat in the
    manager queue before the single GPU worker picked it up),
  * how big is each frame (vision tokens), how heavy is vision-encode vs the
    LLM prefill, and how long the writer takes from trigger -> first token ->
    done. Print `prof.summary()` at shutdown.
"""
import threading
import time
from collections import defaultdict

_T0 = time.time()
_LOCK = threading.Lock()


def log(tag, vid_t, msg):
    with _LOCK:
        print(f"[{time.time()-_T0:6.1f}s | vid {vid_t:6.1f}s] {tag:<12} {msg}", flush=True)


class EncoderControl:
    """Orchestrator -> encoder back-channel: the proactive INPUT gate.

    The orchestrator is the brain; it tells the vision encoder *how densely to
    look right now*. When the action is important it raises the fps ("focus");
    when it's boring it lowers it. The encoder reads `get_fps()` every frame and
    paces itself in real (wall-clock) time. This is a lock-protected scalar, so
    neither side ever blocks the other for more than a pointer update.
    """

    def __init__(self, fps, min_fps, max_fps):
        self._lock = threading.Lock()
        self._fps = max(min_fps, min(max_fps, fps))
        self.min_fps, self.max_fps = min_fps, max_fps

    def set_fps(self, f):
        with self._lock:
            self._fps = max(self.min_fps, min(self.max_fps, f))

    def get_fps(self):
        with self._lock:
            return self._fps


class _Stat:
    __slots__ = ("n", "sum", "max", "wsum", "wmax")

    def __init__(self):
        self.n = self.sum = self.max = self.wsum = self.wmax = 0.0

    def add(self, dur, wait=0.0):
        self.n += 1
        self.sum += dur
        self.wsum += wait
        if dur > self.max:
            self.max = dur
        if wait > self.wmax:
            self.wmax = wait


class Profiler:
    """Thread-safe. `record_op` is for GPU ops (dur + queue wait); `observe`
    is for scalar samples (token counts, latencies); `incr` for counters."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._ops = defaultdict(_Stat)        # GPU ops: dur + queue wait
        self._obs = defaultdict(_Stat)        # scalar samples: dur only
        self._ctr = defaultdict(float)

    def record_op(self, label, dur, wait):
        if not self.enabled:
            return
        with self._lock:
            self._ops[label].add(dur, wait)

    def observe(self, label, value):
        if not self.enabled:
            return
        with self._lock:
            self._obs[label].add(value)

    def incr(self, key, val=1.0):
        if not self.enabled:
            return
        with self._lock:
            self._ctr[key] += val

    def summary(self):
        if not self.enabled:
            return "[profiler disabled]"
        with self._lock:
            ops = dict(self._ops)
            obs = dict(self._obs)
            ctr = dict(self._ctr)
        lines = ["", "=" * 78, "PROFILE SUMMARY", "=" * 78]

        # ---- GPU ops: where time goes + who waits on the cache lock ----
        lines.append("GPU ops (per-thread; wait = blocked on shared-cache lock):")
        lines.append(f"  {'op':<16}{'n':>6}{'tot_s':>9}{'mean_ms':>10}"
                     f"{'max_ms':>9}{'wait_tot':>10}{'wait_mean':>11}")
        total = sum(s.sum for s in ops.values())
        for label, s in sorted(ops.items(), key=lambda kv: -kv[1].sum):
            n = int(s.n)
            lines.append(
                f"  {label:<16}{n:>6}{s.sum:>9.2f}{1000*s.sum/max(n,1):>10.1f}"
                f"{1000*s.max:>9.1f}{s.wsum:>10.2f}{1000*s.wsum/max(n,1):>10.1f}")
        lines.append(f"  {'TOTAL GPU':<16}{'':>6}{total:>9.2f}")

        # ---- scalar samples (tokens, latencies) ----
        if obs:
            lines.append("")
            lines.append("Samples (mean / max):")
            for label, s in sorted(obs.items()):
                n = int(s.n)
                lines.append(f"  {label:<22} n={n:<5} mean={s.sum/max(n,1):>9.2f}"
                             f"  max={s.max:>9.2f}")

        # ---- counters ----
        if ctr:
            lines.append("")
            lines.append("Counters: " + "  ".join(f"{k}={int(v)}" for k, v in sorted(ctr.items())))
        lines.append("=" * 78)
        return "\n".join(lines)
