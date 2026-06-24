"""
util.py — tiny shared helpers: a wall-clock-relative timestamped logger.

Log format:  [  wall | vid  ] who  message
  wall = seconds since process start, vid = video timestamp of the event.
Reading the gap between `wall` and `vid` is how you see the async decoupling
(e.g. the writer lagging behind the orchestrator).
"""
import threading
import time

_T0 = time.time()
_LOCK = threading.Lock()


def log(tag, vid_t, msg):
    with _LOCK:
        print(f"[{time.time()-_T0:6.1f}s | vid {vid_t:6.1f}s] {tag:<12} {msg}", flush=True)
