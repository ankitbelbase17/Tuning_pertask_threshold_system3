"""
vision_stream.py — the streaming vision encoder thread (the pacemaker).

Genuinely parallel, CPU-only: decodes the video with PyAV, samples at `fps`,
and pushes (video_time, PIL frame) onto a bounded queue. It NEVER touches the
GPU — the heavy vision-tower work happens later in the manager thread when the
orchestrator chooses to ingest a frame.

Real-time clock: if `realtime`, each frame is held until wall-clock has reached
its video timestamp (scaled by `speed`). This makes "async at its own pace"
literally true against a wall clock and turns the vision stream into the
pacemaker for the whole pipeline. With realtime off, frames flow as fast as
decoding allows (batch mode). If the orchestrator can't keep up, the oldest
queued frame is dropped (real streaming, bounded latency).
"""
import queue
import time

import av

from util import log


def vision_thread(cfg, frame_q, stop):
    container = av.open(cfg.video_path)
    vstream = container.streams.video[0]
    interval = 1.0 / cfg.fps
    last_t = -1e9
    wall_start = time.time()
    for frame in container.decode(video=0):
        if stop.is_set():
            break
        vt = float(frame.pts * vstream.time_base)
        if vt > cfg.max_seconds:
            break
        if vt - last_t < interval:
            continue
        last_t = vt
        if cfg.realtime:
            delay = (wall_start + vt / cfg.speed) - time.time()
            if delay > 0:
                time.sleep(delay)
        img = frame.to_image()
        try:
            frame_q.put_nowait((vt, img))
        except queue.Full:
            try:
                frame_q.get_nowait()          # drop oldest
            except queue.Empty:
                pass
            frame_q.put_nowait((vt, img))
    container.close()
    stop.set()
    log("vision", cfg.max_seconds, "video stream ended")
