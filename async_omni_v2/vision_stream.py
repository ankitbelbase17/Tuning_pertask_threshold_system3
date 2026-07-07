"""
vision_stream.py — the streaming VISION ENCODER (its own async component).

Unlike v2 (where this thread was CPU-only and the orchestrator did the encode),
here the encoder OWNS the vision GPU work and is a first-class concurrent
component:

  * It decodes the video and paces itself to REAL wall-clock time (so a 120 s
    clip takes ~120 s at speed 1.0) — true streaming, not batch.
  * It runs the ViT + multimodal projector (`backend.embed_frame`) and pushes
    PROJECTED visual tokens `[1, N, H]` (already in LLM space) onto `vis_q`.
    `vis_q` is the buffer "in between" the encoder and the orchestrator.
  * Its frame rate is controlled live by the orchestrator through
    `EncoderControl` — the proactive INPUT gate. Focus => encode more frames
    right now; boring => fewer. The orchestrator never has to wait for the
    encoder and vice-versa.

The encode touches NO KV cache, so it runs concurrently with the orchestrator
and writer with zero locking.
"""
import queue
import time

import av
import torch

from util import log


def encoder_thread(cfg, backend, vis_q, ctrl, stop, prof=None):
    container = av.open(cfg.video_path)
    vstream = container.streams.video[0]
    last_emit = -1e9
    wall_start = time.time()
    printed_tok = False                      # print real tokens/frame once
    for frame in container.decode(video=0):
        if stop.is_set():
            break
        vt = float(frame.pts * vstream.time_base)
        if vt > cfg.max_seconds:
            break
        # Follow the live controller-steered fps (focus/idle). In deterministic
        # eval use the fixed base fps instead (avoids the async controller->encoder
        # fps-steering race).
        interval = 1.0 / (cfg.fps if cfg.deterministic else ctrl.get_fps())
        if vt - last_emit < interval:
            continue
        last_emit = vt
        if cfg.realtime:                         # pace to wall clock
            delay = (wall_start + vt / cfg.speed) - time.time()
            if delay > 0:
                time.sleep(delay)

        img = frame.to_image()
        t = time.time()
        embeds = backend.embed_frame(img)        # ViT + projector (GPU)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if not printed_tok:                      # one-time: real tokens/frame
            ip = getattr(backend.processor, "image_processor", None)
            px_per_tok = (getattr(ip, "patch_size", 16) * getattr(ip, "merge_size", 2)) ** 2
            n_tok = embeds.shape[1]
            log("encoder", vt,
                f"FIRST FRAME: {n_tok} vision tokens/frame "
                f"(orig img {img.size[0]}x{img.size[1]}px, "
                f"~{px_per_tok}px/token, max_pixels={cfg.max_pixels})")
            printed_tok = True
        if prof is not None:
            prof.observe("vis_encode_ms", 1000 * (time.time() - t))
            prof.observe("vis_tokens_per_frame", embeds.shape[1])
            prof.incr("frames_emitted")

        if cfg.deterministic:
            vis_q.put((vt, embeds))              # block: process EVERY frame -> reproducible
        else:
            try:
                vis_q.put_nowait((vt, embeds))
            except queue.Full:
                try:
                    vis_q.get_nowait()           # drop oldest (bounded latency)
                    if prof is not None:
                        prof.incr("frames_dropped")
                except queue.Empty:
                    pass
                vis_q.put_nowait((vt, embeds))
    container.close()
    stop.set()
    log("encoder", cfg.max_seconds, "video stream ended")
