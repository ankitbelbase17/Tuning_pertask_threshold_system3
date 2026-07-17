"""
input_ingester.py — the INPUT INGESTER.

It does NOT think, plan, or control. It only:
  1. seeds the shared KV cache with the system prompt (the pinned eviction sink);
  2. ingests the streaming visual tokens (from the encoder's vis_q) into that
     cache -- it is the SOLE writer of the primary cache (single-writer invariant
     => no write-write conflict), i.e. it builds the prefill/KV the controller reads;
  3. bounds memory (StreamingLLM eviction past the KV budget);
  4. publishes each frame's video time on the shared clock so the controller can
     self-pace ("check again in next_check_s of video time").

All proactivity (input gate = fps steering, output gate = emitting an answer) is
owned by the controller thread, which reads this cache via an MVCC snapshot.

DETERMINISTIC LOCKSTEP (cfg.deterministic): the clock is published only AFTER a
frame is fully in the cache, and the ingester then WAITS until the controller has
completed every tick due at <= vt (clock.get_next_check() > vt) before feeding
the next frame. Combined with the encoder's blocking put + fixed fps + greedy
decode, the whole walk is frame-indexed: every tick sees exactly frames [0..vt],
identically on every run — the async snapshot race is gone.
"""
import queue
import time

from util import log


def input_ingester_thread(cfg, mgr, vis_q, ctrl, stop, prof=None, clock=None,
                          feed_done=None):
    system_prompt = cfg.system_prompt.replace("{instruction}", cfg.instruction)
    sink = mgr.seed(system_prompt)
    log("ingester", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget} "
                         f"(deterministic={cfg.deterministic})")

    try:
        while not stop.is_set() or not vis_q.empty():
            try:
                vt, embeds = vis_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if prof is not None:
                prof.observe("visq_depth", vis_q.qsize())
                prof.incr("frames_ingested")     # frames the ingester actually wrote to cache

            # ingest streaming visual tokens (optionally prefixed by a text timestamp
            # so the model has a real-time signal, not just token order)
            if cfg.timestamp_tokens:
                mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
            mgr.ingest(embeds)

            dropped = mgr.evict()                # bounded memory
            if dropped:
                log("ingester.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

            # publish vt only now: a controller tick at vt is guaranteed to see
            # every frame up to and including vt in the cache.
            if clock is not None:
                clock.set(vt)

            # LOCKSTEP: hold the next frame until every tick due at <= vt has
            # COMPLETED, so which frames each tick sees is frame-indexed, not a
            # thread race. (Guard against a dead controller with a long timeout.)
            if cfg.deterministic and clock is not None:
                t_wait = time.time()
                while clock.get_next_check() <= vt:
                    time.sleep(0.002)
                    if time.time() - t_wait > 600:
                        log("ingester", vt, "LOCKSTEP TIMEOUT waiting on controller — continuing")
                        break
    finally:
        if feed_done is not None:
            feed_done.set()                      # tell the controller the feed is drained
    stop.set()
    log("ingester", 0.0, "input ingester stopped")
