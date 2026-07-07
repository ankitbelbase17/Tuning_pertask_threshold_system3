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
"""
import queue

from util import log


def input_ingester_thread(cfg, mgr, vis_q, ctrl, stop, prof=None, clock=None):
    system_prompt = cfg.system_prompt.replace("{instruction}", cfg.instruction)
    sink = mgr.seed(system_prompt)
    log("ingester", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget}")

    while not stop.is_set() or not vis_q.empty():
        try:
            vt, embeds = vis_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if clock is not None:
            clock.set(vt)                    # publish latest video time for the controller
        if prof is not None:
            prof.observe("visq_depth", vis_q.qsize())

        # ingest streaming visual tokens (optionally prefixed by a text timestamp
        # so the model has a real-time signal, not just token order)
        if cfg.timestamp_tokens:
            mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
        mgr.ingest(embeds)

        dropped = mgr.evict()                # bounded memory
        if dropped:
            log("ingester.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

    stop.set()
    log("ingester", 0.0, "input ingester stopped")
