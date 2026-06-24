"""
orchestrator.py — the async THINKER / brain.

Continuously consumes the streaming visual tokens (from the encoder's `vis_q`),
appends them to the shared KV cache, and thinks. It is the SOLE writer of the
primary cache (single-writer invariant => no write-write conflict). Each frame:

  1. ingest the projected visual tokens into the shared cache;
  2. bound memory (StreamingLLM eviction past the KV budget);
  3. INPUT proactivity: every N frames probe "is this important?" and steer the
     encoder's fps up/down via EncoderControl (focus vs idle);
  4. OUTPUT proactivity: probe "did a goal just happen?" and, on a confident +
     debounced yes, trigger the writer.

It never blocks: ingest/probe hold the cache lock only briefly, and the writer
works off an independent snapshot, so the orchestrator keeps thinking while the
writer speaks.
"""
import queue

from util import log


def orchestrator_thread(cfg, mgr, vis_q, writer_q, ctrl, stop, prof=None, evaluator=None):
    sink = mgr.seed(cfg.system_prompt)
    log("orch", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget}")
    n_frames = 0
    last_trigger_vt = -1e9

    while not stop.is_set() or not vis_q.empty():
        try:
            vt, embeds = vis_q.get(timeout=0.5)
        except queue.Empty:
            continue
        n_frames += 1
        if prof is not None:
            prof.observe("visq_depth", vis_q.qsize())

        # 1. ingest streaming visual tokens
        mgr.ingest(embeds)

        # 2. bounded memory
        dropped = mgr.evict()
        if dropped:
            log("orch.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

        # 3. INPUT proactivity -> steer the encoder's frame rate
        if n_frames % cfg.vision_gate_every == 0:
            share = mgr.probe(cfg.vision_question, "probe.vision")
            target = cfg.encoder_focus_fps if share >= 0.5 else cfg.encoder_idle_fps
            ctrl.set_fps(target)
            log("orch.vgate", vt,
                f"important={share:.2f} -> encoder fps={ctrl.get_fps():.1f}")

        # 4. OUTPUT proactivity -> trigger the writer
        share = mgr.probe(cfg.goal_question, "probe.goal")
        if n_frames % cfg.log_gate_every == 0:
            log("orch.ggate", vt, f"goal yes_share={share:.2f}")
        if share >= cfg.goal_threshold and (vt - last_trigger_vt) > cfg.debounce_s:
            last_trigger_vt = vt
            log("orch", vt, f">>> GOAL suspected (yes_share={share:.2f}) -> trigger writer")
            if evaluator is not None:
                evaluator.record_trigger(vt, share)
            try:
                writer_q.put_nowait(vt)
            except queue.Full:
                pass

    stop.set()
    log("orch", 0.0, "orchestrator stopped")
