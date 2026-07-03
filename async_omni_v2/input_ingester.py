"""
input_ingester.py — the INPUT INGESTER (formerly misnamed "orchestrator").

Despite the old name, this thread does NOT think, plan, or control. It only:
  1. ingests the streaming visual tokens (from the encoder's vis_q) into the
     shared KV cache -- it is the SOLE writer of the primary cache (single-writer
     invariant => no write-write conflict), i.e. it builds the prefill/KV the
     writer later reads;
  2. bounds memory (StreamingLLM eviction past the KV budget);
  3. INPUT gate (optional): probe "is this important?" and steer the encoder fps
     (focus/idle). Disabled with cfg.input_gate=False -> the encoder feeds every
     frame at the fixed base fps (all frames fed);
  4. OUTPUT gate (optional): probe "did a goal happen?" and, on a confident +
     debounced yes, trigger the writer.

So it is an ingester + prefill builder + probe answerer -- not a controller.
Real "thinking" (free-text reasoning that persists and conditions decisions) is
a deliberate future component, kept separate from this ingest loop.
"""
import queue

from util import log


def input_ingester_thread(cfg, mgr, vis_q, writer_q, ctrl, stop, prof=None, evaluator=None):
    sink = mgr.seed(cfg.system_prompt)
    log("ingester", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget} "
                         f"(input_gate={cfg.input_gate}, output_gate={cfg.output_gate})")
    n_frames = 0
    last_trigger_vt = -1e9
    armed = True                         # hysteresis: ready to fire on a rising edge

    while not stop.is_set() or not vis_q.empty():
        try:
            vt, embeds = vis_q.get(timeout=0.5)
        except queue.Empty:
            continue
        n_frames += 1
        if prof is not None:
            prof.observe("visq_depth", vis_q.qsize())

        # 1. ingest streaming visual tokens (optionally prefixed by a text
        #    timestamp so the model has a real-time signal, not just token order)
        if cfg.timestamp_tokens:
            mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
        mgr.ingest(embeds)

        # 2. bounded memory
        dropped = mgr.evict()
        if dropped:
            log("ingester.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

        # 3. INPUT gate -> steer the encoder's frame rate. Skipped entirely when
        #    cfg.input_gate is False; the encoder then feeds all frames at cfg.fps.
        if cfg.input_gate and n_frames % cfg.vision_gate_every == 0:
            share = mgr.probe(cfg.vision_question, "probe.vision")
            target = cfg.encoder_focus_fps if share >= 0.5 else cfg.encoder_idle_fps
            ctrl.set_fps(target)
            log("ingester.vgate", vt,
                f"important={share:.2f} -> encoder fps={ctrl.get_fps():.1f}")

        # 4. OUTPUT gate -> trigger the writer (one forward pass; gated to every
        #    goal_gate_every frames since it's the dominant GPU op).
        if cfg.output_gate and n_frames % cfg.goal_gate_every == 0:
            share = mgr.probe(cfg.goal_question, "probe.goal")
            # log_gate_every == -1 -> only log when the probe crosses threshold;
            # otherwise log every N probes.
            if cfg.log_gate_every == -1:
                do_log = share >= cfg.goal_threshold
            else:
                do_log = (n_frames % (cfg.goal_gate_every * cfg.log_gate_every) == 0)
            if do_log:
                log("ingester.ggate", vt, f"goal yes_share={share:.2f}")
            # Decide whether this probe fires.
            if cfg.gate_hysteresis:
                # edge-triggered: re-arm when the signal falls below low_thr (narrow
                # band) OR gate_rearm_s after the last fire (time-based, for recurring
                # events whose signal stays high), then fire on the rising crossing.
                if not armed and (share < cfg.gate_low_thr or
                                  (cfg.gate_rearm_s > 0
                                   and (vt - last_trigger_vt) >= cfg.gate_rearm_s)):
                    armed = True
                fire = (armed and share >= cfg.gate_high_thr
                        and (vt - last_trigger_vt) > cfg.debounce_s)
                if fire:
                    armed = False
            else:
                fire = share >= cfg.goal_threshold and (vt - last_trigger_vt) > cfg.debounce_s
            if fire:
                last_trigger_vt = vt
                log("ingester", vt, f">>> GOAL suspected (yes_share={share:.2f}) -> trigger writer")
                if evaluator is not None:
                    evaluator.record_trigger(vt, share)
                if cfg.deterministic:
                    writer_q.put(vt)             # block: writer sees every trigger in order
                    writer_q.join()              # AND wait for it to finish -> its snapshot
                                                 # is taken while we're paused -> reproducible
                else:
                    try:
                        writer_q.put_nowait(vt)  # drop if writer busy (stay live)
                    except queue.Full:
                        pass

    stop.set()
    log("ingester", 0.0, "input ingester stopped")
