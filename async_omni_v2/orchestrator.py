"""
orchestrator.py — the decoupled THINKER thread.

Owns all the proactive decisions; runs at PRIO_ORCH so it always wins the GPU
over the writer. Per frame it:
  1. (every N frames) runs the VISION gate probe "should I keep looking?" and
     proactively skips the frame if the model says no;
  2. ingests the frame into the shared cache (vision tower runs here, in the
     manager thread);
  3. evicts old vision tokens past the KV budget (bounded memory);
  4. runs the WRITER gate probe "did a goal just happen?" and, on a confident +
     debounced yes, triggers the writer.

It never generates the user-facing text itself — it only decides *when* the
writer should speak. The writer's tokens land in the same shared cache, so on
later frames the orchestrator implicitly "sees what the writer already said".
"""
import queue

from manager import PRIO_ORCH
from util import log


def orchestrator_thread(cfg, mgr, frame_q, writer_q, stop):
    sink = mgr.submit(PRIO_ORCH, mgr.op_seed(cfg.system_prompt))
    log("orch", 0.0, f"seeded cache, sink={sink} tokens")
    n_frames = 0
    last_trigger_vt = -1e9

    while not stop.is_set() or not frame_q.empty():
        try:
            vt, img = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        n_frames += 1

        # 1. proactive vision gate
        if n_frames % cfg.vision_gate_every == 0:
            share = mgr.submit(PRIO_ORCH, mgr.op_probe(cfg.vision_question))
            look = share >= 0.5
            log("orch.vgate", vt, f"look-closer? yes_share={share:.2f} -> {look}")
            if not look:
                continue

        # 2. ingest frame (vision tower + LLM forward happen in the manager)
        mgr.submit(PRIO_ORCH, mgr.op_ingest_frame(img))

        # 3. bounded memory
        dropped = mgr.submit(PRIO_ORCH, mgr.op_evict(cfg.kv_budget, sink))
        if dropped:
            log("orch.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

        # 4. writer gate
        share = mgr.submit(PRIO_ORCH, mgr.op_probe(cfg.goal_question))
        if n_frames % cfg.log_gate_every == 0:
            log("orch.ggate", vt, f"goal yes_share={share:.2f}")
        if share >= cfg.goal_threshold and (vt - last_trigger_vt) > cfg.debounce_s:
            last_trigger_vt = vt
            log("orch", vt, f">>> GOAL suspected (yes_share={share:.2f}) -> trigger writer")
            try:
                writer_q.put_nowait(vt)
            except queue.Full:
                pass

    stop.set()
    log("orch", 0.0, "orchestrator stopped")
