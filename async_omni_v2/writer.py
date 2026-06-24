"""
writer.py — the WRITER (the mouth), now fully decoupled.

On a trigger it takes an MVCC SNAPSHOT of the shared cache (an independent
clone) and free-generates a short commentary line ON THE CLONE. Because the
clone is private:

  * it holds NO lock while generating, so it never blocks the orchestrator and
    the orchestrator never blocks it;
  * the orchestrator's concurrent eviction/ingest cannot rip context out from
    under it mid-sentence (the v2 corruption / gibberish bug);
  * its tokens are NOT written back into the primary cache, preserving the
    single-writer invariant. (Feeding a compact summary back to the orchestrator
    is a deliberate future step, not done here.)

Temperature sampling + an anti-repeat guard keep greedy loops from collapsing.
On one GPU the writer's forwards still time-share kernels with the orchestrator;
move it to a 2nd GPU replica and it becomes truly parallel — no code change to
the concurrency model, only where the snapshot's tensors live.
"""
import queue
import time

import torch

from util import log


def _sample(logits, temperature):
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.argmax(logits).item())


def _looping(ids, window):
    if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
        return True
    if len(ids) >= 4 and ids[-2:] == ids[-4:-2]:
        return True
    return len(ids) >= window and len(set(ids[-window:])) <= 2


def writer_thread(cfg, mgr, writer_q, stop, prof=None, evaluator=None):
    b = mgr.b
    while not stop.is_set() or not writer_q.empty():
        try:
            vt = writer_q.get(timeout=0.5)
        except queue.Empty:
            continue
        t_trig = time.time()

        # MVCC snapshot: independent clone + logical position. No lock held below.
        cache, pos, phys = mgr.snapshot_clone()

        def step(embeds):
            nonlocal pos, phys, cache
            logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
            pos += embeds.shape[1]
            phys += embeds.shape[1]
            return logits

        logits = step(b.embed_text(cfg.writer_cue))
        t_first = time.time()
        ids = []
        for _ in range(cfg.writer_max_tokens):
            tok_id = _sample(logits, cfg.writer_temperature)
            if tok_id == b.eos_id or tok_id in b.newline_ids or _looping(ids, cfg.writer_repeat_window):
                break
            ids.append(tok_id)
            tt = time.time()
            logits = step(b.embed_token(tok_id))
            if prof is not None:
                prof.observe("writer_token_ms", 1000 * (time.time() - tt))

        total = time.time() - t_trig
        if prof is not None:
            prof.observe("writer_trig2first_s", t_first - t_trig)
            prof.observe("writer_total_s", total)
            prof.observe("writer_tokens", len(ids))
        text = b.decode(ids)
        if evaluator is not None:
            evaluator.record_write(vt, text, total)
        log("WRITER", vt, f"\U0001F4E2  {text!r}")
    log("writer", 0.0, "writer stopped")
