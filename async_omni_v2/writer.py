"""
writer.py — the WRITER thread (the mouth).

Waits for a trigger from the orchestrator, then free-generates a short
commentary line. Two design points:

  * It generates ONE token per `submit` at PRIO_WRITER, so the orchestrator
    preempts between every token -> the writer can never block the thinker.
    (On Mobile-O we saw the writer fall tens of video-seconds behind the
    orchestrator; that lag IS the async decoupling working.)
  * Temperature sampling + a simple anti-repeat guard, because greedy decoding
    on these prompts collapses into loops ("0 - 0 - 0 ..."). This is the fix for
    the gibberish you saw.

Its tokens are appended to the shared cache, so the orchestrator later sees what
the writer said.
"""
import queue

import torch

from manager import PRIO_WRITER
from util import log


def _sample(logits, temperature):
    if temperature and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    return int(torch.argmax(logits).item())


def _looping(ids, window):
    """Stop if the tail collapses (same token x3, or a repeated 2-gram)."""
    if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
        return True
    if len(ids) >= 4 and ids[-2:] == ids[-4:-2]:
        return True
    return len(ids) >= window and len(set(ids[-window:])) <= 2


def writer_thread(cfg, mgr, writer_q, stop):
    b = mgr.b
    while not stop.is_set() or not writer_q.empty():
        try:
            vt = writer_q.get(timeout=0.5)
        except queue.Empty:
            continue
        logits = mgr.submit(PRIO_WRITER, mgr.op_append_text(cfg.writer_cue))
        ids = []
        for _ in range(cfg.writer_max_tokens):
            tok_id = _sample(logits, cfg.writer_temperature)
            if tok_id == b.eos_id or tok_id in b.newline_ids or _looping(ids, cfg.writer_repeat_window):
                break
            ids.append(tok_id)
            logits = mgr.submit(PRIO_WRITER, mgr.op_append_token(tok_id))
        log("WRITER", vt, f"\U0001F4E2  {b.decode(ids)!r}")
    log("writer", 0.0, "writer stopped")
