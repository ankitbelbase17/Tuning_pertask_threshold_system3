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


def _sample(logits, prev_ids, cfg, gen=None):
    """Sample one token id using the writer preset: repetition_penalty +
    presence_penalty over already-generated tokens, then temperature / top-k /
    top-p. cfg.writer_greedy -> pure argmax. `gen` is a seeded torch.Generator
    for reproducible sampling (cfg.writer_seed)."""
    logits = logits.clone()

    # penalties over tokens already produced this line
    if prev_ids and (cfg.writer_repetition_penalty != 1.0 or cfg.writer_presence_penalty != 0.0):
        for t in set(prev_ids):
            if cfg.writer_repetition_penalty != 1.0:
                logits[t] = (logits[t] / cfg.writer_repetition_penalty
                             if logits[t] > 0 else logits[t] * cfg.writer_repetition_penalty)
            logits[t] = logits[t] - cfg.writer_presence_penalty

    if cfg.writer_greedy or not cfg.writer_temperature or cfg.writer_temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / cfg.writer_temperature

    # top-k
    if cfg.writer_top_k and cfg.writer_top_k > 0:
        k = min(cfg.writer_top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    # top-p (nucleus)
    if cfg.writer_top_p and 0 < cfg.writer_top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = (cum - sp) > cfg.writer_top_p     # keep tokens up to the p mass
        sp[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, 1, generator=gen).item())


def _looping(ids, window):
    if len(ids) >= 3 and ids[-1] == ids[-2] == ids[-3]:
        return True
    if len(ids) >= 4 and ids[-2:] == ids[-4:-2]:
        return True
    return len(ids) >= window and len(set(ids[-window:])) <= 2


def _cache_to(cache, device):
    """Move a cloned cache's K/V tensors onto `device` (the writer's GPU).
    Handles transformers 5.x (`.layers`) and 4.x (`.key_cache`) layouts."""
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            if getattr(layer, "keys", None) is None:
                continue
            layer.keys = layer.keys.to(device, non_blocking=True)
            layer.values = layer.values.to(device, non_blocking=True)
    else:
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i].to(device, non_blocking=True)
            cache.value_cache[i] = cache.value_cache[i].to(device, non_blocking=True)
    return cache


def writer_thread(cfg, mgr, writer_q, stop, prof=None, evaluator=None, wb=None):
    # `wb` is the writer's backend: a 2nd-GPU replica in two-GPU mode, else the
    # shared primary backend. Generating on `wb` keeps the writer off the
    # orchestrator's GPU so the two run truly in parallel.
    b = wb if wb is not None else mgr.b
    cross_gpu = b.device != mgr.b.device
    # Seeded CPU RNG (forward returns CPU logits) for reproducible sampling.
    gen = torch.Generator().manual_seed(int(cfg.writer_seed))
    while not stop.is_set() or not writer_q.empty():
        try:
            vt = writer_q.get(timeout=0.5)
        except queue.Empty:
            continue
        t_trig = time.time()

        # MVCC snapshot: independent clone + logical position. No lock held below.
        cache, pos, phys = mgr.snapshot_clone()
        # In two-GPU mode the clone lives on the orchestrator's GPU; ship it to
        # the writer's GPU once, then generate there. The copy happens OUTSIDE
        # the cache lock, so the orchestrator keeps thinking meanwhile.
        if cross_gpu:
            cache = _cache_to(cache, b.device)

        def step(embeds):
            nonlocal pos, phys, cache
            logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
            pos += embeds.shape[1]
            phys += embeds.shape[1]
            return logits

        # splice the writer's own instruction + cue onto the snapshot, then speak
        logits = step(b.embed_text(cfg.writer_prompt + cfg.writer_cue))
        t_first = time.time()
        ids = []
        for _ in range(cfg.writer_max_tokens):
            tok_id = _sample(logits, ids, cfg, gen)
            if tok_id == b.eos_id or _looping(ids, cfg.writer_repeat_window):
                break
            # A live shout is ONE line. Stop at a newline even when the model
            # fuses it into a multi-char token ("  \n", "\n\n") that the bare
            # newline-id set misses -- check the decoded surface instead.
            if tok_id in b.newline_ids or "\n" in b.tok.decode([tok_id]):
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
