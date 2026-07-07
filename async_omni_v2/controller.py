"""
controller.py — the PURE-GENERATIVE proactivity controller (icl_ingester_writer).

Replaces the fixed-cadence input/output gates with a single agentic loop. The
controller IS the writer: because it reads the shared KV cache (via an MVCC
snapshot) it already knows what's happening, so it acts as the orchestrator.

Each cycle:
  1. wait until video time reaches the self-scheduled next-check point;
  2. snapshot the shared cache and generate ONE control JSON, e.g.
       {"fps": 3, "have_enough_info": true, "new_event": true,
        "answer": "A goal was scored.", "next_check_s": 5}
  3. apply it: steer encoder fps (input gate); if have_enough_info AND new_event
     (and not a repeat of the last answer), emit the answer to the user (output
     gate + writer in one); schedule the next check at vt + next_check_s.
  4. loop — go back to reading the incoming video until the next check.

Over-firing is minimized by the model's own new_event dedup + chosen cadence,
plus a local last-answer guard. It never writes back to the primary cache
(single-writer invariant preserved; "no writer cache").
"""
import json
import re
import time

import torch

from util import log


def _sample(logits, prev_ids, cfg, gen=None):
    """Sample one token id using the controller preset: repetition_penalty +
    presence_penalty over already-generated tokens, then temperature / top-k /
    top-p. cfg.writer_greedy -> pure argmax. `gen` is a seeded torch.Generator
    for reproducible sampling (cfg.writer_seed)."""
    logits = logits.clone()

    if prev_ids and (cfg.writer_repetition_penalty != 1.0 or cfg.writer_presence_penalty != 0.0):
        for t in set(prev_ids):
            if cfg.writer_repetition_penalty != 1.0:
                logits[t] = (logits[t] / cfg.writer_repetition_penalty
                             if logits[t] > 0 else logits[t] * cfg.writer_repetition_penalty)
            logits[t] = logits[t] - cfg.writer_presence_penalty

    if cfg.writer_greedy or not cfg.writer_temperature or cfg.writer_temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / cfg.writer_temperature

    if cfg.writer_top_k and cfg.writer_top_k > 0:
        k = min(cfg.writer_top_k, logits.numel())
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = float("-inf")

    probs = torch.softmax(logits, dim=-1)

    if cfg.writer_top_p and 0 < cfg.writer_top_p < 1.0:
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        drop = (cum - sp) > cfg.writer_top_p
        sp[drop] = 0.0
        probs = torch.zeros_like(probs).scatter_(0, si, sp)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, 1, generator=gen).item())


def _cache_to(cache, device):
    """Move a cloned cache's K/V tensors onto `device` (the controller's GPU).
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


def _extract_json(text):
    """Pull the first flat {...} object out of the model's text; {} on failure."""
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def _clamp(x, lo, hi, default):
    try:
        return max(lo, min(hi, float(x)))
    except (TypeError, ValueError):
        return default


def controller_thread(cfg, mgr, ctrl, clock, stop, prof=None, evaluator=None, wb=None):
    b = wb if wb is not None else mgr.b
    cross_gpu = b.device != mgr.b.device
    gen = torch.Generator().manual_seed(int(cfg.writer_seed))
    log("controller", 0.0, "model-scheduled proactivity ON (pure-generative control loop)")

    next_check_vt = 0.0
    last_answer = None
    pending_q = ""                             # deferred-verification question to re-check

    while not stop.is_set():
        vt = clock.get()
        if vt < next_check_vt:                 # not time to check yet — keep reading
            time.sleep(0.05)
            continue

        t0 = time.time()
        cache, pos, phys = mgr.snapshot_clone()
        if cross_gpu:
            cache = _cache_to(cache, b.device)

        def step(embeds):
            nonlocal pos, phys, cache
            logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
            pos += embeds.shape[1]
            phys += embeds.shape[1]
            return logits

        # generate the control JSON; if we deferred a check last time, focus on it
        # (this is the "ask me again in Ns: did X happen?" verification loop).
        prompt = cfg.controller_prompt
        if pending_q:
            prompt = (f"\nYou previously deferred this check: '{pending_q}'. "
                      f"Judge it from the MOST RECENT frames." + prompt)
        logits = step(b.embed_text(prompt))
        ids, saw_open = [], False
        for _ in range(cfg.controller_max_tokens):
            tok_id = _sample(logits, ids, cfg, gen)
            if tok_id == b.eos_id:
                break
            ids.append(tok_id)
            piece = b.tok.decode([tok_id])
            if "{" in piece:
                saw_open = True
            if saw_open and "}" in piece:
                break
            logits = step(b.embed_token(tok_id))

        ctl = _extract_json(b.decode(ids))
        gen_s = time.time() - t0

        # ---- apply the control config ----
        # input gate: steer encoder fps
        fps = _clamp(ctl.get("fps"), cfg.encoder_idle_fps, cfg.encoder_focus_fps,
                     cfg.encoder_idle_fps)
        ctrl.set_fps(fps)
        # schedule next check (self-paced), clamped so it can't spin or stall
        nxt = _clamp(ctl.get("next_check_s"), cfg.probe_min_s, cfg.probe_max_s,
                     cfg.probe_default_s)
        next_check_vt = vt + nxt

        have = bool(ctl.get("have_enough_info"))
        new = bool(ctl.get("new_event"))
        answer = (ctl.get("answer") or "").strip()
        pending_q = (ctl.get("question") or "").strip()   # carry forward what to re-check
        log("ctrl.gate", vt, f"fps={fps:.1f} have_info={have} new={new} "
                             f"next={nxt:.1f}s q={pending_q!r}")

        # output gate + writer in one: emit only on a NEW, non-repeat event
        if have and new and answer and answer != last_answer:
            last_answer = answer
            pending_q = ""                       # event resolved -> drop the deferred check
            if evaluator is not None:
                evaluator.record_trigger(vt, 1.0)
                evaluator.record_write(vt, answer, gen_s)
            log("CONTROLLER", vt, f"\U0001F4E2  {answer!r}")
        if prof is not None:
            prof.observe("controller_gen_s", gen_s)
            prof.observe("controller_tokens", len(ids))

    log("controller", 0.0, "controller stopped")
