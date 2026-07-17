"""
controller.py — the PURE-GENERATIVE proactivity controller (icl_ingester_writer).

Replaces the fixed-cadence input/output gates with a single agentic loop. The
controller IS the writer: because it reads the shared KV cache (via an MVCC
snapshot) it already knows what's happening, so it acts as the orchestrator.

Each cycle:
  1. wait until video time reaches the self-scheduled next-check point;
  2. snapshot the shared cache and generate ONE control JSON, e.g.
       {"fps": 3, "have_enough_info": true, "new_event": true,
        "answer": "The target event just occurred.", "next_check_s": 5}
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

    # penalties over already-generated tokens — VECTORIZED (a few kernels) instead
    # of a Python loop of per-index scalar writes, so it stays cheap on the GPU.
    if prev_ids and (cfg.writer_repetition_penalty != 1.0 or cfg.writer_presence_penalty != 0.0):
        idx = torch.tensor(sorted(set(prev_ids)), device=logits.device, dtype=torch.long)
        if cfg.writer_repetition_penalty != 1.0:
            v = logits[idx]
            logits[idx] = torch.where(v > 0, v / cfg.writer_repetition_penalty,
                                      v * cfg.writer_repetition_penalty)
        if cfg.writer_presence_penalty != 0.0:
            logits[idx] -= cfg.writer_presence_penalty

    if cfg.writer_greedy or not cfg.writer_temperature or cfg.writer_temperature <= 0:
        return int(torch.argmax(logits).item())     # GPU argmax; only the id crosses

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


def _word_sim(a, b):
    """Jaccard word overlap in [0,1]; rewordings of the SAME occurrence score high
    ('reports 80 dead' vs 'now reports 80 dead'), different occurrences score low
    ('match date August 14th' vs 'ticket costs and purchase website')."""
    wa = set(re.findall(r"[a-z0-9]+", a.lower()))
    wb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def controller_thread(cfg, mgr, ctrl, clock, stop, prof=None, evaluator=None, wb=None,
                      feed_done=None):
    b = wb if wb is not None else mgr.b
    cross_gpu = b.device != mgr.b.device
    # generator on the backend's device: logits now stay on GPU, so multinomial
    # sampling (non-greedy path) needs a matching-device generator.
    try:
        gen = torch.Generator(device=b.device).manual_seed(int(cfg.writer_seed))
    except (RuntimeError, TypeError):
        gen = torch.Generator().manual_seed(int(cfg.writer_seed))
    log("controller", 0.0, "model-scheduled proactivity ON (pure-generative control loop)")

    next_check_vt = cfg.probe_min_s           # skip the empty-cache tick at t=0
    reported = []                              # conversation history: (vt, answer) already emitted
    pending_q = ""                             # question_for_next: what to verify on the next tick
    # DIFF-MERGE: keep a persistent control config; the model emits only the fields
    # that CHANGE each tick (a compact JSON diff), so it rarely decodes the string
    # fields -> much less latency. Transient fields (have_enough_info/answer/seen/
    # event_time_s) RESET to default every tick and must be re-asserted.
    state = {"fps": cfg.encoder_idle_fps, "next_check_s": cfg.probe_default_s,
             "have_enough_info": False, "new_event": False, "answer": "",
             "question_for_next": "", "seen": "", "event_time_s": None}
    # LEVEL -> EDGE in CODE (semantic Schmitt gate): the model reports whether the
    # condition is satisfied NOW (a LEVEL it can judge from the cache snapshot);
    # firing happens only on the RISING edge (false -> true across ticks), so a
    # condition that stays on screen cannot re-fire. Within a true stretch, a fire
    # is also allowed when the answer describes a clearly DIFFERENT occurrence
    # (low word overlap with the last fired answer) — e.g. date poster then ticket
    # prices with no gap in between.
    prev_level = False
    clock.set_next_check(next_check_vt)        # LOCKSTEP: tell the ingester when the
                                               # first tick is due (it waits on this)

    while True:
        vt = clock.get()
        due = vt >= next_check_vt
        # Exit: stop requested AND nothing due. In lockstep the ingester holds the
        # stream while a tick is due, so we must keep servicing ticks during the
        # drain and only leave once the feed is fully done (feed_done set by the
        # ingester). Without feed_done (standalone/async), plain stop suffices.
        if stop.is_set() and not due and (feed_done is None or feed_done.is_set()):
            break
        if not due:                            # not time to check yet — keep reading
            time.sleep(0.005 if cfg.deterministic else 0.05)
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

        # Build the controller prompt: task ICL + (optional) deferred check +
        # CONVERSATION HISTORY of what has already been reported, so the model can
        # set new_event=false for repeats and only fire on genuinely new details.
        prompt = cfg.controller_prompt.rstrip()
        if pending_q:
            prompt += (f"\nYou previously asked yourself: '{pending_q}'. "
                       f"Judge it now from the MOST RECENT frames.")
        # timestamped history: a LATER onset is a new event, not a repeat of these
        convo = "".join(f"assistant @{rvt:.0f}s: {a}\n" for rvt, a in reported) or "assistant: none\n"
        prompt += ("\n\nAlready reported so far (PAST occurrences with their times; a fresh "
                   "onset at a later time is a NEW event — set new_event=false ONLY while the "
                   "SAME occurrence is still on screen):\n" + convo)
        prompt += "\nNow emit ONLY your control JSON for the current stream:\n"

        # PRIME the decoder with an open brace: Qwen3-VL is an instruct model and,
        # spliced as raw text onto the cache (no assistant-turn markers), it would
        # otherwise emit EOS immediately at the splice point. Starting mid-object
        # forces it to complete the JSON. We reconstruct raw = "{" + generated.
        t_prefill0 = time.time()
        logits = step(b.embed_text(prompt + "{"))
        prefill_s = time.time() - t_prefill0      # ICL+convo prefill cost this tick
        ids = []
        t_dec0 = time.time()
        for _ in range(cfg.controller_max_tokens):
            # MASK EOS: as an instruct model spliced raw onto the cache, Qwen often
            # samples the end token as the very first token (-> empty output). We
            # stop on the closing "}" ourselves, so EOS is never wanted here.
            logits[b.eos_id] = float("-inf")
            tok_id = _sample(logits, ids, cfg, gen)
            ids.append(tok_id)
            if "}" in b.tok.decode([tok_id]):     # first close -> flat object done
                break
            logits = step(b.embed_token(tok_id))
        decode_s = time.time() - t_dec0
        if prof is not None:
            prof.observe("ctrl_prefill_s", prefill_s)
            prof.observe("ctrl_decode_s", decode_s)
            if ids:
                prof.observe("ctrl_decode_ms_per_tok", 1000 * decode_s / len(ids))

        raw = "{" + b.decode(ids)
        diff = _extract_json(raw)                 # the model's DIFF (partial dict); {} = no change
        gen_s = time.time() - t0

        # ---- apply the DIFF onto the persistent config ----
        # reset the transient fields first (they only hold this tick), then merge
        # whatever the model re-stated; persistent fields (fps/next_check_s/question)
        # survive.
        state["have_enough_info"] = False
        state["answer"] = ""
        state["seen"] = ""
        state["event_time_s"] = None
        for k, v in diff.items():
            key = "question_for_next" if k == "question" else k   # accept legacy key
            if key in state:
                state[key] = v

        fps = _clamp(state["fps"], cfg.encoder_idle_fps, cfg.encoder_focus_fps,
                     cfg.encoder_idle_fps)
        ctrl.set_fps(fps)
        nxt = _clamp(state["next_check_s"], cfg.probe_min_s, cfg.probe_max_s,
                     cfg.probe_default_s)
        next_check_vt = vt + nxt

        level = bool(state["have_enough_info"])   # "condition satisfied NOW"
        answer = (state["answer"] or "").strip()
        pending_q = (state.get("question_for_next") or "").strip()

        # fire decision: rising edge of the level signal; OR, within a true
        # stretch, an answer that is clearly a DIFFERENT occurrence than the last
        # fired one (low word overlap) — back-to-back distinct events, no gap.
        rising = level and not prev_level
        distinct = (level and prev_level and answer and reported
                    and _word_sim(answer, reported[-1][1]) < 0.5)
        fire = bool(answer) and (rising or distinct)

        # log EVERY tick's raw diff so all responses are inspectable in the log file
        vid = cfg.video_id or "?"
        log("ctrl.raw", vt, f"[{vid}] " + (raw.strip()[:240] if diff else f"NO-DIFF raw={raw[:120]!r}"))
        log("ctrl.gate", vt, f"[{vid}] fps={fps:.1f} level={level} rise={rising} "
                             f"new_occ={distinct} fire={fire} next={nxt:.1f}s "
                             f"gen={gen_s:.1f}s ntok={len(ids)} q={pending_q!r}")

        if fire:
            # the onset may predate this tick; if the model read the event time off
            # the in-context "time Xs" markers, record that (clamped to recent past)
            t_rec = vt
            try:
                ev = state.get("event_time_s")
                if ev is not None:
                    t_rec = min(vt, max(vt - 10.0, float(ev)))
            except (TypeError, ValueError):
                pass
            reported.append((t_rec, answer))
            if evaluator is not None:
                evaluator.record_trigger(t_rec, 1.0)
                evaluator.record_write(t_rec, answer, gen_s)
            log("CONTROLLER", vt, f"[{vid}] \U0001F4E2 @{t_rec:.1f}s  {answer!r}")

        # latch the level ONLY when backed by an answer: a bare true (no answer)
        # must not swallow the edge — the next answered tick can still fire.
        if not level:
            prev_level = False
        elif answer:
            prev_level = True
        if prof is not None:
            prof.observe("controller_gen_s", gen_s)
            prof.observe("controller_tokens", len(ids))
        # LOCKSTEP: publish only now, after the fire + history update completed —
        # the waiting ingester may feed the next frame from this instant on.
        clock.set_next_check(next_check_vt)

    log("controller", 0.0, "controller stopped")
