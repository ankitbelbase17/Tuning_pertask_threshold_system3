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
import math
import re
import time

import torch

from util import log

# =============================================================================
# TODO-7BC — bring back `note`, `count`, `phase` (PARKED 2026-08-12)
# =============================================================================
# All three are commented out, here and in prompts.py. This is a PARK, not a
# deletion: each one is the mechanism for a pillar we still claim, and the
# measurement says the mechanism was never actually wired, NOT that the idea is
# wrong. Deleting them would quietly delete three claims from §7.
#
# WHY EACH SHOULD EVENTUALLY ADD VALUE, and what has to change first:
#
# `note` — pillar 7b, the append-only trace ("the essence of what it thought at
#   each moment"). The value: the controller's only durable memory today is
#   `reported`, i.e. what it SAID. A thought it had but did not speak is lost, so
#   it cannot carry a suspicion across ticks ("someone is approaching the door")
#   and re-decides from scratch every time. That is the memory 7b is for.
#   Blocker: fill rate 0/199,909. The consumer (the "NOTES YOU KEPT" prompt
#   block) exists; NO PROMPT EVER TEACHES THE FIELD — `note` appears in no
#   format block in prompts.py. It never had a fair test. Before re-enabling:
#   teach it in the format block, then measure fill; a field the model is never
#   told about is not evidence about whether the model wants it.
#
# `count` — the running total for the three counting tasks. The value: those
#   tasks are scored on an integer the model must carry ACROSS ticks, and today
#   it re-derives that integer from `reported` text on every tick. A real
#   accumulator is the difference between counting and re-counting.
#   Blocker: WRITE-ONLY. 16.2% fill and the only reader is a log line, so the
#   model's carried count cannot influence the answer it emits. Before
#   re-enabling: write the consumer FIRST — feed the accumulator into the prompt
#   (and/or into answer construction) — then re-enable the field. Fill rate was
#   never the problem here; the missing half was the read.
#
# `phase` — the current-state label for realtime_state_monitor. The value: that
#   task is "tell me when the state CHANGES", which is undefined without a
#   memory of the state you last reported. `phase` is that memory, and the
#   prompt built the whole have_enough_info rule on comparing screen vs `phase`.
#   Blocker: two, and both are structural. (1) No consumer — init only, not even
#   logged, so the comparison the prompt describes happens only inside the
#   model's head. (2) It lives in the `more` tail, gated on p_more >= 0.5, so a
#   field the prompt says to restate on EVERY tick is reachable ~2.3% of ticks.
#   Before re-enabling: move it out of the tail (it is 1-4 words on one task, so
#   a task-conditional schema slot is affordable) AND wire the comparison in
#   code, rather than asking the model to remember its own last answer.
#
# The common lesson, and the reason all three are parked together: on every one
# of them we shipped the WRITE and never shipped the READ. Re-enable one at a
# time, consumer first, and measure fill + effect per field — MISSION §10 Todo.
# =============================================================================


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


def _single_tok(tok, s):
    """Token id for a surface form that MUST be one token; None otherwise."""
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def _read_bool(logits, true_id, false_id):
    """P(true) restricted to {true,false} at the CURRENT position — a logit read,
    not a decode. Costs zero extra forward passes: these logits were already
    produced by the forward that force-fed the key. Returns a CONTINUOUS
    confidence, which is what lets the tuned Schmitt/hysteresis gate apply here at
    all (a decoded 'true'/'false' token gives only a hard bit)."""
    lt, lf = float(logits[true_id]), float(logits[false_id])
    m = max(lt, lf)
    et, ef = math.exp(lt - m), math.exp(lf - m)
    return et / (et + ef)


def _decode_until(step, b, logits, cfg, gen, stop_chars, max_tokens):
    """Sample a VALUE slot until a stop char appears (or the cap is hit).

    The stop token is deliberately NOT fed back into the cache — the caller
    force-feeds the next literal (which begins with that same delimiter) instead,
    so the JSON stays well-formed by construction."""
    ids, text = [], ""
    for _ in range(max_tokens):
        logits[b.eos_id] = float("-inf")
        tid = _sample(logits, ids, cfg, gen)
        piece = b.tok.decode([tid])
        stops = [c for c in stop_chars if c in piece]
        if stops:
            text += piece[:min(piece.index(c) for c in stops)]
            return text, ids, logits
        ids.append(tid)
        text += piece
        logits = step(b.embed_token(tid))
    return text, ids, logits


def _schema_tick(b, cfg, gen, step, prompt, ids_bool):
    """SCHEMA-WALKED DECODE — the fix for "the diff never actually diffs".

    Instead of handing the model an open brace and hoping it obeys prose rules
    ("omit fps unless it changed" — measured 0/15 compliance), the CODE walks a
    fixed skeleton and the model only fills value slots:

        force  {"seen":"          4 tokens, ONE forward (prefill is parallel)
        SAMPLE <seen text>        the only free decode on a quiet tick
        force  ","have_enough_info":
        READ   P(true)            ZERO decode steps, continuous confidence
        [if hit] force ,"event_time_s":  SAMPLE int
                 force ,"answer":"       SAMPLE text
        force  ","more":
        READ   P(true)            escape hatch; if true, free-decode the tail
                                  (fps / next_check_s)   # note/count/phase: PARKED

    Prefill of k tokens costs ONE forward; decoding k tokens costs k forwards.
    So every forced token is ~free. A quiet tick drops from 35 sequential decodes
    to ~6 decodes + ~4 batched prefills, and the model becomes PHYSICALLY UNABLE
    to emit fps/next_check_s on the default path.

    Returns (diff, meta)."""
    true_id, false_id = ids_bool
    diff, meta = {}, {}
    n_dec = 0

    # ---- seen: forced key, sampled value (look BEFORE judging, every tick) ----
    # SEEN_MODE is the Priority-1 experiment (ROADMAP): does the hit read actually
    # NEED `seen` decoded first, or does only the ANSWER need it?
    #   "before" -- current: describe the scene, THEN read the level (~1.3s/tick)
    #   "off"    -- read the level immediately, no decode at all (~0.15s/tick)
    #   "after"  -- read the level FIRST, then describe. Separates "does the
    #               perception step help?" from "does its ORDER matter?"
    # `seen` took F1 from 0.0 to 0.255 when it was introduced, so this is not a
    # refactor to be assumed safe — it is measured.
    if cfg.seen_mode == "before":
        logits = step(b.embed_text(prompt + '{"seen":"'))
        seen, sids, logits = _decode_until(step, b, logits, cfg, gen,
                                           '"', cfg.schema_max_seen_tokens)
        n_dec += len(sids)
        diff["seen"] = seen.strip()
        logits = step(b.embed_text('","have_enough_info":'))
    else:
        logits = step(b.embed_text(prompt + '{"have_enough_info":'))
    p_hit = _read_bool(logits, true_id, false_id)
    hit = p_hit >= cfg.hit_threshold
    diff["have_enough_info"] = hit
    meta["p_hit"] = p_hit
    if cfg.verify_logit_read:
        # VERIFICATION (ROADMAP 1.1): what would a FREE decode have produced here?
        # If the unrestricted argmax is not a boolean at all, the logit read is
        # imposing structure the model did not intend — we must know that before
        # trusting this path. Logged every tick, costs nothing.
        top = int(torch.argmax(logits).item())
        meta["argmax_tok"] = b.tok.decode([top])
        meta["argmax_is_bool"] = top in (true_id, false_id)
        meta["argmax_agrees"] = (top == (true_id if hit else false_id))

    lit = "true" if hit else "false"

    # seen_mode="after": the level is already read; NOW describe the scene. If
    # this scores like "before", the perception step helps by existing; if it
    # scores like "off", the ORDER is what mattered.
    if cfg.seen_mode == "after":
        logits = step(b.embed_text(lit + ',"seen":"'))
        seen, sids, logits = _decode_until(step, b, logits, cfg, gen,
                                           '"', cfg.schema_max_seen_tokens)
        n_dec += len(sids)
        diff["seen"] = seen.strip()
        lit = '"'                       # we are mid-string; close it, not a bool

    # ---- hot path only: onset time + answer -----------------------------------
    if hit:
        logits = step(b.embed_text(lit + ',"event_time_s":'))
        t_txt, tids, logits = _decode_until(step, b, logits, cfg, gen,
                                            ',"}', cfg.schema_max_int_tokens)
        n_dec += len(tids)
        try:
            diff["event_time_s"] = float(t_txt.strip())
        except (TypeError, ValueError):
            pass
        logits = step(b.embed_text(',"answer":"'))
        ans, aids, logits = _decode_until(step, b, logits, cfg, gen,
                                          '"', cfg.schema_max_answer_tokens)
        n_dec += len(aids)
        diff["answer"] = ans.strip()
        logits = step(b.embed_text('","more":'))
    else:
        logits = step(b.embed_text(lit + ',"more":'))

    # ---- more: the escape hatch ----------------------------------------------
    # One forced literal + one logit read. On the ~99% quiet path that is ONE
    # forward and ZERO decodes, yet the model keeps full power to change cadence,
    # append memory or bump a counter when it actually wants to. Pay only to speak.
    p_more = _read_bool(logits, true_id, false_id)
    meta["p_more"] = p_more
    if p_more >= cfg.more_threshold:
        logits = step(b.embed_text('true,'))
        tail, mids, logits = _decode_until(step, b, logits, cfg, gen,
                                           '}', cfg.schema_max_tail_tokens)
        n_dec += len(mids)
        for k, v in (_extract_json("{" + tail + "}") or {}).items():
            diff[k] = v

    meta["n_decode"] = n_dec
    return diff, meta


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

    # ---- snapshot vs in-place (cfg.controller_cache_mode) ---------------------
    # "inplace" removes the per-tick deep copy of the KV cache. It is only sound
    # when nothing else can mutate the primary while we generate, and when the
    # primary is on OUR device. Both conditions are checked here and a failure
    # DOWNGRADES LOUDLY -- a silent fallback would make a latency/memory result
    # unattributable, and a silent non-fallback would corrupt the cache.
    use_inplace = (cfg.controller_cache_mode == "inplace")
    if use_inplace and not cfg.deterministic:
        log("controller", 0.0, "WARNING: controller_cache_mode=inplace requires "
                               "deterministic=True (lockstep) -- a free-running "
                               "ingester would append inside the borrow. Falling "
                               "back to snapshot.")
        use_inplace = False
    if use_inplace and cross_gpu:
        log("controller", 0.0, "WARNING: controller_cache_mode=inplace cannot cross "
                               "GPUs (the primary lives on the manager's device). "
                               "Falling back to snapshot.")
        use_inplace = False
    log("controller", 0.0, f"cache_mode={'inplace' if use_inplace else 'snapshot'}")

    next_check_vt = cfg.probe_min_s           # skip the empty-cache tick at t=0
    reported = []                              # conversation history: (vt, answer) already emitted
    pending_q = ""                             # question_for_next: what to verify on the next tick
    # DIFF-MERGE: keep a persistent control config; the model emits only the fields
    # that CHANGE each tick (a compact JSON diff), so it rarely decodes the string
    # fields -> much less latency. Transient fields (have_enough_info/answer/seen/
    # event_time_s) RESET to default every tick and must be re-asserted.
    #
    # ┌── PARKED 2026-08-12: `count`, `phase`, `note` ──────────────────────────┐
    # count/phase WERE persistent task accumulators (MISSION.md pillar 7c); note
    # was the model-authored trace (pillar 7b). All three are commented out here
    # and in prompts.py, deliberately and reversibly. See TODO-7BC below for what
    # they were for and what has to be true before they come back.
    #
    # Measured on output_full9 (199,909 ticks, MISSION §10): `note` 0.00% fill —
    # never emitted once; `phase` 0.54% overall / 2.3% on the ONE task that uses
    # it; `count` 16.2%. And on the read side all three have NO consumer: `count`
    # is init + one log line, `phase` is init only, `note` has a consumer whose
    # producer never fires. They are paid for in tail-decode tokens and return
    # nothing. Parking them removes the only fields in the schema that are pure
    # cost — it cannot change any score, because nothing reads them.
    # └─────────────────────────────────────────────────────────────────────────┘
    state = {"fps": cfg.encoder_idle_fps, "next_check_s": cfg.probe_default_s,
             "have_enough_info": False, "new_event": False, "answer": "",
             "question_for_next": "", "seen": "", "event_time_s": None}
    #        "count": 0, "phase": ""}          # PARKED — see TODO-7BC
    # notes = []                               # PARKED — bounded ring, pillar 7b
    seen_trace = []                            # (vt, seen) — the perception trace;
                                               # consecutive duplicates collapsed
    # true/false must be SINGLE tokens for the logit read; verified for Qwen3-VL
    # ('true'->1866, 'false'->3849). Fall back to free decode if a tokenizer splits them.
    ids_bool = (_single_tok(b.tok, "true"), _single_tok(b.tok, "false"))
    use_schema = (cfg.decode_mode == "schema" and None not in ids_bool)
    if cfg.decode_mode == "schema" and not use_schema:
        log("controller", 0.0, "WARNING: true/false are not single tokens -> "
                               "falling back to free decode")
    log("controller", 0.0, f"decode_mode={'schema' if use_schema else 'free'}")
    # LEVEL -> EDGE in CODE (semantic Schmitt gate): the model reports whether the
    # condition is satisfied NOW (a LEVEL it can judge from the cache snapshot);
    # firing happens only on the RISING edge (false -> true across ticks), so a
    # condition that stays on screen cannot re-fire. Within a true stretch, a fire
    # is also allowed when the answer describes a clearly DIFFERENT occurrence
    # (low word overlap with the last fired answer) — e.g. date poster then ticket
    # prices with no gap in between.
    prev_level = False
    armed = True             # hysteresis gate (cfg.gate_strategy): ready to fire
    last_fire_vt = -1e9      # for debounce + timed re-arm
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
        # ---- READ THE SHARED CACHE ------------------------------------------
        # Two paths, identical arithmetic (see cfg.controller_cache_mode):
        #   inplace  -> generate on the PRIMARY at (next_pos, len), erase after.
        #   snapshot -> deep-copy the primary and generate on the copy.
        # Both start the generation at the same pos_start/phys_start over the same
        # key/value prefix, so the logits are the same numbers; only the copy is
        # skipped. The try/finally below is load-bearing: if a tick raises, the
        # borrow MUST still be erased or the primary keeps the controller's own
        # prompt tokens and every later frame is conditioned on them.
        if use_inplace:
            pos, phys = mgr.borrow_begin("ctrl.borrow")
            cache = mgr.cache
        else:
            cache, pos, phys = mgr.snapshot_clone()
            if cross_gpu:
                cache = _cache_to(cache, b.device)
        try:

            def step(embeds):
                nonlocal pos, phys, cache
                logits, cache = b.forward(embeds, cache, pos_start=pos, phys_start=phys)
                if use_inplace:
                    # forward() may hand back a different cache object; the manager
                    # must keep pointing at the one that actually grew, or
                    # borrow_end() would truncate a stale object and leave the
                    # controller's tokens in the primary.
                    mgr.cache = cache
                pos += embeds.shape[1]
                phys += embeds.shape[1]
                return logits

            # Build the controller prompt: task ICL + (optional) deferred check +
            # CONVERSATION HISTORY of what has already been reported, so the model can
            # set new_event=false for repeats and only fire on genuinely new details.
            # With icl_in_sink the ICL was already seeded into the pinned sink by the
            # ingester, so DON'T splice it again here — re-splicing would both duplicate
            # it and re-insert the 1400-token wall between the newest frame and the
            # generation point, which is exactly what we are removing.
            prompt = "" if cfg.icl_in_sink else cfg.controller_prompt.rstrip()
            if pending_q:
                prompt += (f"\nYou previously asked yourself: '{pending_q}'. "
                           f"Judge it now from the MOST RECENT frames.")
            # timestamped history: a LATER onset is a new event, not a repeat of these
            # ---- MEMORY (MISSION pillar 7): the writer's own trace, fed back in ----
            # Two halves, both timestamped and append-only:
            #   WHAT I SAW  -- the `seen` trace. Until now `seen` was reset and thrown
            #                  away every tick, so the controller had no idea what it
            #                  had already looked at. Consecutive duplicates are
            #                  collapsed (the log showed the same scene repeated 3-4
            #                  ticks running), so this reads as a scene-CHANGE history
            #                  and stays cheap.
            #   WHAT I SAID -- `reported`, code-owned. The model must never be able to
            #                  write this: if it could, it could hallucinate having
            #                  already answered and dedup would silently fail open.
            # Both are bounded rings -> per-tick prompt cost is O(1), not O(stream).
            # This is what makes dedup and accumulation possible at all.
            # ⚠️ FROZEN-PERCEPTION BUG (measured 2026-07-30). Feeding the `seen` trace
            # back here froze the perception channel: 20 of 58 videos emitted exactly ONE
            # byte-identical scene description for the WHOLE video (578 ticks, 100%), and
            # 31 of 58 emitted <=2 distinct descriptions ever. Failure-mode breakdown over
            # 87 GT triggers: PERCEPTION 81.6%, JUDGMENT 3.4%. A null model using only the
            # timestamp — no pixels at all — BEAT p_hit on 3 of 4 tasks.
            #
            # Mechanism: the last `seen` line sits a few tokens before the `{"seen":"` slot
            # the model must now fill. Under greedy decode with a 12-token cap, copying the
            # adjacent line is the cheapest continuation — and once copied, "repeats
            # collapsed" makes the trace shorter and the copy even more attractive. A
            # self-reinforcing induction loop. The memory feature caused it.
            #
            # Default OFF while we re-measure. ONE VARIABLE: leave `now_anchor` off too.
            if cfg.seen_trace_in_prompt:
                if seen_trace:
                    trace = "".join(f"  @{svt:.0f}s {s}\n" for svt, s in seen_trace)
                else:
                    trace = "  (nothing yet)\n"
                prompt += ("\n\nWHAT YOU HAVE SEEN so far (your own observations, newest last; "
                           "repeats collapsed):\n" + trace)
            convo = "".join(f"  @{rvt:.0f}s {a}\n" for rvt, a in reported) or "  (nothing yet)\n"
            prompt += ("\nWHAT YOU HAVE ALREADY TOLD THE USER (past occurrences with their "
                       "times; a fresh onset at a later time is a NEW event — it is only a "
                       "repeat while the SAME occurrence is still on screen):\n" + convo)
            # PARKED — see TODO-7BC. This consumer works; its producer never fired
            # (0/199,909 ticks), because no prompt ever taught the `note` field.
            # if notes:
            #     prompt += ("\nNOTES YOU KEPT:\n"
            #                + "".join(f"  @{nvt:.0f}s {n}\n" for nvt, n in notes))
            # Optional PRESENT anchor — the second candidate cause of frozen perception:
            # nothing in the prompt says what time it is NOW or that `seen` must describe
            # the LATEST frame rather than any frame in the cache. Kept OFF by default so
            # it is tested as its own variable, not confounded with seen_trace_in_prompt.
            if cfg.now_anchor:
                prompt += (f"\n\nIt is now {vt:.0f}s. Describe ONLY what is on screen in the "
                           f"MOST RECENT frame, not what you saw earlier.")
            prompt += "\nNow emit ONLY your control JSON for the current stream:\n"

            # PRIME the decoder with an open brace: Qwen3-VL is an instruct model and,
            # spliced as raw text onto the cache (no assistant-turn markers), it would
            # otherwise emit EOS immediately at the splice point. Starting mid-object
            # forces it to complete the JSON. We reconstruct raw = "{" + generated.
            meta = {}
            t_dec0 = time.time()
            if use_schema:
                # SCHEMA WALK: code forces every key/punctuation, model fills only the
                # value slots, booleans come from a logit read. The model can no longer
                # emit fps/next_check_s on the default path -> the diff is a diff.
                diff, meta = _schema_tick(b, cfg, gen, step, prompt, ids_bool)
                ids = [None] * meta.get("n_decode", 0)   # count only, for telemetry
                raw = json.dumps(diff, separators=(",", ":"))
            else:
                logits = step(b.embed_text(prompt + "{"))
                ids = []
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
                raw = "{" + b.decode(ids)
                diff = _extract_json(raw)             # the model's DIFF (partial dict); {} = no change
            decode_s = time.time() - t_dec0
            if prof is not None:
                prof.observe("ctrl_decode_s", decode_s)
                if ids:
                    prof.observe("ctrl_decode_ms_per_tok", 1000 * decode_s / len(ids))
                if "p_hit" in meta:
                    prof.observe("ctrl_p_hit", meta["p_hit"])
                    prof.incr("ctrl_argmax_agrees" if meta.get("argmax_agrees")
                              else "ctrl_argmax_disagrees")
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
                # PARKED — see TODO-7BC. The append-never-rewrite rule below is the
                # part worth keeping when `note` comes back: MEMORY IS A LOG, NOT A
                # DOCUMENT. A memory the model rewrites each tick costs tokens
                # proportional to its length, so tick latency would grow with watch
                # time — backwards for an unbounded stream. The bounded ring keeps
                # per-tick cost O(1).
                # if key == "note":
                #     if str(v).strip():
                #         notes.append((vt, str(v).strip()))
                #         del notes[:-cfg.notes_ring]
                # elif key in state:
                #
                # `count` / `phase` are no longer in `state`, so this line now drops
                # them the way it drops any unknown key. That is the parking
                # mechanism, and it is the SAME silent-discard bug fixed in the
                # comment at the state dict above — deliberate this time, not an
                # accident. Re-adding the keys to `state` re-enables the merge.
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

            # MEMORY: record what we just saw. Collapse consecutive duplicates so the
            # trace is a scene-CHANGE history rather than one line per tick, then keep
            # only the last `seen_trace_ring` entries so the prompt cost stays O(1).
            seen_now = (state["seen"] or "").strip()
            if seen_now and (not seen_trace or seen_trace[-1][1] != seen_now):
                seen_trace.append((vt, seen_now))
                del seen_trace[:-cfg.seen_trace_ring]

            # ---- FIRE DECISION -------------------------------------------------
            # "edge"       : rising edge of the boolean level (original).
            # "hysteresis" : Schmitt gate on the CONTINUOUS p_hit — only possible now
            #   that the logit read gives a real number instead of a bit. This is the
            #   tuned "hyst2b" from the probe-gate arm (EXPERIENCE.md), which lifted
            #   joint_f1 0.149 -> 0.206 there. Aimed at PRECISION: measured 0.112, i.e.
            #   89% of emits were false positives (206 emits for 84 GT).
            #     fire   when armed AND p_hit >= high AND debounce elapsed
            #     re-arm when p_hit < low  OR  rearm_s elapsed since the last fire
            # A single threshold cannot do this: it re-fires on every jitter across the
            # line, which is what the raw level did.
            rising = level and not prev_level
            distinct = (level and prev_level and answer and reported
                        and _word_sim(answer, reported[-1][1]) < cfg.distinct_sim_thr)
            p_hit = meta.get("p_hit")
            if cfg.gate_strategy == "hysteresis" and p_hit is not None:
                if not armed and (p_hit < cfg.gate_low_thr
                                  or (cfg.gate_rearm_s > 0
                                      and (vt - last_fire_vt) >= cfg.gate_rearm_s)):
                    armed = True
                fire = bool(answer) and armed and p_hit >= cfg.gate_high_thr \
                    and (vt - last_fire_vt) > cfg.debounce_s
                if fire:
                    armed = False
            else:
                fire = bool(answer) and (rising or distinct)
            if fire:
                last_fire_vt = vt

            # log EVERY tick's raw diff so all responses are inspectable in the log file
            #
            # NO CAP. This was `raw.strip()[:240]`, which silently cut 8.5% of ticks
            # (16,995 of 199,909 in output_full9 -- 100% of them landing EXACTLY at
            # the cap, which is how we know it was truncation and not the model).
            # The cut was selective: it ate the LONGEST emissions, and the longest
            # emissions are precisely the ticks that used the `more` tail
            # (fps/next_check_s/note/count/phase), so every tail-field measurement
            # taken off these logs was biased low -- 57% of event_narration ticks,
            # 20% of sequential_step_instruction. The SYSTEM was never affected: the
            # full diff is applied to `state` above regardless. Only our ability to
            # see it afterwards was clipped.
            #
            # Newlines are ESCAPED, not dropped. Every consumer of these logs
            # (fields.py, auc.py, compliance.py) reads one tick per line, so a raw
            # newline from the free-decode path would silently split one record into
            # two. Escaping keeps the log both lossless and one-line-per-tick. Only
            # literal CR/LF are touched: in schema mode `raw` is json.dumps output,
            # which already encodes newlines as the two characters \ and n, so this
            # leaves schema-mode lines byte-for-byte parseable by json.loads.
            vid = cfg.video_id or "?"
            _raw = raw.strip() if diff else f"NO-DIFF raw={raw!r}"
            log("ctrl.raw", vt,
                f"[{vid}] " + _raw.replace("\r", "\\r").replace("\n", "\\n"))
            # p_hit is the CONTINUOUS confidence from the logit read; agree= is the
            # verification that a free decode would have produced the same boolean.
            extra = ""
            if meta:
                extra = (f" p_hit={meta.get('p_hit', float('nan')):.3f}"
                         f" p_more={meta.get('p_more', float('nan')):.3f}")
                if cfg.verify_logit_read and "argmax_agrees" in meta:
                    extra += (f" agree={meta['argmax_agrees']}"
                              f" argmax={meta['argmax_tok']!r}")
            # count=/notes= dropped from this line with the fields (TODO-7BC). NOTE
            # for whoever re-enables them: fields.py and auc.py parse ctrl.gate
            # positionally-by-key, so putting them back must keep `key=value` form.
            log("ctrl.gate", vt, f"[{vid}] fps={fps:.1f} level={level} rise={rising} "
                                 f"new_occ={distinct} fire={fire} next={nxt:.1f}s "
                                 f"gen={gen_s:.1f}s ntok={len(ids)} q={pending_q!r}"
                                 f"{extra}")

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
            # RESTORE BEFORE RELEASE. set_next_check() frees the waiting ingester,
            # and the ingester's first act is mgr.ingest() -- so the primary must
            # already be restored at that instant. Publishing first left a window a
            # few statements wide in which the ingester wrote into a STILL-BORROWED
            # cache: measured 3 and 13 RuntimeErrors across two 18-sample arms
            # (2026-08-13). Each one killed the ingester THREAD, so the video stopped
            # being fed and the rest of that sample went silent -- one arm lost two
            # whole tasks and half its emits. The guard turned what would have been a
            # silent KV-cache corruption into a loud crash; this ordering is what
            # makes the "no concurrent writer in lockstep" premise actually true.
            if use_inplace:
                mgr.borrow_end()
            clock.set_next_check(next_check_vt)
        finally:
            # erase the controller's splice from the primary. Idempotent (so the
            # release above is not repeated work) and a no-op in snapshot mode. This
            # stays for the EXCEPTION path, where the tick never reached the release.
            if use_inplace:
                mgr.borrow_end()

    log("controller", 0.0, "controller stopped")
