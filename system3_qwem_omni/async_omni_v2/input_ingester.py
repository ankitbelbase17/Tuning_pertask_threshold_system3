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

DETERMINISTIC LOCKSTEP (cfg.deterministic): the clock is published only AFTER a
frame is fully in the cache, and the ingester then WAITS until the controller has
completed every tick due at <= vt (clock.get_next_check() > vt) before feeding
the next frame. Combined with the encoder's blocking put + fixed fps + greedy
decode, the whole walk is frame-indexed: every tick sees exactly frames [0..vt],
identically on every run — the async snapshot race is gone.

------------------------------------------------------------------------------
AUDIO, "OPTION A" (FORK: system3_qwem_omni). There is deliberately NO separate
free-running audio-encoder thread. A second independent producer with its own
latency profile (audio ~160ms/chunk vs. vision's per-frame variable cost) can
append out of true-time order relative to video with no correctness guardrail
short of a nontrivial 2-way timestamp merge -- see the design discussion this
fork implements. Instead: THIS thread -- already the single writer of the
primary cache -- is also the ONLY caller of backend.embed_audio(), invoked
SYNCHRONOUSLY every cfg.audio_seconds_per_chunk of video time, right in the
same control flow as frame ingestion. Correctness is free (one thread, one
sequence of steps, nothing to race); the cost is that ~160ms lands on the
per-tick critical path rather than being hidden in a parallel thread. That
trade is deliberate, not an oversight -- see OMNI_FEASIBILITY.md sec 5 for the
latency measurement it's based on, and cfg.use_audio to disable it.
------------------------------------------------------------------------------
"""
import queue
import time

from util import log


def input_ingester_thread(cfg, mgr, vis_q, ctrl, stop, prof=None, clock=None,
                          feed_done=None, writer_q=None, evaluator=None):
    use_audio = cfg.use_audio and hasattr(mgr.b, "embed_audio")
    if cfg.use_audio and not hasattr(mgr.b, "embed_audio"):
        # LOUD, not silent (project standing rule): if audio was asked for,
        # a backend that cannot provide it must not quietly run vision-only.
        raise RuntimeError(
            f"cfg.use_audio=True but backend {type(mgr.b).__name__} has no "
            f"embed_audio() -- set cfg.use_audio=False for an explicit "
            f"vision-only A/B, or use cfg.backend='qwen2_5_omni'.")
    audio_reader = None
    next_chunk_end_vt = cfg.audio_seconds_per_chunk
    last_chunk_end_vt = 0.0
    if use_audio:
        from audio_io import AudioChunkReader
        audio_reader = AudioChunkReader(cfg.video_path, sample_rate=cfg.audio_sampling_rate)
        mgr.chunk_anchor_vt = 0.0
        log("ingester", 0.0, f"audio ON: chunk={cfg.audio_seconds_per_chunk}s "
                             f"rate={cfg.audio_position_id_per_seconds}/s "
                             f"has_audio_track={audio_reader.has_audio}")

    system_prompt = cfg.system_prompt.replace("{instruction}", cfg.instruction)
    # ICL IN THE SINK (cfg.icl_in_sink) — PRIVILEGE THE PRESENT.
    # The task ICL is CONSTANT, but it was being spliced fresh after the video
    # tokens on every tick, which put ~1400 tokens of instruction prose BETWEEN
    # the newest frame and the point of generation:
    #     [~100k vision tokens] .. [newest frame] -> [1400 tok of ICL] -> {"seen":"
    # so the last thing the model saw before answering was the manual, not the
    # video. Seeding it into the pinned eviction sink instead means the newest
    # frame is adjacent to the decision, and the ICL is prefilled ONCE per run
    # rather than re-prefilled every tick.
    if cfg.icl_in_sink and cfg.controller_prompt:
        system_prompt = system_prompt.rstrip() + "\n" + cfg.controller_prompt.rstrip() + "\n"
    sink = mgr.seed(system_prompt)
    log("ingester", 0.0, f"seeded cache, sink={sink} tokens, budget={cfg.kv_budget} "
                         f"(deterministic={cfg.deterministic}, gate_mode={cfg.gate_mode})")

    # PROBE-GATE (gate_mode="probe"): yes/no logit probe + hysteresis, fires writer_q.
    probe_mode = cfg.gate_mode == "probe" and writer_q is not None
    gate_q = cfg.goal_question.replace("{event}", cfg.event or cfg.instruction)
    n_frames = 0
    armed = True
    last_trigger_vt = -1e9
    last_vt = 0.0                                # FORK: for the final partial-chunk flush below

    try:
        while not stop.is_set() or not vis_q.empty():
            try:
                vt, embeds, grid_hw = vis_q.get(timeout=0.5)
            except queue.Empty:
                continue
            n_frames += 1
            last_vt = vt
            if prof is not None:
                prof.observe("visq_depth", vis_q.qsize())
                prof.incr("frames_ingested")     # frames the ingester actually wrote to cache

            # ingest streaming visual tokens (optionally prefixed by a text timestamp
            # so the model has a real-time signal, not just token order)
            if cfg.timestamp_tokens:
                mgr.ingest(mgr.b.embed_text(cfg.timestamp_fmt.format(t=vt)))
            if use_audio and cfg.tmrope_positions:
                mgr.ingest(embeds, token_kind="vision", real_seconds=vt, grid_hw=grid_hw)
            else:
                mgr.ingest(embeds)

            dropped = mgr.evict()                # bounded memory
            if dropped:
                log("ingester.evict", vt, f"evicted {dropped} KV tokens (budget={cfg.kv_budget})")

            # ---- AUDIO: synchronous, chunk-boundary-triggered (Option A) -----
            # Fires once every cfg.audio_seconds_per_chunk of VIDEO time, driven
            # by the vision clock crossing the boundary -- never by a separate
            # audio-side clock, so it is always ingested causally after the
            # video content it co-occurs with, never ahead of it.
            if use_audio and vt >= next_chunk_end_vt:
                t0, t1 = last_chunk_end_vt, next_chunk_end_vt
                waveform = audio_reader.read(t0, t1)
                a_embeds = mgr.b.embed_audio(waveform)
                if prof is not None:
                    prof.observe("audio_tokens_per_chunk", a_embeds.shape[1])
                mgr.ingest(a_embeds, token_kind="audio", real_seconds=t0)
                log("ingester.audio", vt, f"chunk [{t0:.1f},{t1:.1f})s -> "
                                          f"{a_embeds.shape[1]} audio tokens, "
                                          f"chunk_anchor_pos now {mgr.chunk_anchor_pos}")
                last_chunk_end_vt = next_chunk_end_vt
                next_chunk_end_vt += cfg.audio_seconds_per_chunk
                mgr.chunk_anchor_vt = last_chunk_end_vt   # open the NEXT chunk's window

            # ---- PROBE-GATE: one forward pass, Schmitt/hysteresis edge, fire writer
            if probe_mode and n_frames % cfg.goal_gate_every == 0:
                share = mgr.probe(gate_q, "probe.goal")
                if cfg.gate_hysteresis:
                    if not armed and (share < cfg.gate_low_thr or
                                      (cfg.gate_rearm_s > 0
                                       and (vt - last_trigger_vt) >= cfg.gate_rearm_s)):
                        armed = True
                    fire = (armed and share >= cfg.gate_high_thr
                            and (vt - last_trigger_vt) > cfg.debounce_s)
                    if fire:
                        armed = False
                else:
                    fire = (share >= cfg.goal_threshold
                            and (vt - last_trigger_vt) > cfg.debounce_s)
                log("ingester.gate", vt, f"[{cfg.video_id or '?'}] share={share:.2f} "
                                         f"armed={armed} fire={fire}")
                if fire:
                    last_trigger_vt = vt
                    if evaluator is not None:
                        evaluator.record_trigger(vt, share)
                    if cfg.deterministic:
                        writer_q.put(vt)         # block, then WAIT for the write to
                        writer_q.join()          # finish -> snapshot is frame-indexed
                    else:
                        try:
                            writer_q.put_nowait(vt)
                        except queue.Full:
                            pass

            # publish vt only now: a controller tick at vt is guaranteed to see
            # every frame up to and including vt in the cache.
            if clock is not None:
                clock.set(vt)

            # LOCKSTEP: hold the next frame until every tick due at <= vt has
            # COMPLETED, so which frames each tick sees is frame-indexed, not a
            # thread race. (Guard against a dead controller with a long timeout.)
            if cfg.deterministic and clock is not None:
                t_wait = time.time()
                while clock.get_next_check() <= vt:
                    time.sleep(0.002)
                    if time.time() - t_wait > 600:
                        log("ingester", vt, "LOCKSTEP TIMEOUT waiting on controller — continuing")
                        break
    finally:
        # FORK: flush whatever's left of the final, necessarily-partial audio
        # chunk (last_chunk_end_vt .. last_vt) -- otherwise the last <2s of
        # audio is silently dropped every single run, not just at EOF edge
        # cases. Silent != acceptable per this project's own standing rule.
        if use_audio and last_vt > last_chunk_end_vt:
            try:
                waveform = audio_reader.read(last_chunk_end_vt, last_vt)
                a_embeds = mgr.b.embed_audio(waveform)
                mgr.ingest(a_embeds, token_kind="audio", real_seconds=last_chunk_end_vt)
                log("ingester.audio", last_vt,
                    f"FINAL partial chunk [{last_chunk_end_vt:.1f},{last_vt:.1f})s -> "
                    f"{a_embeds.shape[1]} audio tokens")
            except Exception as e:
                log("ingester.audio", last_vt, f"final chunk flush FAILED: {e!r}")
        if feed_done is not None:
            feed_done.set()                      # tell the controller the feed is drained
    stop.set()
    log("ingester", 0.0, "input ingester stopped")
