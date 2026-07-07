# from dataclasses import dataclass, field


# @dataclass
# class AsyncOmniConfig:
#     # ---- model ----
#     model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
#     device: str = "cuda"
#     writer_device: str = ""
#     encoder_device: str = "" #for independent vision encoder
#     dtype: str = "bfloat16"            # float16 / bfloat16 / float32
#     # Cap vision tokens per frame by limiting the image's total pixel area.
#     # tokens/frame = image_area_px / px_per_token, where one vision token covers
#     # a (patch_size * merge_size)^2 region. For Qwen3-VL that's (16*2)^2 = 32x32
#     # = 1024 px/token, so:
#     #   200704 px / 1024 px/token = ~196 tokens/frame (upper bound)
#     # Actual is ~180 (aspect-ratio resize + rounding down to whole merged
#     # patches). 0 = leave the processor default (~2040 tokens/frame at full res,
#     # near-memoryless). Lower = cheaper/blurrier; higher = more detail/cost.
#     max_pixels: int = 200704
#     profile: bool = True              # collect + print the profiling summary

#     # ---- visual token pruning (VisionZip, training-free) ----
#     # When on, each frame's projected tokens are reduced BEFORE they enter the KV
#     # cache: keep the top `prune_dominant_frac` tokens by attention-received
#     # (dominant), then merge the rest into `prune_contextual_frac` representative
#     # tokens by key-similarity (contextual). Total kept ~= dominant+contextual.
#     # Text-agnostic, runs in the vision encoder (off the LLM critical path).
#     prune_img_tokens: bool = False
#     prune_dominant_frac: float = 0.65     # ~0.65 + 0.05 = 70% retention (safe)
#     prune_contextual_frac: float = 0.05

#     # ---- video / pacing ----
#     video_path: str = ""
#     fps: float = 1.0                  # encoder base frame rate (frames/s of video)
#     max_seconds: float = 300.0
#     realtime: bool = True             # pace the encoder to a wall clock
#     speed: float = 1.0                # real-time multiplier (1.0 = live)
#     frame_q_size: int = 16             # vis_q depth (encoder -> orchestrator buffer)

#     # ---- encoder control (proactive INPUT gate, orchestrator-steered) ----
#     encoder_focus_fps: float = 3.0    # fps when the orchestrator says "focus"
#     encoder_idle_fps: float = 1     # fps when the action is boring

#     # ---- memory management (shared linear KV cache) ----
#     kv_budget: int = 183000           # 256K-token context (eviction only past this)
#     # `sink` (the protected system-prompt prefix) is measured at seed time.

#     # ---- proactivity gating ----
#     goal_threshold: float = 0.5       # writer fires when goal yes_share >= this
#     debounce_s: float = 2.0           # min video-seconds between writer triggers
#     goal_gate_every: int = 3          # run the "goal?" probe every N frames (1 fwd
#                                       # pass each). 2 halves the dominant GPU op.
#     vision_gate_every: int = 8        # run the "important?" probe every N frames
#     log_gate_every: int = -1           # log the goal probe every N probes


#     # ---- writer sampling (official Qwen3-VL-8B-Instruct *text* preset) ----
#     writer_greedy: bool = False
#     writer_seed: int = 3407            # RNG seed for reproducible sampling
#     writer_temperature: float = 0.7
#     writer_top_p: float = 0.8
#     writer_top_k: int = 20
#     writer_repetition_penalty: float = 1.0
#     writer_presence_penalty: float = 1.5
#     writer_max_tokens: int = 100
#     writer_repeat_window: int = 8      # local anti-loop guard (keep; cheap safety, if too many repeating tokens in this window stop the writer)

#     # ---- orchestrator / VL probe sampling preset ----
#     # NOTE: the orchestrator does NOT sample tokens -- its gate is a deterministic
#     # yes/no logit ratio (proactivity.yes_share), so seed/top_p/top_k/temperature
#     # are kept as the documented preset for reference / any future free-text
#     # thinking by the orchestrator. Randomizing the gate would only make triggering
#     # flaky.
#     orch_temperature: float = 0.7# 1.0
#     orch_top_p: float = 0.8#0.95
#     orch_presence_penalty: float = 1.5#0.0
#     # orch_temperature: float = 1.0
#     # orch_top_p: float = 0.95
#     # orch_presence_penalty: float = 0.0
#     orch_greedy: bool = False
#     orch_seed: int = 1234
#     orch_top_k: int = 20
#     orch_repetition_penalty: float = 1.0
#     orch_out_seq_length: int = 40960

#     # ---- prompts (biggest behaviour levers) ----
#     system_prompt: str = (
#         "You are a live football commentator watching a match frame by frame. "
#         "Watch the action carefully. You ingest the streaming vision tokens(features) and correctly answer the probe questions if you are asked while you are watching the live. You only say what you see. Be sure about saying yes and no when the question asking for whether a goal has been scored recently or not. you get punishment if you miss a goal or give false alarm. Enjoy.")
#     goal_question: str = (
#         "\nQuestion: Judging ONLY from the most recent frames, has a goal just "
#         "been scored right now (ball in the net, players celebrating)? "
#         "Answer yes or no: ")
#     vision_question: str = (
#         "\nQuestion: Is the action important enough that I should keep looking "
#         "closely at the next frame? Answer yes or no: ")
#     # The writer generates off a snapshot of the orchestrator's cache, so it
#     # already "sees" everything the analyst (orchestrator) has been watching and
#     # thinking about. This instruction is spliced on right before it speaks.
#     writer_prompt: str = (
#         "\nYou are the live commentary VOICE who only talks when there is a goal. The analyst above has been watching "
#         "the match frame by frame and has just suspected a GOAL on screen right "
#         "now. A goal might have been scored -- look in the most recent images. you can also see the scores in the top of the screen. In ONE "
#         "short, excited line shout: WHICH SIDE scored, the PLAYER if you can "
#         "identify them, and the running TOTAL SCORE. Output only that single line "
#         "and nothing else. Do not say anything if you are not sure a goal has been scored. Do not say what time is it or things like that. Just talk about goal.")
#     writer_cue: str = "\nLIVE: "

#     # ---- yes/no surface forms scored by the proactivity probe ----
#     yes_words: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes"])
#     no_words: list = field(default_factory=lambda: ["no", "No", " no", " No"])


#     # [NOT IMPORTANT]---- ground-truth eval (only activates for the France-Senegal highlight) ----
#     groundtruth: bool = True          # score triggers/writer vs known goal times
#     gt_window: float = 3.0           # +/- video seconds to count a goal as detected




from dataclasses import dataclass, field


@dataclass
class AsyncOmniConfig:
    # ---- model ----
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    writer_device: str = ""
    encoder_device: str = "" #for independent vision encoder
    dtype: str = "bfloat16"            # float16 / bfloat16 / float32
    # Cap vision tokens per frame by limiting the image's total pixel area.
    # tokens/frame = image_area_px / px_per_token, where one vision token covers
    # a (patch_size * merge_size)^2 region. For Qwen3-VL that's (16*2)^2 = 32x32
    # = 1024 px/token, so:
    #   200704 px / 1024 px/token = ~196 tokens/frame (upper bound)
    # Actual is ~180 (aspect-ratio resize + rounding down to whole merged
    # patches). 0 = leave the processor default (~2040 tokens/frame at full res,
    # near-memoryless). Lower = cheaper/blurrier; higher = more detail/cost.
    max_pixels: int = 200704
    profile: bool = True              # collect + print the profiling summary

    # ---- reproducibility ----
    seed: int = 0                     # seeds python/numpy/torch (+cuda) RNGs
    deterministic: bool = False       # also force deterministic CUDA kernels AND
                                      # make the encoder block instead of dropping
                                      # frames -> bit-reproducible eval (use batch).

    # ---- absolute time signal ----
    # Prepend a short text timestamp before each frame's tokens so the model can
    # reason about real match time (in-distribution for Qwen3-VL, which was
    # trained with timestamp tokens). {t} = video seconds of the frame.
    timestamp_tokens: bool = True
    timestamp_fmt: str = "\ntime {t:.1f}s\n"

    # ---- visual token pruning (VisionZip, training-free) ----
    # When on, each frame's projected tokens are reduced BEFORE they enter the KV
    # cache: keep the top `prune_dominant_frac` tokens by attention-received
    # (dominant), then merge the rest into `prune_contextual_frac` representative
    # tokens by key-similarity (contextual). Total kept ~= dominant+contextual.
    # Text-agnostic, runs in the vision encoder (off the LLM critical path).
    prune_img_tokens: bool = False
    prune_dominant_frac: float = 0.65     # ~0.65 + 0.05 = 70% retention (safe)
    prune_contextual_frac: float = 0.05

    # ---- video / pacing ----
    video_path: str = ""
    fps: float = 1.0                  # encoder base frame rate (frames/s of video)
    max_seconds: float = 300.0
    realtime: bool = True             # pace the encoder to a wall clock
    speed: float = 1.0                # real-time multiplier (1.0 = live)
    frame_q_size: int = 8             # vis_q depth (encoder -> orchestrator buffer)

    # ---- encoder control (proactive INPUT gate, orchestrator-steered) ----
    encoder_focus_fps: float = 3.0    # fps when the orchestrator says "focus"
    encoder_idle_fps: float = 0.5     # fps when the action is boring

    # ---- memory management (shared linear KV cache) ----
    kv_budget: int = 262144           # 256K-token context (eviction only past this)
    # `sink` (the protected system-prompt prefix) is measured at seed time.

    # ---- proactivity gating ----
    goal_threshold: float = 0.5       # writer fires when goal yes_share >= this
    debounce_s: float = 2.0           # min video-seconds between writer triggers
    # ---- edge-triggered gate (hysteresis / Schmitt) — precision fix ----
    # Baseline gate fires on LEVEL: every probe with share>=goal_threshold re-fires
    # while a condition stays true -> over-triggering. With gate_hysteresis, fire
    # only on a RISING crossing: emit once when share>=gate_high_thr (while armed),
    # then disarm; re-arm only after share drops below gate_low_thr. => one emit per
    # onset. Set gate_high_thr==goal_threshold to isolate the edge effect alone.
    gate_hysteresis: bool = True      # DEFAULT gate = tuned hyst2 (best f1 so far)
    gate_high_thr: float = 0.5        # rising threshold to FIRE (when armed)
    gate_low_thr: float = 0.40        # re-arm after share falls below this (narrow
                                      # band -> re-arms on small dips -> more onsets)
    gate_rearm_s: float = 5.0         # ALSO re-arm this many seconds after a fire even
                                      # if the signal stayed high (recurring events);
                                      # 0 = signal-only re-arm.
    goal_gate_every: int = 3          # run the "goal?" probe every N frames (1 fwd
                                      # pass each). 2 halves the dominant GPU op.
    vision_gate_every: int = 8        # run the "important?" probe every N frames
    log_gate_every: int = -1           # log the goal probe every N probes

    # ---- proactivity scheduler ----
    # "fixed": input/output gates probe on a fixed cadence (ALL current behavior).
    # "model": PURE-GENERATIVE agentic loop. The controller (writer) reads the shared
    #   KV cache and emits ONE control JSON per cycle carrying fps (input gate),
    #   have_enough_info (output gate), new_event (dedup vs already-answered), answer,
    #   and next_check_s (self-paced cadence). On a NEW triggered event it emits the
    #   user answer, then resumes reading + emitting config. Over-firing is minimized
    #   by the new_event dedup + model-chosen cadence rather than fixed probing.
    probe_scheduler: str = "fixed"
    probe_default_s: float = 3.0      # fallback cadence if next_check_s is missing/unparsed
    probe_min_s: float = 1.0          # clamp next_check_s floor (anti-spin)
    probe_max_s: float = 10.0         # clamp next_check_s ceiling (anti-stall)
    controller_max_tokens: int = 160  # cap for the control-JSON generation
    # In-context "control language" (VISPROG-style): a compact JSON DSL the frozen
    # model emits each tick to drive its own probing. Taught via worked
    # (Situation -> Control) pairs that demonstrate the DECISIONS (stay quiet /
    # defer-with-question / fire-once / suppress-repeat), not just the syntax.
    controller_prompt: str = (
        "\nYou are the CONTROLLER of a live video monitor for the target event in your "
        "task. Each turn you read the stream so far (above) and emit exactly ONE control "
        "command as a compact JSON object with these fields:\n"
        "  fps              : how densely to sample next (1-3; raise when the scene is busy)\n"
        "  have_enough_info : true ONLY when you are confident the target event has occurred\n"
        "  new_event        : true ONLY if this occurrence is NEW (not already reported)\n" #TODO: SHOULD keep memory of what's reported
        '  answer           : if have_enough_info AND new_event -> ONE sentence describing the REAL event; else ""\n' #TODO: may be say what the target task asks
        '  question         : if unsure but something may be developing -> a short yes/no check to verify next time; this is optional you may keep it empty if there is no important question at this time; else ""\n'
        "  next_check_s     : seconds until the next check (small when unsure or busy, larger when idle)\n"
        "Rules: describe only what you actually see; NEVER copy the example text; NEVER "
        "re-report an event you already reported; when unsure, DEFER with a question and a "
        "short next_check_s, then confirm on the next turn before answering.\n" # TODO: how many seconds is good?
        "Worked examples (Situation -> Control):\n"
        "# nothing relevant on screen yet -> stay quiet, sample slowly\n"
        "# something may be starting but you are not sure -> sample densely and DEFER a targeted check\n"
        '{"fps":3,"have_enough_info":false,"new_event":false,"answer":"","question":"has the target event just started?","next_check_s":2}\n'
        "# the deferred check is now clearly confirmed -> report it ONCE\n"
        '{"fps":2,"have_enough_info":true,"new_event":true,"answer":"<one sentence describing the real event>","question":"","next_check_s":5}\n'
        "# the event is still on screen but you ALREADY reported it -> stay silent (dedup)\n"
        '{"fps":1,"have_enough_info":true,"new_event":false,"answer":"","question":"","next_check_s":4}\n'
        "Now emit ONLY your control JSON for the current stream:\n")

    # ---- benchmark matrix (ablation switches; toggle these per experiment) ----
    # input_gate: run the INPUT proactivity probe ("is this important?") to steer
    #   the encoder fps (focus/idle). False => NO input probe: the encoder feeds
    #   every frame at the fixed base fps (cfg.fps). [wired]
    input_gate: bool = True
    # output_gate: run the OUTPUT proactivity probe ("did a goal happen?") that
    #   triggers the writer. [wired]
    output_gate: bool = True
    # writer_cache: writer reuses the ingester's KV cache via an MVCC snapshot
    #   (True, current) vs. generates without it (False). [scaffold: False TODO]
    writer_cache: bool = True
    # proactivity_in_prompt: bake the proactivity question into the system prompt
    #   instead of splicing a per-frame probe. [scaffold: TODO]
    proactivity_in_prompt: bool = False


    # ---- writer sampling (official Qwen3-VL-8B-Instruct *text* preset) ----
    writer_greedy: bool = False
    writer_seed: int = 3407            # RNG seed for reproducible sampling
    writer_temperature: float = 0.7
    writer_top_p: float = 0.8
    writer_top_k: int = 20
    writer_repetition_penalty: float = 1.0
    writer_presence_penalty: float = 1.5
    writer_max_tokens: int = 100
    writer_repeat_window: int = 8      # local anti-loop guard (keep; cheap safety, if too many repeating tokens in this window stop the writer)

    # ---- orchestrator / VL probe sampling preset ----
    # NOTE: the orchestrator does NOT sample tokens -- its gate is a deterministic
    # yes/no logit ratio (proactivity.yes_share), so seed/top_p/top_k/temperature
    # are kept as the documented preset for reference / any future free-text
    # thinking by the orchestrator. Randomizing the gate would only make triggering
    # flaky.
    orch_greedy: bool = False
    orch_seed: int = 1234
    orch_temperature: float = 1.0
    orch_top_p: float = 0.95
    orch_top_k: int = 20
    orch_repetition_penalty: float = 1.0
    orch_presence_penalty: float = 0.0
    orch_out_seq_length: int = 40960

    # ---- prompts (biggest behaviour levers) ----
    system_prompt: str = (
        "You are a live football commentator watching a match frame by frame. "
        "Watch the action carefully. The instant a goal is scored, you shout it "
        "with excitement; otherwise you stay quiet.")
    goal_question: str = (
        "\nQuestion: Judging ONLY from the most recent frames, has a goal just "
        "been scored right now (ball in the net, players celebrating)? "
        "Answer yes or no: ")
    vision_question: str = (
        "\nQuestion: Is the action important enough that I should keep looking "
        "closely at the next frame? Answer yes or no: ")
    # The writer generates off a snapshot of the orchestrator's cache, so it
    # already "sees" everything the analyst (orchestrator) has been watching and
    # thinking about. This instruction is spliced on right before it speaks.
    writer_prompt: str = (
        "\nYou are the live commentary VOICE. The analyst above has been watching "
        "the match frame by frame and has just confirmed a GOAL on screen right "
        "now. A goal HAS been scored -- do NOT second-guess or deny it. In ONE "
        "short, excited line shout: WHICH SIDE scored, the PLAYER if you can "
        "identify them, and the running TOTAL SCORE. Output only that single line "
        "and nothing else.")
    writer_cue: str = "\nLIVE: "

    # ---- yes/no surface forms scored by the proactivity probe ----
    yes_words: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes"])
    no_words: list = field(default_factory=lambda: ["no", "No", " no", " No"])


    # [NOT IMPORTANT]---- ground-truth eval (only activates for the France-Senegal highlight) ----
    groundtruth: bool = True          # score triggers/writer vs known goal times
    gt_window: float = 3.0           # +/- video seconds to count a goal as detected
