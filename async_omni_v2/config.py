"""
config.py — the single config for the icl_ingester_writer pipeline.

ONE proactivity mode only: the pure-generative CONTROLLER (probe_scheduler was
"model"). The encoder streams frames -> the ingester prefills them into a shared
KV cache -> the controller reads that cache each tick and emits ONE control JSON
(fps / have_enough_info / new_event / answer / question / next_check_s), which is
both the output gate and the writer. There are no fixed yes/no gates here.

This is also the SINGLE place to edit all prompt text: `system_prompt` (the
role/task template, {instruction} filled per-run) and `controller_prompt` (the
control-JSON DSL taught to the model). The eval harness carries no prompt text.

(The fixed-gate ablations, multi-GPU replicas, and VisionZip pruning live on the
`main` branch.)
"""
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Task-specific controller ICL prompts.
# Each replaces the generic `controller_prompt` for one OmniPro task category
# (the adapter selects by sample.task). They teach the SAME control-JSON DSL but
# with a worked, task-concrete (Situation -> Control) example. Schema per tick:
#   fps, have_enough_info, new_event, answer, next_check_s, question_for_next
# The controller appends an "Already reported" conversation history + the final
# emit cue at runtime, so these strings END with the worked example.
# ---------------------------------------------------------------------------
# NOTE ON AUDIO: the OmniPro paper defines semantic_condition_alert as "audio-first",
# but our pipeline does NOT use audio at all right now — we monitor VISUAL + semantic
# cues only. This ICL therefore deliberately avoids any audio/speech references.
#
# Paper's task intent (baked into this prompt): SCA tests COMPREHENSION/JUDGMENT, not
# perception. The model must (1) understand the user's abstract intent, (2) reason about
# whether what's on screen actually MEETS it, (3) alert at each occurrence. Each response
# should state WHAT happened AND WHY it satisfies the condition, in under 25 words. In the
# eval, a temporal match only counts if the response is also content-correct (LLM-judged).
_SEMANTIC_CONDITION_ALERT = (
    "\nYou are the CONTROLLER of a live video monitor. Your monitoring task (the CONDITION "
    "to alert on) is in the system instruction above. This condition is SEMANTIC: it needs "
    "COMPREHENSION and REASONING, not mere object/keyword spotting. You must understand the "
    "user's intent and JUDGE whether what is happening on screen actually MEETS it, then alert "
    "at EACH occurrence where it is satisfied.\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS start with "
    "seen, then have_enough_info — looking BEFORE judging, every tick. Whenever have_enough_info "
    "is true, ALSO include event_time_s and answer. Include fps, next_check_s, or "
    "question_for_next ONLY when they change from their current value (otherwise omit them to stay "
    "short). Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now that is relevant to the condition\n'
    "  have_enough_info  : true when the condition is satisfied on screen NOW; keep reporting true "
    "for as long as it remains satisfied; back to false when it no longer is\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds when THIS "
    "occurrence appeared (read it off the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED whenever have_enough_info is true -> ONE sentence (UNDER 25 '
    'words) stating WHAT is happening AND WHY it satisfies the condition; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the scene is busy)\n"
    "  next_check_s      : seconds until the next check (1-3; keep it small — more may come)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does. It alerts the user only when "
    "have_enough_info goes false -> true (or when your answer describes a clearly DIFFERENT "
    "occurrence), so you will NEVER double-alert by keeping it true while the same thing stays on "
    "screen. Your only job each tick: judge the level honestly and describe what you see. The "
    "condition typically recurs (about 3 times per video on average), so after it stops, keep "
    "watching for the NEXT occurrence.\n"
    "The 'Already reported' list below shows PAST occurrences WITH THEIR TIMES.\n"
    "Rules: base every judgment on what is actually visible; reason about the user's intent; "
    "NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — condition: 'alert whenever the video provides specific "
    "logistical details for the match, such as the date, location, or ticket pricing'). The video "
    "is a football club TV commercial: fans in body paint, the crowd roaring, then a poster with "
    "the match date and location, then ticket pricing. Every output starts with seen + "
    "have_enough_info; other fields appear only when they change:\n"
    "At 3s, none of the asked-for details shown yet -> condition NOT satisfied:\n"
    '{"seen":"fans buying green body paint","have_enough_info":false,"fps":1.0,"question_for_next":"Is the date, location or ticket price shown now?"}\n'
    "At 6s still nothing (question unchanged -> omit it):\n"
    '{"seen":"crowd roaring in the stadium","have_enough_info":false}\n'
    "At 16s the poster shows the date and location -> satisfied; describe it and read its time:\n"
    '{"seen":"poster with match date and venue","have_enough_info":true,"event_time_s":16,"answer":"The match date is August 14th and the location is Dairy Farmers Stadium.","question_for_next":"Is the ticket price shown now?"}\n'
    "At 17s the same poster is still on screen -> STILL satisfied; keep reporting it (the system "
    "will not double-alert):\n"
    '{"seen":"same date and venue poster","have_enough_info":true,"event_time_s":16,"answer":"The match date is August 14th and the location is Dairy Farmers Stadium."}\n'
    "At 20s the poster is gone, players celebrating -> no longer satisfied:\n"
    '{"seen":"players celebrating on the pitch","have_enough_info":false}\n'
    "At 23s ticket pricing appears -> satisfied AGAIN by a new detail:\n"
    '{"seen":"ticket prices and purchase info on screen","have_enough_info":true,"event_time_s":23,"answer":"The video now details ticket costs for different groups and gives a purchase website and phone number.","fps":3.0,"next_check_s":3}\n'
    "(all asked-for details reported -> sample sparsely, keep watching in case more appear)\n"
)


@dataclass
class AsyncOmniConfig:
    # ---- model ----
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"               # primary GPU: ingester + shared KV cache
    # Multi-GPU replicas measured NO decode speedup (the per-tick cost is inherent
    # decode, not GPU contention) -> default single-GPU. Set to "cuda:1"/"cuda:2"
    # (with >=3 visible GPUs) to re-enable the split.
    writer_device: str = ""            # controller replica GPU ("" = share primary)
    encoder_device: str = ""           # vision-encoder replica GPU ("" = share primary)
    dtype: str = "bfloat16"            # float16 / bfloat16 / float32
    # Cap vision tokens/frame by limiting image pixel area. One vision token covers
    # a (patch_size*merge_size)^2 = 32x32 = 1024 px region for Qwen3-VL, so
    # 200704 px / 1024 = ~196 tokens/frame (actual ~180 after aspect resize).
    # 0 = processor default (~2040 tokens/frame, near-memoryless).
    max_pixels: int = 200704
    profile: bool = True              # collect + print the profiling summary

    # ---- reproducibility ----
    seed: int = 0                     # seeds python/numpy/torch (+cuda) RNGs
    deterministic: bool = True        # DETERMINISTIC LOCKSTEP WALK (default for eval):
                                      # fixed-fps frames, blocking queues (no drops),
                                      # clock published only after a frame is in the
                                      # cache, and the ingester WAITS for every due
                                      # controller tick to complete before the next
                                      # frame + deterministic CUDA kernels + greedy
                                      # decode => bit-reproducible runs (kills the
                                      # async snapshot race that made F1 noise).
                                      # Set False for the realtime/async demo.

    # ---- absolute time signal ----
    # Prepend a short text timestamp before each frame's tokens so the model can
    # reason about real time (in-distribution for Qwen3-VL). {t} = video seconds.
    timestamp_tokens: bool = True
    timestamp_fmt: str = "\ntime {t:.1f}s\n"

    # ---- video / pacing ----
    video_path: str = ""
    fps: float = 1.0                  # encoder base frame rate (frames/s of video)
    max_seconds: float = 300.0
    realtime: bool = True             # pace the encoder to a wall clock (the
                                      # controller self-paces in VIDEO time, so it
                                      # needs realtime; batch fast-forwards the clip)
    speed: float = 1.0                # real-time multiplier (1.0 = live)
    frame_q_size: int = 8             # vis_q depth (encoder -> ingester buffer)

    # ---- encoder fps bounds (the controller steers fps within these) ----
    encoder_focus_fps: float = 3.0    # fps ceiling when the controller says "focus"
    encoder_idle_fps: float = 1.0     # fps floor (1-3 fps range; was 0.5)

    # ---- memory (shared linear KV cache) ----
    kv_budget: int = 262144           # 256K-token context (StreamingLLM eviction
                                      # past this; the system-prompt sink is pinned).

    # ---- the CONTROLLER (icl_ingester_writer) ----
    # Self-paced cadence: the controller picks next_check_s each tick; clamp it so
    # it can't spin (min) or stall (max); fall back to default if unparsed.
    probe_default_s: float = 1.0
    probe_min_s: float = 0.2      # finer check grid (was 1.0) so vt lands closer to onsets
    probe_max_s: float = 1.5      # finer check grid (was 3.0)
    controller_max_tokens: int = 300  # cap for the control-JSON generation
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
        "  new_event        : true ONLY if this occurrence is NEW (not already reported)\n"
        '  answer           : if have_enough_info AND new_event -> ONE sentence describing the REAL event; else ""\n'
        '  question         : if unsure but something may be developing -> a short yes/no check to verify next time; this is optional you may keep it empty if there is no important question at this time; else ""\n'
        "  next_check_s     : seconds until the next check(1-3) (small when unsure or busy, larger when idle)\n"
        "Rules: describe only what you actually see; NEVER copy the example text; NEVER "
        "re-report an event you already reported; when unsure, DEFER with a question and a "
        "short next_check_s, then confirm on the next turn before answering.\n"
        "Worked examples (Situation -> Control):\n"
        "# nothing relevant on screen yet -> stay quiet, sample slowly\n"
        "# something may be starting but you are not sure -> sample densely and DEFER a targeted check\n"
        '{"fps":3,"have_enough_info":false,"new_event":false,"answer":"","question":"has the target event just started?","next_check_s":2}\n'
        "# the deferred check is now clearly confirmed -> report it ONCE\n"
        '{"fps":2,"have_enough_info":true,"new_event":true,"answer":"<one sentence describing the real event>","question":"","next_check_s":5}\n'
        "# the event is still on screen but you ALREADY reported it -> stay silent (dedup)\n"
        '{"fps":1,"have_enough_info":true,"new_event":false,"answer":"","question":"","next_check_s":4}\n'
        "Now emit ONLY your control JSON for the current stream:\n") #TODO: add a few more worked examples task specific concrete
    # These drive the control-JSON generation.
    # GREEDY by default: the control JSON is a DECISION, not creative text —
    # temperature sampling injected tick-to-tick noise into the boolean judgment
    # (random flip-flops / re-fires). Argmax makes the controller deterministic.
    writer_greedy: bool = True
    writer_seed: int = 3407
    writer_temperature: float = 0.7
    writer_top_p: float = 0.8
    writer_top_k: int = 20
    writer_repetition_penalty: float = 1.0
    writer_presence_penalty: float = 1.5

    
    # In eval the adapter sets `instruction` = the sample's task; for standalone runs the default below is used.
    instruction: str = "report the target event the instant it happens, and stay quiet otherwise"
    video_id: str = ""                # set per-sample in eval; shown in controller logs
    system_prompt: str = (
        "You are a helpful assistant watching a live video stream. "
        "According to the video you are watching, your task is: {instruction}")

    # Per-task ICL prompts (override `controller_prompt` when sample.task matches;
    # the adapter selects by task). We fill these in one category at a time.
    task_controller_prompts: dict = field(default_factory=lambda: {
        "semantic_condition_alert": _SEMANTIC_CONDITION_ALERT,
    })