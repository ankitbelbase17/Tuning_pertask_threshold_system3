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
_SEMANTIC_CONDITION_ALERT = (
    "\nYou are the CONTROLLER of a live video monitor. Your monitoring task is in the "
    "system instruction above. Each turn, read the stream so far and emit EXACTLY ONE "
    "control command as a compact JSON object with these fields:\n"
    "  fps               : how densely to sample next (1-3; raise when the scene is busy)\n"
    "  have_enough_info  : true ONLY when the asked-for detail is actually shown on screen now\n"
    "  new_event         : true ONLY if this detail is NEW (not already under 'Already reported')\n"
    '  answer            : if have_enough_info AND new_event -> ONE sentence stating the detail; else ""\n'
    "  next_check_s      : seconds until the next check (1-3; small while waiting, larger once done)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "Rules: describe ONLY what you actually see; NEVER copy the example text; NEVER re-report "
    "a detail already listed under 'Already reported'.\n"
    "Worked example (a DIFFERENT video — learn the DECISION pattern, never copy the text):\n"
    "Task question: Please alert me whenever the video provides specific logistical details "
    "for the match, such as the date, location, or ticket pricing.\n"
    "Video: a TV commercial showing fans, then a poster with the match date and location, "
    "then ticket pricing.\n"
    "At 3s, nothing asked-for is shown yet:\n"
    '{"fps":1.0,"have_enough_info":false,"new_event":false,"answer":"","next_check_s":1,"question_for_next":"Is the date, location or ticket price shown now?"}\n'
    "(keep probing like this until the asked-for info appears)\n"
    "At 16s the date and location appear (a NEW detail):\n"
    '{"fps":1.0,"have_enough_info":true,"new_event":true,"answer":"The match date is August 14th and the location is Dairy Farmers Stadium.","next_check_s":1,"question_for_next":"Is the ticket price shown now?"}\n'
    "At 17s the same date/location still on screen (already reported -> do NOT repeat):\n"
    '{"fps":1.0,"have_enough_info":true,"new_event":false,"answer":"","next_check_s":1,"question_for_next":"Is the ticket price shown now?"}\n'
    "At 23s ticket pricing appears (a NEW detail):\n"
    '{"fps":3.0,"have_enough_info":true,"new_event":true,"answer":"The video is now detailing ticket costs and providing a website and phone number for purchases.","next_check_s":3,"question_for_next":""}\n'
    "(all asked-for info reported -> sample sparsely, no more questions, keep watching)\n"
)


@dataclass
class AsyncOmniConfig:
    # ---- model ----
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    dtype: str = "bfloat16"            # float16 / bfloat16 / float32
    # Cap vision tokens/frame by limiting image pixel area. One vision token covers
    # a (patch_size*merge_size)^2 = 32x32 = 1024 px region for Qwen3-VL, so
    # 200704 px / 1024 = ~196 tokens/frame (actual ~180 after aspect resize).
    # 0 = processor default (~2040 tokens/frame, near-memoryless).
    max_pixels: int = 200704
    profile: bool = True              # collect + print the profiling summary

    # ---- reproducibility ----
    seed: int = 0                     # seeds python/numpy/torch (+cuda) RNGs
    deterministic: bool = False       # force deterministic CUDA kernels AND make the
                                      # encoder block instead of dropping frames
                                      # -> bit-reproducible eval (use batch mode).

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
    encoder_idle_fps: float = 0.5     # fps floor when the action is boring

    # ---- memory (shared linear KV cache) ----
    kv_budget: int = 262144           # 256K-token context (StreamingLLM eviction
                                      # past this; the system-prompt sink is pinned).

    # ---- the CONTROLLER (icl_ingester_writer) ----
    # Self-paced cadence: the controller picks next_check_s each tick; clamp it so
    # it can't spin (min) or stall (max); fall back to default if unparsed.
    probe_default_s: float = 3.0
    probe_min_s: float = 1.0
    probe_max_s: float = 3.0
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
    # These drive the control-JSON generation
    writer_greedy: bool = False
    writer_seed: int = 3407
    writer_temperature: float = 0.7
    writer_top_p: float = 0.8
    writer_top_k: int = 20
    writer_repetition_penalty: float = 1.0
    writer_presence_penalty: float = 1.5

    
    # In eval the adapter sets `instruction` = the sample's task; for standalone runs the default below is used.
    instruction: str = "report the target event the instant it happens, and stay quiet otherwise"
    system_prompt: str = (
        "You are a helpful assistant watching a live video stream. "
        "According to the video you are watching, your task is: {instruction}")

    # Per-task ICL prompts (override `controller_prompt` when sample.task matches;
    # the adapter selects by task). We fill these in one category at a time.
    task_controller_prompts: dict = field(default_factory=lambda: {
        "semantic_condition_alert": _SEMANTIC_CONDITION_ALERT,
    })