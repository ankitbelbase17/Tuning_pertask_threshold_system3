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


# NOTE ON AUDIO: the paper defines ETG "audio-first" (prefer audio triggers); we run the
# audio_dependency=none subset only, so this ICL is purely VISUAL — no audio references.
#
# Paper's task intent: the question names a TRIGGER moment and a TARGET object. When the
# trigger fires (instantaneous, frame-precise), locate the target on a 3x3 grid. Scoring
# is EXACT-MATCH on the grid cell extracted from the answer text (no LLM judge), plus the
# usual ±3s temporal match — so the answer MUST contain exactly one of the nine cell
# names, and firing time must be sharp. Usually ONE trigger per video (max 4).
_EXPLICIT_TARGET_GROUNDING = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) names a TRIGGER moment and a TARGET object: the moment the "
    "trigger happens, report WHERE the target is on screen using EXACTLY ONE of these "
    "nine grid cells (imagine the frame divided 3x3):\n"
    "  top-left | top-center | top-right | center-left | center | center-right | "
    "bottom-left | bottom-center | bottom-right\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS "
    "start with seen, then have_enough_info — looking BEFORE judging, every tick. "
    "Whenever have_enough_info is true, ALSO include event_time_s and answer. Include "
    "fps, next_check_s, or question_for_next ONLY when they change. Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now, relevant to the trigger/target\n'
    "  have_enough_info  : true when the TRIGGER is happening on screen NOW and the "
    "target is visible; back to false once it is over\n"
    "  event_time_s      : when true -> the video time when the trigger occurred (read "
    "the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED when true -> ONE sentence UNDER 20 words naming the '
    'trigger AND the target\'s grid cell; it MUST contain exactly one cell name; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the trigger feels close)\n"
    "  next_check_s      : seconds until the next check (small — the trigger is a precise instant)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does (it alerts only when "
    "have_enough_info goes false -> true). The trigger usually happens ONCE; after "
    "reporting, return to false when it is over and keep watching in case it recurs.\n"
    "The 'Already reported' list below shows PAST reports with their times.\n"
    "Rules: report only what is actually visible; the answer must contain exactly ONE "
    "grid-cell name; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — a fast-paced montage of short sports clips: "
    "cycling, tennis, cardio, big crowds. Task: 'When the drummer first appears playing "
    "his drums, tell me where his red cap is in the frame.'):\n"
    "At 5s, cyclists racing, no drummer yet:\n"
    '{"seen":"cyclists racing past a crowd","have_enough_info":false,"fps":1.0,"question_for_next":"Has the drummer appeared playing his drums?"}\n'
    "At 12s, tennis rally, still no drummer:\n"
    '{"seen":"tennis player mid-rally","have_enough_info":false}\n'
    "At 19s the drummer appears playing, red cap visible in the upper middle -> TRIGGER; "
    "locate the cap and read the time:\n"
    '{"seen":"drummer playing, red cap upper middle","have_enough_info":true,"event_time_s":19,"answer":"The drummer is now playing - his red cap is in the top-center of the frame."}\n'
    "At 21s the montage cut away, drummer gone -> trigger over:\n"
    '{"seen":"crowd cheering, drummer gone","have_enough_info":false}\n'
)

# Probe-gate arm needs an ETG-specific WRITER prompt too (its generic writer says
# "what+why", which would never contain a grid cell -> auto-wrong on exact match).
_ETG_WRITER_PROMPT = (
    "\nThe trigger just occurred on screen. In ONE sentence UNDER 20 words, state the "
    "trigger and WHERE the target is, using exactly one of: top-left, top-center, "
    "top-right, center-left, center, center-right, bottom-left, bottom-center, "
    "bottom-right. Output only that sentence.")


# NOTE ON AUDIO: the paper defines IEA "audio-first" (prefer whistles/doorbells/
# spoken phrases). We run the audio_dependency=none subset only, so this ICL is
# purely VISUAL — the target is a sharp VISIBLE onset and there are no audio cues.
#
# Task intent: a standing instruction ("Let me know when X happens"). Alert the
# INSTANT the event STARTS on screen; timing is frame-precise (±3s temporal match)
# and the answer is a short natural description (LLM-judged, not exact-match).
# Usually ONE occurrence per video (max ~3), so after it ends keep watching.
_INSTANT_EVENT_ALERT = (
    "\nYou are the CONTROLLER of a live video monitor. Your task (in the system "
    "instruction above) is a STANDING INSTRUCTION to alert the user the moment a "
    "specific EVENT happens on screen (e.g. 'let me know when the audience starts "
    "clapping'). The event is a SHARP ONSET — the instant something STARTS — so your "
    "timing must be precise: fire when it BEGINS, not before. It usually happens ONCE, "
    "but can recur a few times; after it ends, keep watching for the next occurrence.\n"
    "Each turn, read the stream so far and emit a compact JSON control update. ALWAYS "
    "start with seen, then have_enough_info — looking BEFORE judging, every tick. "
    "Whenever have_enough_info is true, ALSO include event_time_s and answer. Include "
    "fps, next_check_s, or question_for_next ONLY when they change from their current "
    "value (otherwise omit them to stay short). Fields:\n"
    '  seen              : ALWAYS FIRST — 3-8 words: what is on screen now, relevant to the event\n'
    "  have_enough_info  : true the moment the event is happening on screen NOW; stays true "
    "while it continues; back to false once it is over\n"
    "  event_time_s      : when have_enough_info is true -> the video time in seconds when THIS "
    "occurrence STARTED (read it off the 'time Xs' markers in the stream)\n"
    '  answer            : REQUIRED whenever have_enough_info is true -> ONE natural sentence '
    '(UNDER 20 words) stating WHAT just happened; else ""\n'
    "  fps               : how densely to sample next (1-3; raise when the event feels close)\n"
    "  next_check_s      : seconds until the next check (small — the onset is a precise instant)\n"
    '  question_for_next : a short check to verify on the next turn; else ""\n'
    "YOU DO NOT DECIDE WHEN TO ALERT — the system does. It alerts the user only when "
    "have_enough_info goes false -> true (or when your answer describes a clearly DIFFERENT "
    "occurrence), so you will NEVER double-alert by keeping it true while the same event stays on "
    "screen. Your only job each tick: judge honestly whether the event is happening NOW and "
    "describe what you see.\n"
    "The 'Already reported' list below shows PAST occurrences WITH THEIR TIMES.\n"
    "Rules: base every judgment on what is actually visible; do NOT fire before the event "
    "actually starts; NEVER copy the example text.\n"
    "Worked example (a DIFFERENT video — instruction: 'Let me know when the audience starts "
    "clapping'). The video is a university auditorium: speakers give speeches about the "
    "university, the camera cutting between the podium and the seated audience. Every output "
    "starts with seen + have_enough_info; other fields appear only when they change:\n"
    "At 20s a speaker is talking, the audience is seated and still -> event NOT happening:\n"
    '{"seen":"speaker at podium, audience seated still","have_enough_info":false,"fps":1.0,"question_for_next":"Has the audience started clapping?"}\n'
    "At 34s the speaker is wrapping up his segment, still no clapping (question unchanged -> omit it):\n"
    '{"seen":"speaker finishing his remarks","have_enough_info":false}\n'
    "At 38s the camera cuts to the audience clapping -> event STARTS; describe it and read its time:\n"
    '{"seen":"audience clapping in their seats","have_enough_info":true,"event_time_s":38,"answer":"The audience is clapping now after the opening remarks.","next_check_s":1}\n'
    "At 40s they are still clapping -> STILL happening; keep reporting it (the system will not double-alert):\n"
    '{"seen":"audience still applauding","have_enough_info":true,"event_time_s":38,"answer":"The audience is clapping now after the opening remarks."}\n'
    "At 48s the clapping has stopped and the next speaker is walking up -> event over:\n"
    '{"seen":"applause stopped, new speaker approaching","have_enough_info":false}\n'
    "At 64s the audience breaks into applause again as a family is invited on stage -> a NEW occurrence:\n"
    '{"seen":"audience applauding again","have_enough_info":true,"event_time_s":64,"answer":"The audience has started applauding again as the family is invited to the stage."}\n'
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
    max_seconds: float = 600.0
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
    event: str = ""                   # the monitored condition (adapter: sample.event)
    video_id: str = ""                # set per-sample in eval; shown in controller logs

    # ---- PROBE-GATE system (gate_mode="probe"): the main-branch baseline --------
    # Fixed-cadence yes/no LOGIT gate + separate writer, restored for the
    # probe-vs-controller head-to-head (see EXPERIENCE.md). One forward pass per
    # probe reads yes_share = P(yes)/(P(yes)+P(no)); a Schmitt/hysteresis gate
    # (tuned "hyst2b") fires the writer, which snapshots the cache and answers.
    gate_mode: str = "controller"     # "controller" (icl DSL, default) | "probe"
    goal_question: str = (
        "\nQuestion: Based on the most recent frames, has this happened: {event}? "
        "Answer yes or no: ")
    writer_prompt: str = (
        "\nThe monitored event just occurred on screen. In ONE sentence UNDER 25 "
        "words state WHAT happened AND WHY it satisfies the condition: {event}. "
        "Output only that sentence.")
    writer_cue: str = "\nALERT: "
    writer_max_tokens: int = 60
    writer_repeat_window: int = 8     # anti-loop guard for the free-running writer
    goal_threshold: float = 0.5       # non-hysteresis fallback threshold
    gate_hysteresis: bool = True      # hyst2b (best tuned gate on the 27-video eval)
    gate_high_thr: float = 0.5        # fire on a rising crossing (while armed)
    gate_low_thr: float = 0.40        # re-arm when the share falls below this...
    gate_rearm_s: float = 5.0         # ...OR this many seconds after a fire
    goal_gate_every: int = 1          # probe every frame (@1fps ~= controller's grid)
    debounce_s: float = 2.0           # min video-seconds between fires
    yes_words: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes"])
    no_words: list = field(default_factory=lambda: ["no", "No", " no", " No"])
    system_prompt: str = (
        "You are a helpful assistant watching a live video stream. "
        "According to the video you are watching, your task is: {instruction}")

    # Per-task ICL prompts (override `controller_prompt` when sample.task matches;
    # the adapter selects by task). We fill these in one category at a time.
    task_controller_prompts: dict = field(default_factory=lambda: {
        "semantic_condition_alert": _SEMANTIC_CONDITION_ALERT,
        "explicit_target_grounding": _EXPLICIT_TARGET_GROUNDING,
        "instant_event_alert": _INSTANT_EVENT_ALERT,
    })
    # Per-task writer prompts for the PROBE-GATE arm (same idea: the answer format
    # is task knowledge both systems get; the architecture is what differs).
    task_writer_prompts: dict = field(default_factory=lambda: {
        "explicit_target_grounding": _ETG_WRITER_PROMPT,
    })



"""
# Prompt generation information from me to ai agent for remaining tasks should be similar to those above 3 tasks.
For now we do not use the audio signal at all. so the prompts should refrain from telling anything about audio, evn though the prompts i pasted here from the original eval paper contain the prompt about audio
https://github.com/RuixiangZhao/OmniPro this is the official repo of the OmniPro paper. The prompts above are copied from the original repo. avoiding the audio paert.

next we will use audio_dependency=none or helpful subsets only. on all the tasks.

Below i give some real video examples for in context learning ICL you should use to generate the prompts for the remaining tasks.

## Task: dedup_counting 
Video description: A video showing a man inteviewing in front of the camera giving the review of the restaurant, tables of a restaurant, a chef handling pizza in the oven and a girl working in the center of the restaurant while other people are eating and talking.

Question: Count how many different people are featured as the primary subject of a scene while working or giving an interview.
ground truth answer: [{"trigger_time": "00:02", "trigger_time_sec": 2, "response": "First person — Rice, the interviewee wearing a t-shirt.", "trigger_type": "visual", "event_description": "Rice first appears giving an interview", "count": 1}, {"trigger_time": "00:8", "trigger_time_sec": 10, "response": "Second person — the chef in a white shirt and black cap working in the kitchen.", "trigger_type": "visual", "event_description": "The chef first appears working in the kitchen", "count": 2}, {"trigger_time": "00:26", "trigger_time_sec": 26, "response": "Third person — a blonde girl with glasses working at a table.", "trigger_type": "visual", "event_description": "The girl first appears working", "count": 3}]


## Task: realtime_state_monitor
 AI has to trigger every time the question asked event happens in the video. 
 Video description: Video shows a promo poster of a red van, then shows the van's exterior from every angle slowly rotating around. The van is in a parking lot. There are many vehicles parked in every direction. then shows the driver's seat and controls then the other seats and interior of the van. At the end, it shows the van driving away. The video is a promo for a new van model.
  Question: Monitor the type of footage being shown and let me know when it switches between the ad graphic, the van's exterior, and the van's interior. 
  ground truth answer: [{"trigger_time": "00:05", "trigger_time_sec": 2, response: "Switched from ad graphic to van's exterior.", "trigger_type": "visual", "event_description": "The video switches from the ad graphic to the van's exterior."}, {"trigger_time": "00:10", "trigger_time_sec": 10, response: "Switched from van's exterior to van's interior.", "trigger_type": "visual", "event_description": "The video switches from the van's exterior to the van's interior."}, {"trigger_time": "00:15", "trigger_time_sec": 15, response: "Switched from van's interior to ad graphic.", "trigger_type": "visual", "event_description": "The video switches from the van's interior back to the ad graphic."}]

## Task: static_object_counting
for this task there is only 1 triggering per task
Video description: A video showing players practicing penalty shooting in a football field. A keeper is standing in front of the goal post. The players are taking turns to shoot the ball towards the goal post. Many players miss the goal. Some are saved by the keeper. At one point a man in the red shirt scores a goal. Then the replay is shown slowly. There are 3 balls seen in the goal post area. After that, video continues to show the players practicing penalty shooting. 

Question: When a player in red shirt first scores a goal count how many balls are inside the net?

ground truth answer: [{"trigger_time": "00:42", "trigger_time_sec": 42, "response": "There are 3 balls inside the net.", "trigger_type": "visual", "event_description": "The player in red shirt scores the first goal and there are 3 balls inside the net."}]

## Task: cumulative_counting
Video description: A long video showing the latest innovations in technology for everyday people. It starts with mosquito repellent, shows people using it in differnt ways in different locations. Then it shows a new technology for cleaning the air, then shows people using it in different locations. Then it shows phone covers that protect from water and dust, then shows people using it in different locations. Then it shows a new technology for cleaning the water, then shows people using it in different locations.

Question: Count how many times you see poeple using the mobile phone in the video.

Ground truth answer: [{"trigger_time": "00:50", "trigger_time_sec": 50, "response": "First occurrence — a man using a phone while walking in the park.", "trigger_type": "visual", "event_description": "A man is seen using a mobile phone while walking in the park.", "count": 1}, {"trigger_time": "01:20", "trigger_time_sec": 80, "response": "Second occurrence — a woman using a phone while sitting in a cafe.", "trigger_type": "visual", "event_description: "A woman is seen using a mobile phone while sitting in a cafe.", "count": 2}, {"trigger_time": "02:10", "trigger_time_sec": 130, "response": "Third occurrence — a man using a phone while waiting at a bus stop.", "trigger_type": "visual", "event_description": "A man is seen using a mobile phone while waiting at a bus stop.", "count": 3}, {"trigger_time": "02:50", "trigger_time_sec": 170, "response": "Fourth occurrence — a blonde woman using a phone while taking care of her child in the kitchen.", "trigger_type": "visual", "event_description": "A blonde woman is seen using   a mobile phone while taking care of her child in the kitchen.", "count": 4}}]

## Task: sequential_step_instruction

Video Description: Video shows a person preparing a hot cup of green tea. The person starts by boiling water in a kettle. Once the water is boiled then places kettle on table for a while to let it cool down a little. Then put a green tea bag into a cup. they pour it over the tea bag in the cup. After allowing the tea to steep for a few minutes, they remove the tea bag and add honey . Finally, they stir the tea and enjoy their hot cup of green tea.

Question: I'm insterested in how to prepare hot cup of green tea. Please guide me through the process of making a hot cup of green tea step by step.

ground truth answer: [{"trigger_time": "00:02", "trigger_time_sec": 2, "response": "Step 1 — Boil water in a kettle.", "trigger_type": "visual", "event_description": "The person is seen boiling water in a kettle.", "step": 1}, {"trigger_time": "00:10", "trigger_time_sec": 10, "response": "Step 2 — Place the boiled water on the table to cool slightly.", "trigger_type": "visual", "event_description": "The person places the kettle on the table to let it cool down a little.", "step": 2}, {"trigger_time": "00:20", "trigger_time_sec": 20, "response": "Step 3 — Put a green tea bag into a cup.", "trigger_type": "visual", "event_description": "The person puts a green tea bag into a cup.", "step": 3}, {"trigger_time": "00:30", "trigger_time_sec": 30, "response": "Step 4 — Pour the hot water over the tea bag in the cup.", "trigger_type": "visual", "event_description": "The person pours the hot water over the tea bag in the cup.", "step": 4}, {"trigger_time": "00:40", "trigger_time_sec": 40, "response": "Step 5 — Allow the tea to steep for a few minutes.", "trigger_type": "visual", "event_description": "The person allows the tea to steep for a few minutes.", "step": 5}, {"trigger_time": "00:50", "trigger_time_sec": 50, "response": "Step 6 — Remove the tea bag and add honey.", "trigger_type": "visual", "event_description": "The person removes the tea bag and adds honey to the cup.", "step": 6}, {"trigger_time": "01:00", "trigger_time_sec": 60, response: Step 7 — Stir the tea and enjoy your hot cup of green tea.", trigger_type: visual, event_description: The person stirs the tea and enjoys their hot cup of green tea., step: 7}]


## Task: event_narration

Video Description: A video showing highlights of a football match. The video starts with the players warming up on the field, followed by the kickoff. The match progresses with several attacks and defenses from both teams. A player from the France team scores a goal, leading to celebrations from the fans. The video also shows a few fouls and yellow cards being issued. Towards the end, the Senegal team manages to score an equalizer, and the match ends in a draw.

Question: Provide a running narration of the highlight of a football match.

Ground truth answer: [{"trigger_time": "00:02", "trigger_time_sec": 2, "response": "The players are warming up on the field.", "trigger_type": "visual", "event_description": "The video shows the players warming up on the field."}, {"trigger_time": "00:10", "trigger_time_sec": 10, "response": "The match kicks off with both teams ready to play.", "trigger_type": "visual", "event_description": "The video shows the kickoff of the match."}, {"trigger_time": "00:23", "trigger_time_sec": 23, "response": "France team attacks and scores a goal.", "trigger_type": "visual", "event_description": "A player from the France team scores a goal."}, {"trigger_time": "00:29", "trigger_time_sec": 29, "response": "Fans celebrate the goal scored by France.", "trigger_type": "visual", "event_description": "The video shows fans celebrating the goal scored by France."}, {"trigger_time": "00:40", "trigger_time_sec": 40, response: "Senegal team attacks and gets a yellow card for a foul.", "trigger_type": "visual", "event_description": "A player from the Senegal team commits a foul and receives a yellow card."}, {"trigger_time: "00:45", "trigger_time_sec": 45, "response": "France team attacks again and gets a yellow card for a foul.", "trigger_type": "visual", "event_description": "A player from the France team commits a foul and receives a yellow card."} {"trigger_time": "00:50", "trigger_time_sec": 56,
response: Senegal team scores an equalizer., trigger_type: visual, event_description: The Senegal team manages to score an equalizer., step: 7}]

## Task: 
"""

