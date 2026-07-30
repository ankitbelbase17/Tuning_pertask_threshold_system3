"""
config.py — the single config for the icl_ingester_writer pipeline.

ONE proactivity mode only: the pure-generative CONTROLLER (probe_scheduler was
"model"). The encoder streams frames -> the ingester prefills them into a shared
KV cache -> the controller reads that cache each tick and emits ONE control JSON
(fps / have_enough_info / new_event / answer / question / next_check_s), which is
both the output gate and the writer. There are no fixed yes/no gates here.

SETTINGS ONLY. All prompt TEXT lives in `prompts.py` — one home for every string
we say to the model (per-task ICL for all 9 tasks, the generic controller DSL,
the probe-gate/system templates). This file just names and wires them. The eval
harness carries no prompt text either.

(The fixed-gate ablations, multi-GPU replicas, and VisionZip pruning live on the
`main` branch.)
"""
from dataclasses import dataclass, field

# ALL prompt text lives in prompts.py (settings here, content there).
from prompts import (CONTROLLER_PROMPT_GENERIC, GOAL_QUESTION, SYSTEM_PROMPT,
                     TASK_CONTROLLER_PROMPTS, TASK_WRITER_PROMPTS, WRITER_CUE,
                     WRITER_PROMPT)
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
    controller_max_tokens: int = 300  # cap for the control-JSON generation (free mode)

    # ---- SCHEMA-WALKED DECODE (the fix for "the diff never diffs") ------------
    # "schema": code force-feeds every key/punctuation as a batched PREFILL and the
    #           model only samples value slots; booleans are READ from the logits at
    #           the forced position (zero decode steps, continuous confidence).
    #           Measured motivation: 25 of the 35 tokens in a quiet tick were JSON
    #           structure the code already knew, and the prose rule "omit fps unless
    #           it changed" was obeyed 0/15 ticks. A diff is a DECODER CONSTRAINT,
    #           not a prompt instruction.
    # "free":   the legacy open-brace generate loop (kept for A/B).
    decode_mode: str = "schema"
    # PRIORITY-1 EXPERIMENT (ROADMAP): does the hit read need `seen` first?
    #   "before" -- describe the scene, then read the level (~1.3s/tick, current)
    #   "off"    -- read the level immediately, zero decodes (~0.15s/tick)
    #   "after"  -- read the level first, then describe (separates the effect of
    #               the perception step from the effect of its ORDER)
    # "off" would make the always-on trigger affordable; `seen` is also the single
    # biggest accuracy lever we have (F1 0.0 -> 0.255), so this must be measured.
    seen_mode: str = "before"
    schema_max_seen_tokens: int = 12    # cap on the `seen` value slot
    schema_max_answer_tokens: int = 32  # cap on the `answer` value slot (hot ticks only)
    schema_max_int_tokens: int = 4      # cap on `event_time_s`
    schema_max_tail_tokens: int = 60    # cap on the `more` escape-hatch tail
    hit_threshold: float = 0.5          # P(true) above which the level is TRUE
    # PER-TASK firing thresholds. The single global 0.5 was wrong for every task
    # measured: two tasks scored time_f1 0.000 not because the model failed to
    # perceive, but because p_hit never crossed 0.5 so the gate could never fire.
    # Fitted offline by auc.py (replay the rising-edge gate over saved p_hit, then
    # greedy temporal match) on the 2026-07-30 smoke run:
    #     task                      thr    time_f1@thr   was@0.5
    #     dedup_counting           0.75       0.632       0.444
    #     realtime_state_monitor   0.14       0.455       0.133
    #     semantic_condition_alert 0.85       0.353       0.000
    #     instant_event_alert      0.12       0.222       0.000
    # ⚠️ FITTED ON ONLY 3 VIDEOS PER TASK. These are provisional: they must be
    # re-fitted and validated on held-out videos before ANY reported number.
    task_hit_thresholds: dict = field(default_factory=lambda: {
        "dedup_counting": 0.75,
        "realtime_state_monitor": 0.14,
        "semantic_condition_alert": 0.85,
        "instant_event_alert": 0.12,
    })
    more_threshold: float = 0.5         # P(true) above which the tail is decoded
    # Log, every tick, what an UNRESTRICTED argmax would have produced at the
    # boolean slot. If it is not a boolean at all, the logit read is imposing
    # structure the model did not intend — we must know before trusting this path.
    verify_logit_read: bool = True
    notes_ring: int = 8                 # model-authored notes (bounded ring)
    # MEMORY (MISSION pillar 7) — the writer's own trace, fed back into the prompt:
    # WHAT I SAW (`seen`, consecutive duplicates collapsed) + WHAT I SAID
    # (`reported`, code-owned so the model cannot fake having answered).
    # Bounded so per-tick prompt cost is O(1), not O(stream length) — a memory that
    # grows without bound would make the system slower the longer it watches.
    seen_trace_ring: int = 10           # how many past observations to show

    # In-context "control language" (VISPROG-style): a compact JSON DSL the frozen
    # model emits each tick to drive its own probing. Taught via worked
    # (Situation -> Control) pairs that demonstrate the DECISIONS (stay quiet /
    # defer-with-question / fire-once / suppress-repeat), not just the syntax.
    # This GENERIC block is only the fallback: `task_controller_prompts` overrides
    # it for every one of the 9 OmniPro tasks. Text lives in prompts.py.
    controller_prompt: str = CONTROLLER_PROMPT_GENERIC

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
    # Prompt TEXT for these lives in prompts.py; {event} / {instruction} are filled
    # at runtime by the ingester/writer.
    goal_question: str = GOAL_QUESTION
    writer_prompt: str = WRITER_PROMPT
    writer_cue: str = WRITER_CUE
    system_prompt: str = SYSTEM_PROMPT
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

    # Per-task ICL prompts (override `controller_prompt` when sample.task matches;
    # the adapter selects by task). ALL 9 OmniPro tasks are covered. Text lives in
    # prompts.py; counting/state tasks additionally carry `count` / `phase`, which
    # the controller state now persists.
    # Copied per instance so a caller mutating one config cannot corrupt the shared
    # module-level dict.
    task_controller_prompts: dict = field(
        default_factory=lambda: dict(TASK_CONTROLLER_PROMPTS))
    # Per-task writer prompts for the PROBE-GATE arm (same idea: the answer format
    # is task knowledge both systems get; the architecture is what differs).
    task_writer_prompts: dict = field(
        default_factory=lambda: dict(TASK_WRITER_PROMPTS))



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

