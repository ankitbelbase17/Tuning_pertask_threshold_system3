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
    goal_gate_every: int = 3          # run the "goal?" probe every N frames (1 fwd
                                      # pass each). 2 halves the dominant GPU op.
    vision_gate_every: int = 8        # run the "important?" probe every N frames
    log_gate_every: int = -1           # log the goal probe every N probes


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
