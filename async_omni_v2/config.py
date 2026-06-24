"""
config.py — every knob in one place.

All tunables (model, pacing, memory budget, gating thresholds) and all prompts
live here as a single dataclass so the rest of the code never hard-codes a
magic number. Construct one `AsyncOmniConfig`, pass it around.
"""
from dataclasses import dataclass, field


@dataclass
class AsyncOmniConfig:
    # ---- model ----
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    device: str = "cuda"
    dtype: str = "float16"            # float16 / bfloat16 / float32

    # ---- video / pacing ----
    video_path: str = ""
    fps: float = 1.5                  # frames sampled per second of video
    max_seconds: float = 120.0
    realtime: bool = True             # pace the vision stream to a wall clock
    speed: float = 2.0                # real-time multiplier (2.0 = 2x live)
    frame_q_size: int = 8

    # ---- memory management (shared linear KV cache) ----
    kv_budget: int = 4096             # max tokens kept in the cache
    # `sink` (the protected system-prompt prefix) is measured at seed time.

    # ---- proactivity gating ----
    goal_threshold: float = 0.5       # writer fires when goal yes_share >= this
    debounce_s: float = 8.0           # min video-seconds between writer triggers
    vision_gate_every: int = 8        # run the "look closer?" probe every N frames
    log_gate_every: int = 1           # log the goal probe every N frames

    # ---- writer ----
    writer_max_tokens: int = 32
    writer_temperature: float = 0.7   # >0 -> sampling (kills greedy loops)
    writer_repeat_window: int = 8     # block immediate n-gram repeats

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
    writer_cue: str = "\nLIVE: "

    # ---- yes/no surface forms scored by the proactivity probe ----
    yes_words: list = field(default_factory=lambda: ["yes", "Yes", " yes", " Yes"])
    no_words: list = field(default_factory=lambda: ["no", "No", " no", " No"])
