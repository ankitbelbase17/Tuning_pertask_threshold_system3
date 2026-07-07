from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PromptVariant:
    key: str
    desc: str
    system_prompt: str        # uses {instruction}
    probe_question: str       # uses {event}
    writer_prompt: str        # uses {event}
    # gate knobs that some variants deliberately shift (precision vs recall)
    goal_threshold: float = 0.5

    def fill(self, instruction: str, event: str) -> dict:
        safe = {"instruction": instruction, "event": event}
        return {
            "system_prompt": self.system_prompt.format(**safe),
            "goal_question": self.probe_question.format(**safe),
            "writer_prompt": self.writer_prompt.format(**safe),
            "goal_threshold": self.goal_threshold,
        }


# A shared writer/content template reused by most variants (one short factual line).
_W = ("\nThe target event just occurred on screen. In ONE short factual sentence, "
      "describe what just happened regarding: {event}. Output only that sentence.")

VARIANTS: list[PromptVariant] = [
    # 1 — control: minimal, direct.
    PromptVariant(
        key="v01_baseline_direct",
        desc="Minimal direct framing; plain yes/no probe (control).",
        system_prompt=("You are watching a video frame by frame. "
                       "Your task: {instruction}"),
        probe_question=("\nQuestion: Based on the most recent frames, has this "
                        "happened: {event}? Answer yes or no: "),
        writer_prompt=_W,
    ),
    # 2 — explicit real-time monitor role.
    PromptVariant(
        key="v02_role_monitor",
        desc="Explicit real-time monitor role; stay-silent-until-event framing.",
        system_prompt=("You are a real-time video monitor. You watch a live stream "
                       "frame by frame and stay silent until a specific event occurs. "
                       "You are monitoring for: {instruction}"),
        probe_question=("\nQuestion: Judging only from the most recent frames, is the "
                        "monitored event happening right now ({event})? Answer yes or no: "),
        writer_prompt=_W,
    ),
    # 3 — evidence-grounded (anti-hallucination): demand visible evidence.
    PromptVariant(
        key="v03_evidence_grounded",
        desc="Forces the probe to rely on visible on-screen evidence only.",
        system_prompt=("You are a careful visual observer. Report an event only when "
                       "it is directly visible in the current frames. Watch for: {instruction}"),
        probe_question=("\nQuestion: Is there clear VISIBLE evidence in the most recent "
                        "frames that this is happening now: {event}? Answer only yes if you "
                        "can actually see it. Answer yes or no: "),
        writer_prompt=_W,
    ),
    # 4 — precision-leaning: only fire when certain (higher threshold).
    PromptVariant(
        key="v04_strict_precision",
        desc="Conservative: speak only when certain; higher trigger threshold (0.65).",
        system_prompt=("You are a precise event detector. Do NOT speak unless you are "
                       "confident the event is occurring. False alarms are costly. "
                       "Target event: {instruction}"),
        probe_question=("\nQuestion: Are you confident the event is occurring right now "
                        "({event})? Only answer yes if certain. Answer yes or no: "),
        writer_prompt=_W,
        goal_threshold=0.65,
    ),
    # 5 — recall-leaning: flag early (lower threshold).
    PromptVariant(
        key="v05_sensitive_recall",
        desc="Sensitive: flag at first sign; lower trigger threshold (0.4).",
        system_prompt=("You are a vigilant event spotter. It is better to flag the moment "
                       "you suspect the event than to miss it. Watch for: {instruction}"),
        probe_question=("\nQuestion: Is there any sign in the most recent frames that this "
                        "is starting to happen ({event})? Answer yes or no: "),
        writer_prompt=_W,
        goal_threshold=0.40,
    ),
    # 6 — chain-of-thought-lite probe.
    PromptVariant(
        key="v06_cot_lite",
        desc="Probe nudges brief internal check before committing to yes/no.",
        system_prompt=("You are an attentive analyst watching a video frame by frame. "
                       "Watch for: {instruction}"),
        probe_question=("\nQuestion: Consider what is visible in the latest frames, then "
                        "decide: has the event ({event}) just occurred? Give your final "
                        "answer as yes or no: "),
        writer_prompt=_W,
    ),
    # 7 — timeliness / recency emphasis.
    PromptVariant(
        key="v07_temporal_recent",
        desc="Emphasises 'just now / this instant' to sharpen trigger timing.",
        system_prompt=("You are a live commentator who reacts the INSTANT something "
                       "happens, not before and not after. Watch for: {instruction}"),
        probe_question=("\nQuestion: At THIS instant (the very latest frames, not earlier), "
                        "is the event happening right now: {event}? Answer yes or no: "),
        writer_prompt=_W,
    ),
    # 8 — expert persona.
    PromptVariant(
        key="v08_persona_expert",
        desc="Domain-expert observer persona for richer content.",
        system_prompt=("You are an expert observer narrating a video for someone who "
                       "cannot see it. You only speak when something noteworthy happens. "
                       "You are tracking: {instruction}"),
        probe_question=("\nQuestion: As an expert observer, would you say the event has "
                        "just occurred ({event})? Answer yes or no: "),
        writer_prompt=("\nThe event just occurred. As an expert narrator, give ONE concise, "
                       "informative sentence about what happened regarding: {event}. "
                       "Output only that sentence."),
    ),
    # 9 — restate event as explicit checkable criteria.
    PromptVariant(
        key="v09_checklist_criteria",
        desc="Reframes the event as an explicit condition to check each tick.",
        system_prompt=("You continuously check whether a specific condition is met in a "
                       "video stream. The condition to detect is: {instruction}"),
        probe_question=("\nQuestion: Checking the most recent frames against the condition "
                        "'{event}', is the condition satisfied right now? Answer yes or no: "),
        writer_prompt=_W,
    ),
    # 10 — negation-aware boundary calibration (pre/post discrimination).
    PromptVariant(
        key="v10_negation_aware",
        desc="Contrasts not-yet vs now to calibrate the onset boundary.",
        system_prompt=("You watch a video and distinguish 'not yet' from 'happening now'. "
                       "You report only at the moment of onset. Watch for: {instruction}"),
        probe_question=("\nQuestion: The event is '{event}'. Has it ALREADY started in the "
                        "most recent frames (answer yes), or has it NOT happened yet "
                        "(answer no)? Answer yes or no: "),
        writer_prompt=_W,
    ),
]

VARIANTS_BY_KEY = {v.key: v for v in VARIANTS}


def get_variants(keys: list[str] | None) -> list[PromptVariant]:
    if not keys:
        return VARIANTS
    return [VARIANTS_BY_KEY[k] for k in keys]


# ---------------------------------------------------------------------------
# Probe-mode helpers (OmniPro GT-probe protocol)
# ---------------------------------------------------------------------------
# Pre-probe expects NO (event not yet happened); post-probe expects YES / the
# structured answer. Offsets (seconds) mirror OmniPro's evaluator schedule.
PRE_OFFSETS_BY_TASK = {"semantic_condition_alert": [-5.0]}
DEFAULT_PRE_OFFSETS = [-5.0, -4.0, -3.0, -2.0]
POST_OFFSETS_BY_TASK = {"explicit_target_grounding": [0.0],
                        "snapshot_counting": [1.0],
                        "sequential_step_instruction": [2.0]}
DEFAULT_POST_OFFSETS = [0.0, 1.0, 2.0, 3.0]


def probe_offsets(task: str, *, seed_idx: int = 0):
    """Return (pre_offset, post_offset) for a trigger; deterministic pick from the
    task's allowed offsets (no RNG so runs are reproducible / resumable)."""
    pres = PRE_OFFSETS_BY_TASK.get(task, DEFAULT_PRE_OFFSETS)
    posts = POST_OFFSETS_BY_TASK.get(task, DEFAULT_POST_OFFSETS)
    return pres[seed_idx % len(pres)], posts[seed_idx % len(posts)]


def build_probe_content_question(task: str, sample) -> str:
    """The structured question asked at the POST probe to score content (beyond
    the yes/no temporal probe). Generic, templated from the sample."""
    ev = sample.event or sample.question
    if task in ("snapshot_counting", "cumulative_counting", "dedup_counting"):
        return (f"\nQuestion: Based on the frames so far, how many times has this "
                f"happened: {ev}? Answer with a single number: ")
    if task == "explicit_target_grounding":
        return (f"\nQuestion: In the current frame, where is the target ({ev})? "
                f"Answer with one of: top-left, top-center, top-right, center-left, "
                f"center, center-right, bottom-left, bottom-center, bottom-right: ")
    if task == "realtime_state_monitor":
        return (f"\nQuestion: What is the current state regarding '{ev}'? "
                f"Answer with the state in a few words: ")
    # event_narration / sequential_step_instruction and alerts -> free text
    return (f"\nQuestion: Describe in one short sentence what is happening now "
            f"regarding: {ev}. Answer: ")
