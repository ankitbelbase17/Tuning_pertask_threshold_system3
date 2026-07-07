"""
prompts.py — the single prompt for the icl_ingester_writer eval.

The pipeline is generic: the task reaches the model only through the system
prompt, templated per-sample from the OmniPro fields:
  {instruction} = sample["question"]  (e.g. "Tell me when the video asks us to sing along")

The controller drives everything else via its own control-JSON DSL (see
async_omni_v2/config.py:controller_prompt), so there is no yes/no probe or
separate writer prompt here.

(The 10-variant prompt sweep + probe-mode helpers live on the `main` branch.)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptVariant:
    key: str
    desc: str
    system_prompt: str        # uses {instruction}
    goal_threshold: float = 0.5   # kept for eval-harness compatibility (unused by controller)

    def fill(self, instruction: str, event: str) -> dict:
        return {"system_prompt": self.system_prompt.format(instruction=instruction)}


VARIANTS: list[PromptVariant] = [
    PromptVariant(
        key="v01_baseline_direct",
        desc="Minimal direct framing; the controller drives detection via its JSON DSL.",
        system_prompt=("You are a helpful assistant watching a live video stream. "
                       "According to the video you are watching, your task is: {instruction}"),
    ),
]

VARIANTS_BY_KEY = {v.key: v for v in VARIANTS}


def get_variants(keys: list[str] | None) -> list[PromptVariant]:
    if not keys:
        return VARIANTS
    return [VARIANTS_BY_KEY[k] for k in keys]
