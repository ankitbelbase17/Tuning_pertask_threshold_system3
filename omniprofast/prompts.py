"""
prompts.py — variant glue ONLY. Contains NO prompt text.

All editable prompt text (system_prompt template + controller_prompt DSL) lives
in ONE place: async_omni_v2/config.py. Here we just carry the eval "variant"
identity (key/desc) and pass each sample's task instruction through to the
adapter, which fills config.py's system_prompt template with it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PromptVariant:
    key: str
    desc: str
    goal_threshold: float = 0.5   # kept for eval-harness compatibility (unused)

    def fill(self, instruction: str, event: str) -> dict:
        # No prompt text here — just hand the task instruction to the adapter,
        # which fills config.py's system_prompt template.
        return {"instruction": instruction}


VARIANTS: list[PromptVariant] = [
    PromptVariant(key="v01_baseline_direct",
                  desc="icl_ingester_writer; prompts defined in async_omni_v2/config.py."),
]

VARIANTS_BY_KEY = {v.key: v for v in VARIANTS}


def get_variants(keys: list[str] | None) -> list[PromptVariant]:
    if not keys:
        return VARIANTS
    return [VARIANTS_BY_KEY[k] for k in keys]
