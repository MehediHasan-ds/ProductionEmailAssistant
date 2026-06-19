"""Builds the critique fed back into the prompt on a refinement retry.

Takes the weakest scoring dimensions from the rule and judge groups and turns
them into concrete, model actionable feedback.
"""
from __future__ import annotations

from typing import Any

HINTS = {
    "fact_coverage": "include every key fact explicitly",
    "tone_match": "match the requested tone more clearly",
    "tone_fidelity": "match the requested tone more clearly",
    "structure": "keep a clear subject, greeting, body, and closing",
    "length": "keep the length in a concise professional range",
    "readability": "simplify sentence structure for readability",
    "placeholder_leak": "remove any placeholder text such as brackets",
    "hallucination_flag": "avoid adding details not given in the key facts",
    "redundancy": "reduce repetition",
    "fact_integration": "weave every key fact in naturally, without inventing any",
    "professionalism": "raise professionalism, fix grammar, drop slang",
    "clarity_coherence": "improve clarity and coherence",
    "intent_alignment": "stay tightly aligned with the stated intent",
}


def build_critique(scores: dict[str, Any]) -> str:
    flat: list[tuple[str, float]] = []
    for group in ("rule", "judge"):
        sub = scores.get(group)
        if isinstance(sub, dict):
            for name, value in sub.items():
                if isinstance(value, (int, float)):
                    flat.append((name, float(value)))
    flat.sort(key=lambda item: item[1])
    weakest = [item for item in flat if item[0] in HINTS][:3]
    if not weakest:
        return "Improve the weakest aspects of the email while keeping all key facts."
    lines = [f"{name} scored {value:.2f}: {HINTS[name]}." for name, value in weakest]
    return " ".join(lines)
