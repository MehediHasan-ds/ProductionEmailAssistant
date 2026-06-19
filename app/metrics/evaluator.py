"""Hybrid evaluator: combines rule based, judge, and reference metric groups."""
from __future__ import annotations

from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient
from app.metrics.judge import judge_email
from app.metrics.reference import reference_metrics
from app.metrics.rule_based import rule_based_metrics
from app.models.domain import Scenario

WEIGHTS = {"rule": 0.3, "judge": 0.4, "reference": 0.3}


def weighted_overall(groups: dict[str, dict[str, float]]) -> float:
    total = 0.0
    weight_used = 0.0
    for group, weight in WEIGHTS.items():
        scores = groups.get(group)
        if not scores:
            continue
        mean = sum(scores.values()) / len(scores)
        total += mean * weight
        weight_used += weight
    if weight_used == 0:
        return 0.0
    return round(100 * total / weight_used, 2)


async def evaluate_email(
    subject: str,
    body: str,
    scenario: Scenario,
    client: LLMClient,
    provider: str,
    embedder: JinaEmbedder | None = None,
    with_judge: bool = True,
    with_reference: bool = False,
) -> dict[str, object]:
    groups: dict[str, dict[str, float]] = {
        "rule": rule_based_metrics(subject, body, scenario.key_facts, scenario.tone)
    }
    if with_judge:
        groups["judge"] = await judge_email(
            subject, body, scenario.intent, scenario.key_facts, scenario.tone, client, provider
        )
    if with_reference and embedder is not None:
        groups["reference"] = reference_metrics(body, scenario.reference_email.body, embedder)

    return {**groups, "overall": weighted_overall(groups)}
