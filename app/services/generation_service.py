"""Owns the generate flow: translate a request into a scenario and run the agent.

The service owns the overall timeout around the agent run, so a runaway loop
can never hold a request open beyond the configured budget. When a reference
email is provided, it also computes the reference comparison group so all three
custom metric groups are returned.
"""
from __future__ import annotations

import asyncio

from app.agents.email_agent import EmailAgent
from app.config import Settings
from app.core.embeddings import JinaEmbedder
from app.core.exceptions import AppError
from app.metrics.evaluator import weighted_overall
from app.metrics.reference import reference_metrics
from app.models.domain import AgentResult, ReferenceEmail, Scenario
from app.schemas.api import GenerateRequest


class GenerationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(
        self,
        request: GenerateRequest,
        agent: EmailAgent,
        embedder: JinaEmbedder,
    ) -> AgentResult:
        scenario = Scenario(
            id="adhoc",
            category="adhoc",
            intent=request.intent,
            key_facts=request.facts,
            tone=request.tone,
            reference_email=ReferenceEmail(subject="", body=request.reference or ""),
        )
        try:
            async with asyncio.timeout(self._settings.generation_timeout):
                result = await agent.run(
                    scenario,
                    provider=request.provider,
                    max_attempts=request.max_attempts,
                    threshold=request.threshold,
                )
        except TimeoutError as exc:
            raise AppError("generation timed out", status_code=504) from exc

        if request.reference and embedder is not None:
            reference = await asyncio.to_thread(
                reference_metrics, result.body, request.reference, embedder
            )
            result.scores["reference"] = reference
            groups = {key: value for key, value in result.scores.items() if key != "overall"}
            result.overall = weighted_overall(groups)
            result.scores["overall"] = result.overall

        return result
