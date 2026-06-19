"""The email agent. Owns the generate, evaluate, refine loop.

This is the enforced review pass the root cause analysis demanded. It generates
a draft, scores it, and if the score is below threshold feeds a critique back
into the prompt and tries again, up to a capped number of attempts. It always
returns the best draft by score.
"""
from __future__ import annotations

import structlog

from app.agents.critic import build_critique
from app.agents.prompts import build_messages, parse_email_response
from app.config import Settings
from app.core.llm_client import LLMClient
from app.metrics.evaluator import evaluate_email
from app.models.domain import AgentResult, AttemptRecord, Scenario

log = structlog.get_logger(__name__)


class EmailAgent:
    def __init__(self, client: LLMClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def run(
        self,
        scenario: Scenario,
        provider: str | None = None,
        max_attempts: int | None = None,
        threshold: float | None = None,
    ) -> AgentResult:
        provider = provider or self._settings.default_provider
        max_attempts = max_attempts or self._settings.max_attempts
        threshold = threshold or self._settings.pass_threshold

        feedback: str | None = None
        best: AttemptRecord | None = None
        best_overall = -1.0
        trace: list[AttemptRecord] = []

        for attempt in range(1, max_attempts + 1):
            messages = build_messages(scenario.intent, scenario.key_facts, scenario.tone, feedback)
            raw = await self._client.chat(
                # tried json_mode=True first, openrouter returns empty body
                messages, provider=provider, temperature=0.0, json_mode=False
            )
            email = parse_email_response(raw)
            scores = await evaluate_email(
                email.subject,
                email.body,
                scenario.intent,
                scenario.key_facts,
                scenario.tone,
                self._client,
                provider,
                with_judge=True,
                with_reference=False,
            )
            overall = float(scores["overall"])
            print("attempt", attempt, "overall:", overall)  # debug

            record = AttemptRecord(
                attempt=attempt,
                subject=email.subject,
                body=email.body,
                reasoning=email.reasoning,
                scores=scores,
            )
            trace.append(record)
            log.info(
                "agent.attempt",
                scenario=scenario.id,
                attempt=attempt,
                overall=overall,
                provider=provider,
            )

            if overall > best_overall:
                best_overall = overall
                best = record

            if overall >= threshold:
                break
            feedback = build_critique(scores)

        assert best is not None
        return AgentResult(
            subject=best.subject,
            body=best.body,
            reasoning=best.reasoning,
            scores=best.scores,
            overall=best_overall,
            attempts=len(trace),
            passed=best_overall >= threshold,
            trace=trace,
        )
