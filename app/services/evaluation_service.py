"""Owns the one-shot evaluate flow: score a provided email."""
from __future__ import annotations

from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient
from app.metrics.evaluator import evaluate_email
from app.schemas.api import EvaluateRequest, EvaluationResponse


class EvaluationService:
    async def evaluate(
        self,
        request: EvaluateRequest,
        client: LLMClient,
        embedder: JinaEmbedder | None,
        default_provider: str,
    ) -> EvaluationResponse:
        provider = request.provider or default_provider
        with_reference = request.reference is not None and embedder is not None
        scores = await evaluate_email(
            request.subject,
            request.body,
            request.intent,
            request.facts,
            request.tone,
            client,
            provider,
            embedder=embedder,
            reference_body=request.reference,
            with_judge=True,
            with_reference=with_reference,
        )
        return EvaluationResponse(
            rule=scores["rule"],
            judge=scores.get("judge"),
            reference=scores.get("reference"),
            overall=scores["overall"],
        )
