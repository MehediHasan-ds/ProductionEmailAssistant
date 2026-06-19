"""Evaluate endpoint: scores one provided email without retry."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings
from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient
from app.dependencies import get_embedder, get_llm_client, get_settings
from app.schemas.api import EvaluateRequest, EvaluationResponse
from app.services.evaluation_service import EvaluationService

router = APIRouter()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(
    request: EvaluateRequest,
    client: LLMClient = Depends(get_llm_client),
    embedder: JinaEmbedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> EvaluationResponse:
    return await EvaluationService().evaluate(request, client, embedder, settings.default_provider)
