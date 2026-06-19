"""Generate endpoint: runs the full refinement agent."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.agents.email_agent import EmailAgent
from app.config import Settings
from app.dependencies import get_agent, get_settings
from app.models.domain import AgentResult
from app.schemas.api import GenerateRequest
from app.services.generation_service import GenerationService

router = APIRouter()


@router.post("/generate", response_model=AgentResult)
async def generate(
    request: GenerateRequest,
    agent: EmailAgent = Depends(get_agent),
    settings: Settings = Depends(get_settings),
) -> AgentResult:
    return await GenerationService(settings).generate(request, agent)
