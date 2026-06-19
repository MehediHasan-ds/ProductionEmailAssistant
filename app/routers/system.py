"""System endpoints: health check and provider listing."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_settings
from app.schemas.api import HealthResponse, ProviderInfo, ProvidersResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/providers", response_model=ProvidersResponse)
def providers(settings: Settings = Depends(get_settings)) -> ProvidersResponse:
    info = [
        ProviderInfo(name="openrouter", model=settings.openrouter_model),
        ProviderInfo(name="gemini", model=settings.gemini_model),
    ]
    return ProvidersResponse(default=settings.default_provider, providers=info)
