"""System endpoints: health check and provider listing."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_settings

router = APIRouter()

PROVIDERS = ["openrouter", "gemini"]


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/providers")
def providers(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    return {"default": settings.default_provider, "providers": PROVIDERS}
