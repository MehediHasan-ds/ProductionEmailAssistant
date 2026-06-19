"""FastAPI dependency providers. Heavy clients are created in the lifespan."""
from __future__ import annotations

from fastapi import Request

from app.agents.email_agent import EmailAgent
from app.config import Settings
from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_llm_client(request: Request) -> LLMClient:
    return request.app.state.llm_client


def get_agent(request: Request) -> EmailAgent:
    return request.app.state.agent


def get_embedder(request: Request) -> JinaEmbedder:
    return request.app.state.embedder
