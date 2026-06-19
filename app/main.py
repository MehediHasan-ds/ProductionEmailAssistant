"""Application factory. The lifespan owns startup and shutdown of clients."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.agents.email_agent import EmailAgent
from app.config import get_settings
from app.core.embeddings import JinaEmbedder
from app.core.exceptions import register_exception_handlers
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.routers import system


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging()
    app.state.settings = settings
    app.state.llm_client = LLMClient(settings)
    app.state.agent = EmailAgent(app.state.llm_client, settings)
    # Loaded lazily on first use, since the ONNX warm up is slow.
    app.state.embedder = JinaEmbedder(settings)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Production Email Assistant", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)
    app.include_router(system.router, tags=["system"])
    return app


app = create_app()
