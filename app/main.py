"""Application factory. The lifespan owns startup and shutdown of clients."""
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from structlog.contextvars import bind_contextvars, clear_contextvars

from app.agents.email_agent import EmailAgent
from app.config import get_settings
from app.core.embeddings import JinaEmbedder
from app.core.exceptions import register_exception_handlers
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.routers import evals, evaluate, generate, system, ui


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
    await app.state.llm_client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Production Email Assistant", version="0.1.0", lifespan=lifespan)
    register_exception_handlers(app)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            clear_contextvars()
        response.headers["x-request-id"] = request_id
        return response

    app.include_router(system.router, tags=["system"])
    app.include_router(generate.router, tags=["generate"])
    app.include_router(evaluate.router, tags=["evaluate"])
    app.include_router(evals.router, tags=["evals"])
    app.include_router(ui.router)
    return app


app = create_app()
