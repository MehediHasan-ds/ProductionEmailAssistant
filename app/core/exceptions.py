"""Domain error base and FastAPI exception handlers."""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)


class AppError(Exception):
    status_code: int = 500

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _on_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.message})

    @app.exception_handler(Exception)
    async def _on_unexpected(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled.exception", error=str(exc))
        return JSONResponse(status_code=500, content={"error": "internal server error"})
