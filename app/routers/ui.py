"""Serves the single page interface."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_INDEX = "static/index.html"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX)
