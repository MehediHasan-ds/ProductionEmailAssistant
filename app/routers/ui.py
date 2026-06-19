"""Serves the single page interface."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

router = APIRouter()

_INDEX = "static/index.html"
# Prevent the browser from caching the page during development.
_NO_CACHE = {"cache-control": "no-cache, no-store, must-revalidate"}


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_INDEX, headers=_NO_CACHE)
