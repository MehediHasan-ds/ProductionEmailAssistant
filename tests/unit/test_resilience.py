"""Resilience test: a transient provider failure is retried then surfaced as a
domain error rather than crashing the request."""
import asyncio

import httpx
import pytest
from openai import APIConnectionError

from app.config import Settings
from app.core.exceptions import AppError
from app.core.llm_client import LLMClient


class _ThrowingOpenAI:
    """Stand-in for AsyncOpenAI whose chat completion always raises."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    async def create(self, **_):
        raise self._exc


def test_transient_failure_converted_to_app_error():
    settings = Settings(llm_retry_attempts=1)
    client = LLMClient(settings)
    client._openai_clients["openrouter"] = _ThrowingOpenAI(
        APIConnectionError(request=httpx.Request("POST", "https://openrouter.ai/api/v1"))
    )

    with pytest.raises(AppError):
        asyncio.run(
            client.chat([{"role": "user", "content": "hi"}], provider="openrouter")
        )
