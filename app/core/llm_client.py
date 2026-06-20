"""Provider-agnostic chat client.

OpenRouter runs on the OpenAI compatible API. Gemini runs on the native
generateContent endpoint with the x-goog-api-key header, which is the auth
method the configured key supports. Callers see one uniform chat() method.

Retry counts, backoff, and timeouts come from settings, not literals. The
Gemini HTTP client is created once and reused rather than per call.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings
from app.core.exceptions import AppError

log = structlog.get_logger(__name__)

_OPENAI_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError)
_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
_ROLE_MAP = {"user": "user", "assistant": "model", "system": "user"}

_PY_TYPE_MAP = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}


def _pydantic_to_gemini_schema(schema: dict) -> dict:
    props = schema.get("properties", {})
    required = schema.get("required", [])
    gemini_props = {}
    for key, val in props.items():
        py_type = val.get("type", "string")
        gemini_props[key] = {"type": _PY_TYPE_MAP.get(py_type, "STRING")}
    return {
        "type": "OBJECT",
        "properties": gemini_props,
        "required": required,
    }


class _RetryableGemini(Exception):
    """Raised for transient Gemini failures that should be retried."""


def _retry_logger(event: str):
    def _callback(retry_state) -> None:
        outcome = retry_state.outcome
        error = str(outcome.exception()) if outcome is not None else None
        log.warning(event, attempt=retry_state.attempt_number, error=error)

    return _callback


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._openai_clients: dict[str, AsyncOpenAI] = {}
        self._http = httpx.AsyncClient(timeout=settings.llm_timeout)

        attempts = settings.llm_retry_attempts
        max_wait = settings.llm_retry_max_wait
        self._openai_retry = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=2, max=max_wait),
            retry=retry_if_exception_type(_OPENAI_TRANSIENT),
            before_sleep=_retry_logger("llm.retry.openai"),
            reraise=True,
        )
        self._gemini_retry = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=5, max=max_wait),
            retry=retry_if_exception_type(_RetryableGemini),
            before_sleep=_retry_logger("llm.retry.gemini"),
            reraise=True,
        )
        self._gemini_delay = settings.gemini_rate_limit_delay

    async def aclose(self) -> None:
        await self._http.aclose()
        for client in self._openai_clients.values():
            await client.close()
        self._openai_clients.clear()

    async def chat(
        self,
        messages: list[dict[str, str]],
        provider: str | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> str:
        provider = provider or self._settings.default_provider
        resolved_timeout = timeout or self._settings.llm_timeout
        if provider == "gemini":
            return await self._chat_gemini(messages, temperature, json_mode)
        return await self._chat_openai(provider, messages, temperature, json_mode, resolved_timeout)

    async def chat_structured(
        self,
        messages: list[dict[str, str]],
        output_model: type,
        provider: str | None = None,
        temperature: float = 0.0,
        timeout: float | None = None,
    ):
        """Returns a validated pydantic instance. Retries once if parse fails."""
        from pydantic import ValidationError

        provider = provider or self._settings.default_provider
        resolved_timeout = timeout or self._settings.llm_timeout

        for attempt in range(2):
            if provider == "gemini":
                raw = await self._chat_gemini(messages, temperature, True)
            else:
                raw = await self._chat_openai(provider, messages, temperature, False, resolved_timeout)

            try:
                from app.core.text import extract_json
                import json
                payload = json.loads(extract_json(raw))
                return output_model.model_validate(payload)
            except (json.JSONDecodeError, ValidationError, Exception) as exc:
                if attempt == 0:
                    log.warning("structured.parse_failed_retrying", error=str(exc)[:100])
                    messages = messages + [
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content": f"Your response could not be parsed as valid JSON matching the schema. Return ONLY a JSON object with these keys: {list(output_model.model_json_schema().get('properties', {}).keys())}. Example: {output_model.model_json_schema().get('examples', [''])[0] if output_model.model_json_schema().get('examples') else ''}"},
                    ]
                else:
                    log.error("structured.parse_failed_final", error=str(exc)[:100])
                    raise

    async def _chat_gemini_structured(
        self,
        messages: list[dict[str, str]],
        output_model: type,
        temperature: float,
    ) -> str:
        import json
        model = self._settings.gemini_model
        api_key = self._settings.gemini_api_key.get_secret_value()
        system_text = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]

        schema = output_model.model_json_schema()
        gemini_schema = _pydantic_to_gemini_schema(schema)

        body = {
            "contents": self._to_contents(turns),
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
                "responseSchema": gemini_schema,
            },
        }
        if system_text:
            body["system_instruction"] = {"parts": [{"text": system_text}]}

        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        log.info("llm.chat.start", provider="gemini", model=model, structured=True)

        async def _once() -> httpx.Response:
            try:
                response = await self._http.post(url, headers={"x-goog-api-key": api_key}, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise _RetryableGemini(str(exc)) from exc
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise _RetryableGemini(f"gemini status {response.status_code}")
            return response

        try:
            resp = None
            async for attempt in self._gemini_retry:
                with attempt:
                    resp = await _once()
        except _RetryableGemini as exc:
            raise AppError(f"Gemini call failed after retries: {exc}", status_code=502) from exc

        if resp is None or resp.status_code >= 400:
            raise AppError(f"Gemini call failed with status {getattr(resp, 'status_code', None)}", status_code=502)

        text = self._extract_text(resp.json())
        log.info("llm.chat.done", provider="gemini", model=model, chars=len(text))
        await asyncio.sleep(self._gemini_delay)
        return text

    async def _chat_openai(
        self,
        provider: str,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
        timeout: float,
    ) -> str:
        client, model = self._openai_client_for(provider)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        print("--- calling provider ---")  # TODO: remove this
        log.info("llm.chat.start", provider=provider, model=model, json_mode=json_mode)
        resp = None
        try:
            async for attempt in self._openai_retry:
                with attempt:
                    resp = await client.chat.completions.create(**kwargs)
        except _OPENAI_TRANSIENT as exc:
            raise AppError(f"{provider} call failed after retries: {exc}", status_code=502) from exc

        content = resp.choices[0].message.content if resp else ""
        content = content or ""
        log.info("llm.chat.done", provider=provider, model=model, chars=len(content))
        return content

    async def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        json_mode: bool,
    ) -> str:
        model = self._settings.gemini_model
        api_key = self._settings.gemini_api_key.get_secret_value()
        system_text = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        turns = [m for m in messages if m.get("role") != "system"]

        body: dict[str, Any] = {"contents": self._to_contents(turns)}
        generation_config: dict[str, Any] = {"temperature": temperature}
        if system_text:
            # The native API requires a Content object, not a raw string.
            body["system_instruction"] = {"parts": [{"text": system_text}]}
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
        body["generationConfig"] = generation_config

        url = f"{_GEMINI_BASE}/models/{model}:generateContent"
        print("--- calling provider ---")  # TODO: remove this
        log.info("llm.chat.start", provider="gemini", model=model, json_mode=json_mode)

        async def _once() -> httpx.Response:
            try:
                response = await self._http.post(url, headers={"x-goog-api-key": api_key}, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                raise _RetryableGemini(str(exc)) from exc
            if response.status_code == 429 or 500 <= response.status_code < 600:
                raise _RetryableGemini(f"gemini status {response.status_code}")
            return response

        try:
            resp = None
            async for attempt in self._gemini_retry:
                with attempt:
                    resp = await _once()
        except _RetryableGemini as exc:
            raise AppError(f"Gemini call failed after retries: {exc}", status_code=502) from exc

        if resp is None or resp.status_code >= 400:
            status = getattr(resp, "status_code", None)
            log.error("llm.chat.gemini.error", status=status, body=getattr(resp, "text", "")[:300])
            raise AppError(f"Gemini call failed with status {status}", status_code=502)

        text = self._extract_text(resp.json())
        log.info("llm.chat.done", provider="gemini", model=model, chars=len(text))
        await asyncio.sleep(self._gemini_delay)
        return text

    def _openai_client_for(self, provider: str) -> tuple[AsyncOpenAI, str]:
        if provider != "openrouter":
            raise ValueError(f"Provider {provider!r} is not on the OpenAI compatible API")
        client = self._openai_clients.get(provider)
        if client is None:
            client = AsyncOpenAI(
                base_url=self._settings.openrouter_base_url,
                api_key=self._settings.openrouter_api_key.get_secret_value(),
            )
            self._openai_clients[provider] = client
        return client, self._settings.openrouter_model

    @staticmethod
    def _to_contents(turns: list[dict[str, str]]) -> list[dict[str, Any]]:
        contents: list[dict[str, Any]] = []
        for turn in turns:
            role = _ROLE_MAP.get(turn.get("role", "user"), "user")
            contents.append({"role": role, "parts": [{"text": turn["content"]}]})
        return contents

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            return payload["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return ""
