"""Milestone 1 smoke check: confirm both providers answer a trivial prompt."""
import asyncio

import structlog

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging


async def main() -> None:
    configure_logging()
    log = structlog.get_logger("m1.smoke")
    settings = get_settings()
    client = LLMClient(settings)

    prompt = [{"role": "user", "content": "what can you do?"}]
    for provider in ("openrouter", "gemini"):
        reply = await client.chat(prompt, provider=provider, temperature=0.0, timeout=30.0)
        reply = (reply or "").strip()
        assert reply, f"{provider} returned an empty response"
        log.info("smoke.chat.ok", provider=provider, reply=reply)


if __name__ == "__main__":
    asyncio.run(main())
