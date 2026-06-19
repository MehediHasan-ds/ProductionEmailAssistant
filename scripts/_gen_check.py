"""Throwaway check: generate one scenario email from each provider and parse it."""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.prompts import build_messages, parse_email_response
from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.models.domain import load_scenarios


async def main() -> None:
    configure_logging()
    scenario = load_scenarios()[0]
    client = LLMClient(get_settings())
    messages = build_messages(scenario.intent, scenario.key_facts, scenario.tone)

    for provider in ("openrouter", "gemini"):
        try:
            raw = await client.chat(messages, provider=provider, temperature=0.0, json_mode=True, timeout=60.0)
        except Exception as exc:
            print(provider, "json_mode failed, retrying plain:", str(exc)[:120])
            raw = await client.chat(messages, provider=provider, temperature=0.0, json_mode=False, timeout=60.0)
        email = parse_email_response(raw)
        print("==", provider, "==")
        print("subject:", email.subject)
        print("body:", email.body[:200].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
