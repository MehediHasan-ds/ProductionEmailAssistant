"""Throwaway check: judge a reference email from each provider."""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.metrics.judge import judge_email
from app.models.domain import load_scenarios


async def main() -> None:
    configure_logging()
    scenario = load_scenarios()[0]
    ref = scenario.reference_email
    client = LLMClient(get_settings())
    for provider in ("openrouter", "gemini"):
        scores = await judge_email(
            ref.subject, ref.body, scenario.intent, scenario.key_facts, scenario.tone, client, provider
        )
        mean = round(sum(scores.values()) / len(scores), 3)
        print("==", provider, "== mean", mean)
        for k, v in scores.items():
            print(" ", k, round(v, 3))


if __name__ == "__main__":
    asyncio.run(main())
