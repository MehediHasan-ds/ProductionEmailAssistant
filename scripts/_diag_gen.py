"""Throwaway diagnostic: compare json_mode on vs off for the generate path."""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.agents.prompts import build_messages
from app.config import get_settings
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging


async def main() -> None:
    configure_logging()
    client = LLMClient(get_settings())
    msgs = build_messages(
        "Follow up after a product demo",
        ["Demo with the team on Monday", "Sending pricing next"],
        "professional",
    )
    with_json = await client.chat(msgs, provider="openrouter", json_mode=True)
    print("json_mode length:", len(with_json))
    print("json_mode head:", repr(with_json[:120]))
    plain = await client.chat(msgs, provider="openrouter", json_mode=False)
    print("plain length:", len(plain))
    print("plain head:", repr(plain[:200]))


if __name__ == "__main__":
    asyncio.run(main())
