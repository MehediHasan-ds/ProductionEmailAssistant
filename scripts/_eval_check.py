"""Throwaway check: run the eval suite on one scenario and one provider."""
import asyncio
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings
from app.core.embeddings import get_embedder
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.services.eval_service import run_evals


async def main() -> None:
    configure_logging()
    settings = get_settings()
    client = LLMClient(settings)
    embedder = get_embedder()
    result = await run_evals(
        client,
        settings,
        embedder,
        providers=["openrouter"],
        scenario_ids=["normal_followup"],
    )
    print(json.dumps(result["aggregates"], indent=2))
    for row in result["rows"]:
        print(row)


asyncio.run(main())
