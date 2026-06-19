"""CLI entrypoint for the evaluation suite. Run with: uv run python eval_runner.py"""
from __future__ import annotations

import asyncio
import json

from app.config import get_settings
from app.core.embeddings import get_embedder
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.services.eval_service import run_evals


def main() -> None:
    configure_logging()
    settings = get_settings()
    client = LLMClient(settings)
    embedder = get_embedder()
    result = asyncio.run(run_evals(client, settings, embedder))
    print(json.dumps(result["aggregates"], indent=2))
    print("done!")
    print("report written to reports/eval_report.md and reports/eval_report.csv")


if __name__ == "__main__":
    main()
