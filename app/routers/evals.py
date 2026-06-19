"""Run evals endpoint: runs the evaluation suite and points to the report."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import Settings
from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient
from app.dependencies import get_embedder, get_llm_client, get_settings
from app.schemas.api import RunEvalsRequest, RunEvalsResponse
from app.services.eval_service import run_evals

router = APIRouter()


@router.post("/run-evals", response_model=RunEvalsResponse)
async def run_evals_endpoint(
    request: RunEvalsRequest,
    client: LLMClient = Depends(get_llm_client),
    embedder: JinaEmbedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> RunEvalsResponse:
    result = await run_evals(
        client,
        settings,
        embedder,
        providers=request.providers,
        scenario_ids=request.scenario_ids,
    )
    return RunEvalsResponse(
        aggregates=result["aggregates"],
        scenario_count=len(result["rows"]),
        report_md="reports/eval_report.md",
        report_csv="reports/eval_report.csv",
    )
