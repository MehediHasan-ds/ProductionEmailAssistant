"""Evaluation suite: baseline versus refinement across scenarios and providers.

Reused by the CLI (eval_runner.py) and the /run-evals endpoint. Writes a
markdown and csv report under reports/.
"""
from __future__ import annotations

import csv
import statistics
from datetime import datetime
from pathlib import Path

from app.agents.email_agent import EmailAgent
from app.config import PROVIDERS, Settings, get_settings
from app.core.embeddings import JinaEmbedder
from app.core.llm_client import LLMClient
from app.metrics.evaluator import evaluate_email
from app.models.domain import GeneratedEmail, Scenario, load_scenarios

REPORTS = Path("reports")


async def _full_scores(
    email: GeneratedEmail,
    scenario: Scenario,
    client: LLMClient,
    provider: str,
    embedder: JinaEmbedder,
) -> dict:
    return await evaluate_email(
        email.subject,
        email.body,
        scenario.intent,
        scenario.key_facts,
        scenario.tone,
        client,
        provider,
        embedder=embedder,
        reference_body=scenario.reference_email.body,
        with_judge=True,
        with_reference=True,
    )


async def run_evals(
    client: LLMClient,
    settings: Settings,
    embedder: JinaEmbedder,
    providers: list[str] | None = None,
    scenario_ids: list[str] | None = None,
) -> dict:
    agent = EmailAgent(client, settings)
    scenarios = load_scenarios()
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s.id in wanted]
    providers = providers or list(PROVIDERS)

    rows: list[dict] = []
    for provider in providers:
        for scenario in scenarios:
            baseline = await agent.generate_once(scenario, provider=provider)
            base_scores = await _full_scores(baseline, scenario, client, provider, embedder)

            refined = await agent.run(scenario, provider=provider, max_attempts=settings.max_attempts)
            refined_email = GeneratedEmail(
                reasoning=refined.reasoning, subject=refined.subject, body=refined.body
            )
            refined_scores = await _full_scores(refined_email, scenario, client, provider, embedder)

            rows.append(
                {
                    "scenario": scenario.id,
                    "category": scenario.category,
                    "provider": provider,
                    "baseline_overall": base_scores["overall"],
                    "refined_overall": refined_scores["overall"],
                    "attempts": refined.attempts,
                    "passed": refined.passed,
                    "lift": round(float(refined_scores["overall"]) - float(base_scores["overall"]), 2),
                }
            )

    aggregates = _aggregate(rows, providers)
    write_report(rows, aggregates)
    return {"rows": rows, "aggregates": aggregates}


def _aggregate(rows: list[dict], providers: list[str]) -> dict:
    result = {}
    for provider in providers:
        provider_rows = [r for r in rows if r["provider"] == provider]
        if not provider_rows:
            continue
        result[provider] = {
            "mean_baseline": round(statistics.mean(r["baseline_overall"] for r in provider_rows), 2),
            "mean_refined": round(statistics.mean(r["refined_overall"] for r in provider_rows), 2),
            "mean_lift": round(statistics.mean(r["lift"] for r in provider_rows), 2),
            "pass_rate": round(100 * sum(1 for r in provider_rows if r["passed"]) / len(provider_rows), 1),
            "mean_attempts": round(statistics.mean(r["attempts"] for r in provider_rows), 2),
        }
    return result


def write_report(rows: list[dict], aggregates: dict) -> None:
    REPORTS.mkdir(exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    fieldnames = [
        "scenario", "category", "provider", "baseline_overall",
        "refined_overall", "attempts", "passed", "lift",
    ]
    with (REPORTS / "eval_report.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Evaluation Report", "", f"Generated: {timestamp}", "", "## Aggregate per provider"]
    for provider, agg in aggregates.items():
        lines.append(
            f"- {provider}: baseline {agg['mean_baseline']}, refined {agg['mean_refined']}, "
            f"lift {agg['mean_lift']}, pass rate {agg['pass_rate']}%, "
            f"mean attempts {agg['mean_attempts']}"
        )
    lines.extend(["", "## Scenarios"])
    for row in rows:
        outcome = "passed" if row["passed"] else "below target"
        lines.append(
            f"- {row['provider']} / {row['scenario']} ({row['category']}): "
            f"baseline {row['baseline_overall']}, refined {row['refined_overall']}, "
            f"lift {row['lift']}, attempts {row['attempts']}, {outcome}"
        )
    (REPORTS / "eval_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
