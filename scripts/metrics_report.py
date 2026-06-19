"""Run all 12 scenarios and output a structured CSV with metric definitions,
raw scores per scenario for all 3 custom metric groups, and the model average.

Usage: uv run python scripts/metrics_report.py

When the Gemini API hits a limit, it returns an HTTP 429 RESOURCE_EXHAUSTED error, indicating that a specific quota boundary has been crossed. 
These limits are enforced at the Google Cloud Project level, meaning all API keys within the same project share a single quota pool,
 and limits are measured across four dimensions: RPM (Requests Per Minute), TPM (Tokens Per Minute), RPD (Requests Per Day), and IPM (Images Per Minute). 
That's why I am giving 20 second delay to every hit to the LLM.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

from app.agents.email_agent import EmailAgent
from app.config import PROVIDERS, get_settings
from app.core.embeddings import get_embedder
from app.core.llm_client import LLMClient
from app.core.logging import configure_logging
from app.metrics.evaluator import evaluate_email
from app.models.domain import load_scenarios
DEFINITIONS = [
    {
        "group": "Rule based",
        "definition": "Deterministic signals computed without an LLM, fast and free",
        "logic": (
            "fact_coverage: fraction of key facts whose content words appear in the body. "
            "tone_match: presence of tone specific lexical markers. "
            "structure: subject, greeting, body, and closing all present. "
            "length: word count within a professional band. "
            "readability: Flesch reading ease within a target band. "
            "placeholder_leak: penalty for leftover brackets or TODOs. "
            "hallucination_flag: extra notable content beyond the given facts. "
            "redundancy: trigram diversity ratio."
        ),
    },
    {
        "group": "LLM as judge",
        "definition": "A separate model call grades the email on a rubric, blind to any reference",
        "logic": (
            "tone_fidelity: matches the requested tone. "
            "fact_integration: every key fact woven in, nothing invented. "
            "professionalism: register, grammar, no slang or placeholders. "
            "clarity_coherence: well organized and readable. "
            "intent_alignment: achieves the stated intent. "
            "overall: a single send readiness score. "
            "Each scored 1 to 5, normalized to 0 to 1."
        ),
    },
    {
        "group": "Reference comparison",
        "definition": "Compares the draft against a hand written reference email",
        "logic": (
            "rouge_l: longest common subsequence overlap with the reference. "
            "bleu: n-gram precision against the reference. "
            "cosine: Jina embedding cosine similarity to the reference."
        ),
    },
]

RULE_KEYS = [
    "fact_coverage", "tone_match", "structure", "length",
    "readability", "placeholder_leak", "hallucination_flag", "redundancy",
]
JUDGE_KEYS = [
    "tone_fidelity", "fact_integration", "professionalism",
    "clarity_coherence", "intent_alignment", "overall",
]
REF_KEYS = ["rouge_l", "bleu", "cosine"]


async def run(provider: str, delay: float = 0) -> None:
    configure_logging()
    settings = get_settings()
    client = LLMClient(settings)
    embedder = get_embedder()
    agent = EmailAgent(client, settings)
    scenarios = load_scenarios()

    results: list[dict] = []
    total = len(scenarios)

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{total}] {scenario.id} ({scenario.category})...", flush=True)
        try:
            email = await agent.generate_once(scenario, provider=provider)

            scores = await evaluate_email(
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

            results.append(
                {
                    "scenario": scenario.id,
                    "category": scenario.category,
                    "provider": provider,
                    "rule": scores.get("rule", {}),
                    "judge": scores.get("judge", {}),
                    "reference": scores.get("reference", {}),
                    "overall": scores.get("overall", 0.0),
                }
            )
            print(f"  overall: {scores.get('overall', 0.0)}", flush=True)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            results.append(
                {
                    "scenario": scenario.id,
                    "category": scenario.category,
                    "provider": provider,
                    "rule": {},
                    "judge": {},
                    "reference": {},
                    "overall": 0.0,
                    "error": str(exc),
                }
            )
        if delay and i < total:
            print(f"  waiting {delay}s before next scenario...", flush=True)
            await asyncio.sleep(delay)

    output = Path(f"reports/metrics_report_{provider}.csv")
    write_csv(results, provider, output)
    print(f"\nReport written to {output}", flush=True)


def write_csv(results: list[dict], provider: str, output: Path) -> None:
    output.parent.mkdir(exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)

        writer.writerow(["METRIC DEFINITIONS AND LOGIC"])
        writer.writerow(["Group", "Definition", "Logic"])
        for d in DEFINITIONS:
            writer.writerow([d["group"], d["definition"], d["logic"]])
        writer.writerow([])

        header = ["Scenario", "Category", "Provider"]
        header += [f"Rule_{k}" for k in RULE_KEYS]
        header += [f"Judge_{k}" for k in JUDGE_KEYS]
        header += [f"Ref_{k}" for k in REF_KEYS]
        header += ["Overall", "Error"]
        writer.writerow(header)

        all_keys = RULE_KEYS + JUDGE_KEYS + REF_KEYS
        sums = {k: 0.0 for k in all_keys}
        sums["overall"] = 0.0
        valid = 0

        for r in results:
            row = [r["scenario"], r["category"], r["provider"]]
            for k in RULE_KEYS:
                v = float(r["rule"].get(k, 0))
                sums[k] += v
                row.append(round(v, 4))
            for k in JUDGE_KEYS:
                v = float(r["judge"].get(k, 0))
                sums[k] += v
                row.append(round(v, 4))
            for k in REF_KEYS:
                v = float(r["reference"].get(k, 0))
                sums[k] += v
                row.append(round(v, 4))
            sums["overall"] += float(r["overall"])
            row.append(round(float(r["overall"]), 2))
            row.append(r.get("error", ""))
            if not r.get("error"):
                valid += 1
            writer.writerow(row)

        avg_row = ["AVERAGE", "", provider]
        denom = valid if valid else 1
        for k in all_keys:
            avg_row.append(round(sums[k] / denom, 4))
        avg_row.append(round(sums["overall"] / denom, 2))
        avg_row.append(f"({valid}/{len(results)} succeeded)")
        writer.writerow(avg_row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 12 scenarios and output a metrics CSV")
    parser.add_argument("--provider", default=None, help="Single provider (openrouter or gemini). If omitted, runs both.")
    parser.add_argument("--delay", type=float, default=20.0, help="Seconds to wait between scenarios (default 20)")
    args = parser.parse_args()

    if args.provider:
        asyncio.run(run(args.provider, args.delay))
    else:
        for idx, provider in enumerate(PROVIDERS):
            print(f"\n{'=' * 60}", flush=True)
            print(f"  Provider: {provider}", flush=True)
            print(f"{'=' * 60}\n", flush=True)
            asyncio.run(run(provider, args.delay))
            if idx < len(PROVIDERS) - 1:
                print(f"\nPausing 30s before next provider...", flush=True)
                time.sleep(30)


if __name__ == "__main__":
    main()
