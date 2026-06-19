"""Domain models for evaluation scenarios and generated emails."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ReferenceEmail(BaseModel):
    subject: str
    body: str


class Scenario(BaseModel):
    id: str
    category: str
    intent: str
    key_facts: list[str]
    tone: str
    reference_email: ReferenceEmail
    notes: str = ""


class GeneratedEmail(BaseModel):
    reasoning: str = ""
    subject: str
    body: str


class AttemptRecord(BaseModel):
    attempt: int
    subject: str
    body: str
    reasoning: str = ""
    scores: dict[str, Any]


class AgentResult(BaseModel):
    subject: str
    body: str
    reasoning: str = ""
    scores: dict[str, Any]
    overall: float
    attempts: int
    passed: bool
    trace: list[AttemptRecord]


DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "scenarios.json"


@lru_cache(maxsize=1)
def load_scenarios() -> list[Scenario]:
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return [Scenario(**item) for item in raw["scenarios"]]


def get_scenario(scenario_id: str) -> Scenario:
    for scenario in load_scenarios():
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(scenario_id)
