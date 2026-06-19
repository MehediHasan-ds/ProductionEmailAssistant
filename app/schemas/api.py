"""API request and response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    intent: str
    facts: list[str] = Field(default_factory=list)
    tone: str = "professional"
    provider: str | None = None
    max_attempts: int | None = None
    threshold: float | None = None
    reference: str | None = None


class EvaluateRequest(BaseModel):
    subject: str
    body: str
    intent: str
    facts: list[str] = Field(default_factory=list)
    tone: str = "professional"
    reference: str | None = None
    provider: str | None = None


class EvaluationResponse(BaseModel):
    rule: dict[str, float]
    judge: dict[str, float] | None = None
    reference: dict[str, float] | None = None
    overall: float


class HealthResponse(BaseModel):
    status: str


class ProviderInfo(BaseModel):
    name: str
    model: str


class ProvidersResponse(BaseModel):
    default: str
    providers: list[ProviderInfo]


class RunEvalsRequest(BaseModel):
    providers: list[str] | None = None
    scenario_ids: list[str] | None = None


class RunEvalsResponse(BaseModel):
    aggregates: dict[str, dict[str, float]]
    scenario_count: int
    report_md: str
    report_csv: str
