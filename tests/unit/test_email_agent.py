"""Unit tests for the EmailAgent refinement loop, using a fake LLM client."""
import asyncio
import json

import pytest

from app.agents.email_agent import EmailAgent
from app.config import Settings
from app.models.domain import GeneratedEmail, load_scenarios

LOW_JUDGE = {"tone_fidelity": 2, "fact_integration": 2, "professionalism": 2,
             "clarity_coherence": 2, "intent_alignment": 2, "overall": 2}
HIGH_JUDGE = {"tone_fidelity": 5, "fact_integration": 5, "professionalism": 5,
              "clarity_coherence": 5, "intent_alignment": 5, "overall": 5}


class FakeClient:
    def __init__(self, scenario, always_low=False):
        self.scenario = scenario
        self.always_low = always_low
        self.judge_calls = 0

    async def chat_structured(self, messages, output_model, provider=None, temperature=0.0, timeout=30.0):
        system = messages[0]["content"]
        if "strict editor grading" in system:
            self.judge_calls += 1
            data = LOW_JUDGE if (self.always_low or self.judge_calls == 1) else HIGH_JUDGE
        else:
            ref = self.scenario.reference_email
            data = {"reasoning": "plan", "subject": ref.subject, "body": ref.body}
        return output_model.model_validate(data)

    async def chat(self, messages, provider=None, temperature=0.0, json_mode=False, timeout=30.0):
        return json.dumps({"reasoning": "plan", "subject": "test", "body": "test body"})


def _settings(max_attempts, threshold):
    # Settings loads the real values from .env; only these two are overridden.
    return Settings(max_attempts=max_attempts, pass_threshold=threshold)


def test_loop_stops_when_threshold_met():
    scenario = load_scenarios()[0]
    agent = EmailAgent(FakeClient(scenario), _settings(max_attempts=3, threshold=80))
    result = asyncio.run(agent.run(scenario))
    assert result.attempts == 2
    assert result.passed is True
    assert len(result.trace) == 2
    assert result.subject == scenario.reference_email.subject


def test_loop_caps_at_max_when_below_threshold():
    scenario = load_scenarios()[0]
    agent = EmailAgent(FakeClient(scenario, always_low=True), _settings(max_attempts=3, threshold=80))
    result = asyncio.run(agent.run(scenario))
    assert result.attempts == 3
    assert result.passed is False
    assert len(result.trace) == 3


def test_loop_keeps_best_draft():
    scenario = load_scenarios()[0]
    agent = EmailAgent(FakeClient(scenario), _settings(max_attempts=3, threshold=80))
    result = asyncio.run(agent.run(scenario))
    # attempt 2 scored higher, so the returned draft is from attempt 2
    best_attempt = max(result.trace, key=lambda r: float(r.scores["overall"]))
    assert result.overall == float(best_attempt.scores["overall"])
