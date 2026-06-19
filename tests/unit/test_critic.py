"""Unit tests for the critique builder."""
from app.agents.critic import build_critique


def test_critique_lists_weakest_known_dimensions():
    scores = {
        "rule": {"fact_coverage": 0.3, "structure": 1.0, "tone_match": 0.9},
        "judge": {"fact_integration": 0.4, "overall": 0.5},
        "overall": 55.0,
    }
    critique = build_critique(scores)
    assert "fact_coverage" in critique
    assert "fact_integration" in critique
    assert "structure" not in critique


def test_critique_fallback_when_no_known_metrics():
    scores = {"rule": {"weird_metric": 0.1}, "overall": 10.0}
    critique = build_critique(scores)
    assert critique
