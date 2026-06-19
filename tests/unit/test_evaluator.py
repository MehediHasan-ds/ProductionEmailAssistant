"""Unit tests for the weighted overall aggregation."""
from app.metrics.evaluator import weighted_overall


def test_perfect_scores_reach_one_hundred():
    groups = {
        "rule": {"a": 1.0},
        "judge": {"b": 1.0},
        "reference": {"c": 1.0},
    }
    assert weighted_overall(groups) == 100.0


def test_all_three_groups_weighted():
    groups = {
        "rule": {"a": 0.5},
        "judge": {"b": 1.0},
        "reference": {"c": 0.0},
    }
    assert weighted_overall(groups) == 55.0


def test_missing_reference_renormalizes_over_rule_and_judge():
    groups = {
        "rule": {"a": 0.5},
        "judge": {"b": 1.0},
    }
    assert weighted_overall(groups) == round((0.3 * 0.5 + 0.4 * 1.0) / 0.7 * 100, 2)


def test_only_rule_uses_rule_mean():
    groups = {"rule": {"a": 0.8, "b": 0.8}}
    assert weighted_overall(groups) == 80.0


def test_empty_groups_return_zero():
    assert weighted_overall({}) == 0.0
