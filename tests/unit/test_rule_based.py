"""Unit tests for the rule-based metrics. No network, no LLM, fast."""
from app.metrics.rule_based import rule_based_metrics
from app.models.domain import load_scenarios

EXPECTED_KEYS = {
    "fact_coverage", "tone_match", "structure", "length",
    "readability", "placeholder_leak", "hallucination_flag", "redundancy",
}

FACTS = [
    "Shipment reference VRD-8842 contained insulin",
    "Delayed five days due to a customs hold at Rotterdam",
]

GOOD_BODY = (
    "Hello,\n\nI am very sorry about shipment VRD-8842, the insulin. "
    "It was delayed five days due to a customs hold at Rotterdam. "
    "A replacement will arrive Friday, and I have applied a 15 percent credit "
    "to your next order.\n\nThank you for your patience.\n\nWith sincere apologies,"
)


def test_returns_all_metric_keys():
    result = rule_based_metrics("Subject", GOOD_BODY, FACTS, "empathetic")
    assert set(result) == EXPECTED_KEYS


def test_fact_coverage_rewards_present_facts():
    result = rule_based_metrics("Subject", GOOD_BODY, FACTS, "empathetic")
    assert result["fact_coverage"] >= 0.99


def test_fact_coverage_penalizes_missing_facts():
    body = "Hello,\n\nNothing relevant here at all.\n\nBest regards,"
    result = rule_based_metrics("Subject", body, FACTS, "empathetic")
    assert result["fact_coverage"] <= 0.01


def test_structure_complete_scores_full():
    result = rule_based_metrics("A clear subject", GOOD_BODY, FACTS, "empathetic")
    assert result["structure"] == 1.0


def test_structure_penalizes_missing_greeting_and_closing():
    body = "This body has no greeting or sign off and just rambles on about nothing " * 4
    result = rule_based_metrics("Subject", body, [], "professional")
    assert result["structure"] <= 0.5


def test_placeholder_leak_flags_brackets_and_todo():
    leaky = "Hello,\n\nPlease fill in [your name] and the TODO item.\n\nBest regards,"
    assert rule_based_metrics("S", leaky, [], "professional")["placeholder_leak"] == 0.0


def test_placeholder_leak_clean_passes():
    assert rule_based_metrics("S", GOOD_BODY, FACTS, "empathetic")["placeholder_leak"] == 1.0


def test_length_ideal_band_scores_full():
    scenarios = load_scenarios()
    body = scenarios[0].reference_email.body
    result = rule_based_metrics("S", body, [], "professional")
    assert result["length"] >= 0.9


def test_length_too_short_scores_low():
    result = rule_based_metrics("S", "Hi, ok thanks.", [], "professional")
    assert result["length"] <= 0.1


def test_redundancy_flags_repetition():
    repetitive = "Please review the attached document. Please review the attached document. Please review the attached document."
    result = rule_based_metrics("S", repetitive, [], "professional")
    assert result["redundancy"] < 0.6


def test_tone_match_empathetic_markers_present():
    result = rule_based_metrics("S", GOOD_BODY, FACTS, "empathetic")
    assert result["tone_match"] >= 0.66


def test_reference_emails_are_well_formed():
    scenarios = load_scenarios()
    excluded = {"failure_missing_facts", "failure_contradictory_facts"}
    for scenario in scenarios:
        if scenario.id in excluded:
            continue
        ref = scenario.reference_email
        result = rule_based_metrics(ref.subject, ref.body, scenario.key_facts, scenario.tone)
        assert result["structure"] >= 0.75, scenario.id
        assert result["placeholder_leak"] == 1.0, scenario.id
