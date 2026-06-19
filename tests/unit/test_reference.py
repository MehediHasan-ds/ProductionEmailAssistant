"""Unit tests for reference metrics. Embedder is omitted so no model loads."""
from app.metrics.reference import reference_metrics

REFERENCE = (
    "Hello, thank you for the meeting on Tuesday about the contract. "
    "The next step is to send the proposal this week. Best regards,"
)


def test_returns_lexical_metrics_without_embedder():
    result = reference_metrics(REFERENCE, REFERENCE)
    assert set(result) == {"rouge_l", "bleu"}


def test_identical_text_scores_high():
    result = reference_metrics(REFERENCE, REFERENCE)
    assert result["rouge_l"] >= 0.99
    assert result["bleu"] >= 0.99


def test_unrelated_text_scores_low():
    candidate = "Bananas are on sale at the market this weekend, come grab a bunch."
    result = reference_metrics(candidate, REFERENCE)
    assert result["rouge_l"] < 0.2
    assert result["bleu"] < 0.2


def test_partial_overlap_scores_between():
    candidate = (
        "Hello, thanks for the Tuesday meeting about the contract. "
        "I will send the proposal shortly."
    )
    result = reference_metrics(candidate, REFERENCE)
    assert 0.2 < result["rouge_l"] < 0.95
