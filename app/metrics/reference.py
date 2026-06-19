"""Reference comparison metrics against the human reference email.

ROUGE-L and BLEU measure lexical overlap. Jina cosine measures semantic
closeness via the local embedder. The embedder is optional so these metrics
can be unit tested without loading the ONNX model.
"""
from __future__ import annotations

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

from app.core.embeddings import JinaEmbedder

_SMOOTH = SmoothingFunction().method1
_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def reference_metrics(
    candidate: str,
    reference: str,
    embedder: JinaEmbedder | None = None,
) -> dict[str, float]:
    result: dict[str, float] = {
        "rouge_l": _rouge_l(candidate, reference),
        "bleu": _bleu(candidate, reference),
    }
    if embedder is not None:
        result["cosine"] = embedder.cosine(candidate, reference)
    return result


def _rouge_l(candidate: str, reference: str) -> float:
    return _SCORER.score(reference, candidate)["rougeL"].fmeasure


def _bleu(candidate: str, reference: str) -> float:
    return sentence_bleu(
        [reference.split()],
        candidate.split(),
        smoothing_function=_SMOOTH,
    )
