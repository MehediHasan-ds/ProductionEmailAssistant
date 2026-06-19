"""Deterministic, dependency-light metrics. No LLM calls.

Each metric returns a float in [0.0, 1.0]. These are coarse signals, meant to
be combined with the LLM judge and reference comparison, not used alone.
"""
from __future__ import annotations

import re
import string

import textstat

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "for", "with",
    "at", "by", "from", "is", "are", "was", "were", "be", "been", "being", "as",
    "it", "its", "this", "that", "these", "those", "we", "you", "your", "our",
    "their", "his", "her", "i", "me", "my", "will", "would", "should", "can",
    "could", "have", "has", "had", "do", "does", "did", "if", "then", "than",
    "so", "not", "no", "into", "about", "over", "under", "up", "out",
}

COMMON_EMAIL_WORDS = {
    "hello", "hi", "hey", "dear", "thanks", "thank", "please", "regards", "cheers",
    "best", "warm", "sincerely", "team", "let", "know", "feel", "free", "reach",
    "reply", "email", "meeting", "send", "sent", "attached", "discuss", "discussed",
    "kindly", "looking", "forward", "hearing", "soon", "time", "week", "month",
    "next", "step", "review", "happy", "glad", "sorry", "apologize", "apologies",
    "update", "status", "completed", "done", "scheduled", "agreed", "confirm",
}

TONE_LEXICONS: dict[str, list[str]] = {
    "empathetic": ["sorry", "apolog", "understand", "patience", "inconvenience", "frustrat", "sincere"],
    "firm": ["overdue", "late fee", "must", "required", "prompt", "attention", "outstanding", "clause"],
    "courteous": ["please", "kindly", "thank", "appreciate", "regards"],
    "casual": ["hey", "cheers", "grab", "fun", "ready", "folks"],
    "energetic": ["exciting", "ready", "get ready", "cant wait", "lets", "stakes"],
    "diplomatic": ["unfortunately", "honored", "regret", "appreciate", "pleased", "respectfully"],
    "warm": ["welcome", "excited", "glad", "delighted", "reach out", "aboard"],
    "professional": ["please", "regards", "thank", "review", "discuss", "meeting"],
}

GREETINGS = ("hi", "hello", "hey", "dear", "greetings")
CLOSINGS = ("regards", "thanks", "cheers", "sincerely", "best", "apologies", "warm")

LEAK_PATTERNS = (
    r"\[[^\]]*\]",
    r"\{[^\}]*\}",
    r"<[^>]*>",
    r"\btodo\b",
    r"\binsert\b",
    r"your name",
    r"xxx",
)


# TODO: these thresholds need tuning
def rule_based_metrics(
    subject: str, body: str, facts: list[str], tone: str
) -> dict[str, float]:
    return {
        "fact_coverage": _fact_coverage(body, facts),
        "tone_match": _tone_match(body, tone),
        "structure": _structure(subject, body),
        "length": _length(body),
        "readability": _readability(body),
        "placeholder_leak": _placeholder_leak(body),
        "hallucination_flag": _hallucination(body, facts),
        "redundancy": _redundancy(body),
    }


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _content_words(text: str) -> list[str]:
    return [w for w in _normalize(text).split() if w not in STOPWORDS and len(w) > 2]


def _fact_coverage(body: str, facts: list[str]) -> float:
    if not facts:
        return 1.0
    body_norm = _normalize(body)
    covered = 0
    for fact in facts:
        words = [w for w in _content_words(fact) if len(w) > 3]
        if not words:
            words = _normalize(fact).split()
        if not words:
            covered += 1
            continue
        hits = sum(1 for w in words if w in body_norm)
        if hits / len(words) >= 0.5:
            covered += 1
    return covered / len(facts)


def _tone_match(body: str, tone: str) -> float:
    tone_key = tone.lower()
    markers: set[str] = set()
    for key, words in TONE_LEXICONS.items():
        if key in tone_key:
            markers.update(words)
    if not markers:
        markers = set(TONE_LEXICONS["professional"])
    body_norm = _normalize(body)
    hits = sum(1 for m in markers if _phrase_in(m, body_norm))
    return min(1.0, hits / 3.0)


def _phrase_in(phrase: str, text: str) -> bool:
    return _normalize(phrase) in text


def _structure(subject: str, body: str) -> float:
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    has_greeting = False
    if lines and lines[0].split():
        first_token = lines[0].split()[0].lower().strip(string.punctuation)
        has_greeting = first_token in GREETINGS
    has_closing = False
    for ln in lines:
        toks = [t.lower().strip(string.punctuation) for t in ln.split()]
        if len(toks) <= 3 and any(t in CLOSINGS for t in toks):
            has_closing = True
            break
    has_subject = bool(subject and subject.strip())
    has_body = len(body.split()) >= 15
    components = [has_subject, has_greeting, has_closing, has_body]
    return sum(1 for c in components if c) / len(components)


def _length(body: str) -> float:
    n = len(body.split())
    if n < 15 or n > 280:
        return 0.0
    if 40 <= n <= 180:
        return 1.0
    if n < 40:
        return (n - 15) / (40 - 15)
    return max(0.0, 1.0 - (n - 180) / 100.0)


def _readability(body: str) -> float:
    try:
        fre = textstat.flesch_reading_ease(body)
    except Exception:
        return 0.5
    if 40 <= fre <= 70:
        return 1.0
    if fre > 70:
        return max(0.0, 1.0 - (fre - 70) / 40.0)
    return max(0.0, 1.0 - (40 - fre) / 40.0)


def _placeholder_leak(body: str) -> float:
    lower = body.lower()
    return 0.0 if any(re.search(p, lower) for p in LEAK_PATTERNS) else 1.0


def _hallucination(body: str, facts: list[str]) -> float:
    body_words = set(_content_words(body))
    fact_words: set[str] = set()
    for fact in facts:
        fact_words.update(_content_words(fact))
    extra = body_words - fact_words - COMMON_EMAIL_WORDS
    notable = [w for w in extra if len(w) > 4]
    return max(0.0, 1.0 - len(notable) / 40.0)


def _redundancy(body: str) -> float:
    tokens = _normalize(body).split()
    if len(tokens) < 6:
        return 1.0
    trigrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    if not trigrams:
        return 1.0
    return len(set(trigrams)) / len(trigrams)
