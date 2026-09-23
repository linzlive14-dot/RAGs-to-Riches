"""Input and grounding guardrails for policy retrieval."""

from __future__ import annotations

import math
import re

MAX_QUERY_CHARACTERS = 1_000
INJECTION_PATTERNS = (
    re.compile(r"\bignore (all |any )?(previous|prior|system) instructions?\b", re.IGNORECASE),
    re.compile(r"\breveal (the )?(system prompt|developer message|api key|secret)\b", re.IGNORECASE),
    re.compile(r"\bact as (an? )?(unrestricted|unfiltered)\b", re.IGNORECASE),
)


class UnsafeQueryError(ValueError):
    """Raised when a query tries to override controls or obtain secrets."""


def validate_query(query: str) -> str:
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("Query cannot be empty")
    if len(normalized) > MAX_QUERY_CHARACTERS:
        raise ValueError(f"Query cannot exceed {MAX_QUERY_CHARACTERS} characters")
    if any(pattern.search(normalized) for pattern in INJECTION_PATTERNS):
        raise UnsafeQueryError("The request cannot override system controls or request secrets.")
    return normalized


def has_sufficient_evidence(query: str, texts: list[str], *, minimum_overlap: int = 2) -> bool:
    """Require meaningful lexical support before treating retrieval as grounded."""

    stop_words = {
        "about",
        "could",
        "does",
        "from",
        "have",
        "please",
        "should",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query.lower())
        if len(term) >= 4 and term not in stop_words
    }
    evidence_terms = set(re.findall(r"[a-z0-9]+", " ".join(texts).lower()))
    required_overlap = max(minimum_overlap, math.ceil(len(query_terms) * 0.6))
    return len(query_terms & evidence_terms) >= required_overlap
