"""Small text-processing helpers for deterministic retrieval and extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-']*")
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3}\b")


def normalize_text(value: str) -> str:
    """Normalize text for matching without hiding the original text elsewhere."""

    return " ".join(value.lower().split())


def tokenize(value: str) -> list[str]:
    """Tokenize text for transparent non-vector matching."""

    return [match.group(0).lower() for match in TOKEN_RE.finditer(value)]


def keyword_score(query: str, text: str) -> float:
    """Return simple overlap score between query tokens and text tokens."""

    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    overlap = query_tokens & text_tokens
    return min(1.0, len(overlap) / len(query_tokens))


def extract_lightweight_entities(value: str) -> list[str]:
    """Extract obvious title-cased entity mentions for lightweight matching."""

    entities: list[str] = []
    for match in ENTITY_RE.finditer(value):
        entity = match.group(0).strip()
        if entity.lower() in {"i", "the", "a", "an"}:
            continue
        if entity not in entities:
            entities.append(entity)
    return entities[:12]


def contains_any(value: str, phrases: Iterable[str]) -> bool:
    """Return true when any phrase appears in normalized text."""

    haystack = normalize_text(value)
    return any(phrase in haystack for phrase in phrases)
