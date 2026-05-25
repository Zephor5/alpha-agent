"""Deterministic v1 value-profile derivation for beliefs."""

from __future__ import annotations

import re
from collections.abc import Iterable

from alpha_agent.cognition.models import CognitiveType, Reference, ValueKind, ValueProfile

KEYWORD_WEIGHTS: dict[ValueKind, dict[str, float]] = {
    ValueKind.SAFETY: {
        "safe": 0.5,
        "safety": 0.6,
        "risk": 0.5,
        "danger": 0.6,
        "harm": 0.6,
        "secure": 0.4,
    },
    ValueKind.HONESTY: {
        "accurate": 0.5,
        "evidence": 0.5,
        "truth": 0.6,
        "true": 0.4,
        "verify": 0.5,
        "honest": 0.6,
    },
    ValueKind.HELPFULNESS: {
        "help": 0.4,
        "helpful": 0.5,
        "useful": 0.4,
        "support": 0.4,
        "answer": 0.3,
    },
    ValueKind.AUTONOMY: {
        "choose": 0.5,
        "choice": 0.5,
        "consent": 0.6,
        "permission": 0.5,
        "control": 0.4,
    },
    ValueKind.EFFICIENCY: {
        "fast": 0.4,
        "faster": 0.4,
        "efficient": 0.6,
        "simple": 0.3,
        "concise": 0.4,
        "speed": 0.4,
    },
    ValueKind.LEARNING: {
        "learn": 0.5,
        "learning": 0.5,
        "teach": 0.5,
        "explain": 0.4,
        "improve": 0.4,
    },
}

TYPE_DEFAULTS: dict[CognitiveType, dict[ValueKind, float]] = {
    CognitiveType.FACTUAL: {ValueKind.HONESTY: 0.3},
    CognitiveType.PROCEDURAL: {ValueKind.EFFICIENCY: 0.3, ValueKind.HELPFULNESS: 0.2},
    CognitiveType.PREFERENCE: {ValueKind.AUTONOMY: 0.3, ValueKind.HELPFULNESS: 0.2},
    CognitiveType.VALUE: {ValueKind.HONESTY: 0.2, ValueKind.SAFETY: 0.2},
    CognitiveType.CAUSAL: {ValueKind.HONESTY: 0.2, ValueKind.LEARNING: 0.2},
    CognitiveType.SOCIAL: {ValueKind.HELPFULNESS: 0.2, ValueKind.AUTONOMY: 0.2},
    CognitiveType.TEMPORAL: {ValueKind.HONESTY: 0.2},
    CognitiveType.META: {ValueKind.LEARNING: 0.3, ValueKind.HONESTY: 0.2},
    CognitiveType.CONCEPT: {ValueKind.LEARNING: 0.2, ValueKind.HELPFULNESS: 0.2},
}

ENTITY_DEFAULTS: dict[str, dict[ValueKind, float]] = {
    "counterpart": {ValueKind.HELPFULNESS: 0.15, ValueKind.AUTONOMY: 0.1},
    "tool": {ValueKind.EFFICIENCY: 0.15, ValueKind.SAFETY: 0.1},
    "system": {ValueKind.SAFETY: 0.15, ValueKind.HONESTY: 0.1},
}


def derive_value_profile(
    content: object,
    structure: object | None = None,
    cognitive_type: CognitiveType | str | None = None,
    entities: Iterable[Reference | object] | None = None,
) -> ValueProfile:
    """Derive small deterministic value weights from belief text and shape."""

    text = _token_text(content, structure)
    weights: dict[ValueKind, float] = {}
    notes: list[str] = []

    if cognitive_type is not None:
        kind = (
            cognitive_type
            if isinstance(cognitive_type, CognitiveType)
            else CognitiveType(str(cognitive_type))
        )
        for value, amount in TYPE_DEFAULTS.get(kind, {}).items():
            _add(weights, value, amount)
        notes.append(f"type:{kind.value}")

    tokens = set(re.findall(r"[a-z0-9_]+", text.casefold()))
    for value, keywords in KEYWORD_WEIGHTS.items():
        matched = sorted(keyword for keyword in keywords if keyword in tokens)
        if not matched:
            continue
        amount = sum(keywords[keyword] for keyword in matched)
        _add(weights, value, amount)
        notes.append(f"keywords:{value.value}:{','.join(matched)}")

    for entity in entities or []:
        entity_kind = _entity_kind(entity)
        for value, amount in ENTITY_DEFAULTS.get(entity_kind, {}).items():
            _add(weights, value, amount)
        if entity_kind in ENTITY_DEFAULTS:
            notes.append(f"entity:{entity_kind}")

    return ValueProfile(
        weights={value: round(min(1.0, amount), 3) for value, amount in weights.items()},
        notes=notes,
    )


def _token_text(content: object, structure: object | None) -> str:
    parts = [str(content)]
    if structure is not None:
        parts.append(str(structure))
    return " ".join(parts)


def _entity_kind(entity: object) -> str:
    if isinstance(entity, Reference):
        return entity.kind
    raw = getattr(entity, "kind", "")
    if raw:
        return str(raw)
    if isinstance(entity, dict):
        return str(entity.get("kind", ""))
    return ""


def _add(weights: dict[ValueKind, float], value: ValueKind, amount: float) -> None:
    weights[value] = weights.get(value, 0.0) + amount
