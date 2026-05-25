"""Reflection model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alpha_agent.cognition.models._ids import (
    Instant,
    NLStatement,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record


@dataclass(frozen=True)
class Reflection:
    """Metacognitive observation over prior cognition."""

    id: ReflectionId
    level: Literal["L1", "L2", "L3"]
    kind: ReflectionKind
    severity: Severity
    target: ReflectionTarget
    finding: NLStatement
    suggested_remedy: RemedyHint
    created_at: Instant

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Reflection:
        return dataclass_from_record(cls, record)
