"""Judgment model."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.cognition.models._ids import (
    Applicability,
    BeliefRef,
    Instant,
    JudgmentId,
    NLStatement,
    SituationRef,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import ValueKind


@dataclass(frozen=True)
class Judgment:
    """Short-lived claim formed during a cognition turn."""

    id: JudgmentId
    claim: NLStatement
    supports: list[BeliefRef]
    undermined_by: list[BeliefRef]
    applicable_under: Applicability
    confidence: float
    value_weights: dict[ValueKind, float]
    formed_in: SituationRef
    expires_at: Instant | None = None

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Judgment:
        return dataclass_from_record(cls, record)
