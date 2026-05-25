"""Procedure model."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.cognition.models._ids import (
    EventId,
    NLStatement,
    ProcedureId,
    Step,
    TriggerPattern,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record


@dataclass(frozen=True)
class Procedure:
    """Learned strategy for recurring cognition patterns."""

    id: ProcedureId
    trigger: TriggerPattern
    steps: list[Step]
    expected_outcome: NLStatement
    learned_from: list[EventId]
    success_count: int
    failure_count: int
    confidence: float

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Procedure:
        return dataclass_from_record(cls, record)
