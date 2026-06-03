"""Context window model."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._ids import (
    BeliefRef,
    CompressedSummary,
    CounterpartRef,
    Instant,
    ProcedureRef,
    SituationRef,
    SubjectRef,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.perception import Perception


@dataclass(frozen=True)
class ContextWindow:
    """Projection view assembled for a runtime session."""

    session_id: str
    counterpart: CounterpartRef | None
    foreground: list[Perception]
    background: CompressedSummary | None
    recalled: list[BeliefRef]
    matched_procedures: list[ProcedureRef]
    subject_at: SubjectRef
    situation_at: SituationRef
    assembled_at: Instant
    metadata: dict[str, object] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> ContextWindow:
        return dataclass_from_record(cls, record)
