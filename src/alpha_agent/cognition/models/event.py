"""Cognitive event model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models._ids import (
    ActorRef,
    EventId,
    Instant,
    NLStatement,
    Reference,
    SituationRef,
    SubjectRef,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import CognitiveEventKind


@dataclass(frozen=True)
class CognitiveEvent:
    """Append-only cognition event."""

    id: EventId
    kind: CognitiveEventKind
    subject: SubjectRef
    subject_version: int
    situation: SituationRef | None
    inputs: list[Reference] = field(default_factory=list)
    outputs: list[Reference] = field(default_factory=list)
    rationale: NLStatement = NLStatement("")
    timestamp: Instant = Instant("")
    actor: ActorRef = field(default_factory=lambda: ActorRef("actor", "unknown"))
    causal_parents: list[EventId] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> CognitiveEvent:
        return dataclass_from_record(cls, record)
