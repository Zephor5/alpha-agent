"""Stimulus and perception models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models._ids import (
    CounterpartRef,
    EntityRef,
    Instant,
    IntentMarker,
    PerceptionId,
    Reference,
    SituationRef,
    SubjectRef,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import StimulusKind
from alpha_agent.cognition.models.thread import ThreadId


@dataclass(frozen=True)
class Stimulus:
    """Raw input routed into cognition."""

    kind: StimulusKind
    source: CounterpartRef | None
    payload: Any
    thread_id: ThreadId
    received_at: Instant
    source_refs: list[Reference] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Stimulus:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class Perception:
    """Subject-scoped perception of a stimulus."""

    id: PerceptionId
    source_kind: StimulusKind
    from_counterpart: CounterpartRef | None
    raw: Any
    surface_intent: list[IntentMarker]
    raised_entities: list[EntityRef]
    subject: SubjectRef
    situation: SituationRef
    received_at: Instant

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Perception:
        return dataclass_from_record(cls, record)
