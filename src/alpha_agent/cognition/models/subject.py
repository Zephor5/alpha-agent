"""Subject model for the single agent self."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._ids import (
    BiasMarker,
    Capability,
    CounterpartRef,
    GroupRef,
    Instant,
    Need,
    Role,
    SelfModel,
    SubjectId,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.value import ValueLens

SUBJECT_SELF: SubjectId = SubjectId("agent:self")


@dataclass(frozen=True)
class Subject:
    """A point-in-time snapshot of the sole cognition subject."""

    id: SubjectId = SUBJECT_SELF
    role: Role = Role("agent")
    capabilities: list[Capability] = field(default_factory=list)
    declared_needs: list[Need] = field(default_factory=list)
    value_lens: ValueLens = field(default_factory=ValueLens)
    self_model: SelfModel = field(default_factory=SelfModel)
    membership: list[GroupRef] = field(default_factory=list)
    served_counterparts: list[CounterpartRef] = field(default_factory=list)
    known_biases: list[BiasMarker] = field(default_factory=list)
    held_at: Instant = Instant("")

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Subject:
        return dataclass_from_record(cls, record)
