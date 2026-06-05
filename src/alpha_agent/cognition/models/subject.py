"""Subject model for the single agent self."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._ids import (
    Capability,
    CounterpartRef,
    GroupRef,
    Instant,
    Need,
    Role,
    SubjectId,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record

SUBJECT_SELF: SubjectId = SubjectId("agent:self")


@dataclass(frozen=True)
class Subject:
    """A point-in-time snapshot of the sole cognition subject."""

    id: SubjectId = SUBJECT_SELF
    role: Role = Role("agent")
    capabilities: list[Capability] = field(default_factory=list)
    declared_needs: list[Need] = field(default_factory=list)
    membership: list[GroupRef] = field(default_factory=list)
    served_counterparts: list[CounterpartRef] = field(default_factory=list)
    held_at: Instant = Instant("")

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Subject:
        return dataclass_from_record(cls, record)
