"""Value model skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import ValueKind


@dataclass(frozen=True)
class ValueProfile:
    """Value weights attached to a belief or judgment."""

    weights: dict[ValueKind, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> ValueProfile:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class ValueLens:
    """Subject-level ordering for value conflict resolution."""

    priorities: list[ValueKind] = field(default_factory=list)
    weights: dict[ValueKind, float] = field(default_factory=dict)
    sensitivity: dict[ValueKind, float] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> ValueLens:
        return dataclass_from_record(cls, record)
