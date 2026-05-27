"""Counterpart model skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models._ids import CounterpartId, Instant
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import CounterpartRole


@dataclass(frozen=True)
class ServiceCommitment:
    """A service promise between the agent and a counterpart."""

    id: str
    description: str
    status: str = "committed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)


@dataclass(frozen=True)
class Relationship:
    """Relationship state between subject and counterpart."""

    kind: str = "observed"
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StyleHint:
    """Communication style preference inferred for a counterpart."""

    kind: str
    value: str
    confidence: float = 0.5


@dataclass(frozen=True)
class Counterpart:
    """A party observed or served by the single cognition subject."""

    id: CounterpartId
    role: CounterpartRole
    identity: dict[str, Any]
    relationship: Relationship
    service_contract: list[ServiceCommitment]
    trust_level: float
    communication_style: list[StyleHint]
    first_seen_at: Instant
    last_interaction_at: Instant
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Counterpart:
        return dataclass_from_record(cls, record)
