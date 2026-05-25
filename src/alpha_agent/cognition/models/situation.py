"""Situation model."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._ids import (
    CounterpartRef,
    CulturalContext,
    HistoricalContext,
    InformationalContext,
    InstitutionalContext,
    PhysicalContext,
    SituationId,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record


@dataclass(frozen=True)
class AuthorityHint:
    """Authority relation hint for a present counterpart."""

    counterpart: CounterpartRef
    authority: str

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> AuthorityHint:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class SocialContext:
    """Counterparts and social structure visible in a situation."""

    present_counterparts: list[CounterpartRef] = field(default_factory=list)
    authority_hints: list[AuthorityHint] = field(default_factory=list)
    group_dynamics: list[str] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> SocialContext:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class Situation:
    """Typed context dimensions for a cognition turn."""

    id: SituationId
    physical: PhysicalContext = PhysicalContext("")
    social: SocialContext = field(default_factory=SocialContext)
    institutional: InstitutionalContext = InstitutionalContext("")
    informational: InformationalContext = InformationalContext("")
    cultural: CulturalContext = CulturalContext("")
    historical: HistoricalContext = HistoricalContext("")

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Situation:
        return dataclass_from_record(cls, record)
