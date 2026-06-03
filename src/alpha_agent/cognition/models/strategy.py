"""Strategy override model for background control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models._ids import (
    CounterpartRef,
    Instant,
    StrategyId,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record


@dataclass(frozen=True)
class StrategyOverride:
    """A temporary rule that changes a concrete runtime or worker domain."""

    id: StrategyId
    name: str
    payload: dict[str, Any] = field(default_factory=dict)
    target_domains: list[str] = field(default_factory=list)
    for_counterpart: CounterpartRef | None = None
    set_by: str = "reflector_l2"
    set_at: Instant = Instant("")
    valid_until: Instant = Instant("")

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> StrategyOverride:
        return dataclass_from_record(cls, record)
