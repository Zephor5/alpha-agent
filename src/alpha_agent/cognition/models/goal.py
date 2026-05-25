"""Goal model for Drive Loop scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from alpha_agent.cognition.models._ids import BeliefId, CounterpartRef, GoalId, Instant
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record

GoalStatus = Literal["active", "satisfied", "abandoned"]
GoalSource = Literal["user", "reflector_l2", "external"]


@dataclass(frozen=True)
class Goal:
    """A first-class drive target for the single cognition subject."""

    id: GoalId
    description: str
    target_outcome: str = ""
    priority: int = 0
    status: GoalStatus = "active"
    source: GoalSource = "user"
    linked_belief_ids: list[BeliefId] = field(default_factory=list)
    for_counterpart: CounterpartRef | None = None
    created_at: Instant = Instant("")
    updated_at: Instant = Instant("")
    last_drive_at: Instant | None = None

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Goal:
        return dataclass_from_record(cls, record)
