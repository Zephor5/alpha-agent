"""Decision model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.models._ids import (
    Action,
    DecisionId,
    ExpectedFeedback,
    Instant,
    JudgmentRef,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record


@dataclass(frozen=True)
class Decision:
    """Action choice made by the subject."""

    id: DecisionId
    action: Action
    payload: Any
    justified_by: list[JudgmentRef]
    expected_feedback: ExpectedFeedback
    fallback: Decision | None
    decided_at: Instant

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Decision:
        return dataclass_from_record(cls, record)
