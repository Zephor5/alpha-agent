"""Thread identifiers for conversation and cognition contexts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models._ids import SubjectId
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import ThreadKind


@dataclass(frozen=True)
class ThreadId:
    """Stable identifier for a context window."""

    kind: ThreadKind
    key: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_session(
        cls,
        session_id: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> ThreadId:
        return cls(
            kind=ThreadKind.CONVERSATION,
            key=f"session:{session_id}",
            metadata=source_metadata or {},
        )

    @classmethod
    def cognition(cls, subject_id: SubjectId, topic: str) -> ThreadId:
        normalized = "-".join(topic.strip().lower().split()) or "default"
        return cls(
            kind=ThreadKind.COGNITION,
            key=f"subject:{subject_id}:topic:{normalized}",
            metadata={"topic": topic},
        )

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> ThreadId:
        return dataclass_from_record(cls, record)
