"""Session-level state models for Alpha Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SessionMessageKind = Literal[
    "system_reminder",
    "user_message",
    "assistant_message",
    "tool_message",
    "compressed_message",
]
LLMRole = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class SessionMessage:
    """Append-only source message in a session stream."""

    id: str
    session_id: str
    ordinal: int
    kind: SessionMessageKind
    llm_role: LLMRole | None
    raw_content: str
    model_content: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]]
    tool_result_id: str | None
    provider_metadata: dict[str, Any]
    source_metadata: dict[str, Any]
    compression_point_ordinal: int | None
    compression_version: str | None
    created_at: str
    updated_at: str | None = None
    reasoning_content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSummarySnapshot:
    """Stable summary selected for one session prompt context slot."""

    session_id: str
    summary_kind: str
    target_kind: str
    target_id: str
    source_belief_id: str
    content: str
    created_at: str


@dataclass(frozen=True)
class SessionCounterpart:
    """Counterpart identity bound to a session."""

    session_id: str
    counterpart_id: str
    source_metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class RuntimeTrace:
    """Narrow diagnostic record for runtime behavior."""

    id: str
    session_id: str
    event_type: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
