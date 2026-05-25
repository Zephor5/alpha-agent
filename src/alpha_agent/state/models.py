"""Session-level state models for Alpha Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ConversationRole = Literal["user", "assistant", "tool"]


@dataclass(frozen=True)
class ConversationMessage:
    """Append-only source message in a session transcript."""

    id: str
    session_id: str
    ordinal: int
    role: ConversationRole
    raw_content: str
    model_content: str | None
    tool_call_id: str | None
    tool_calls: list[dict[str, Any]]
    tool_result_id: str | None
    provider_metadata: dict[str, Any]
    source_metadata: dict[str, Any]
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeTrace:
    """Narrow diagnostic record for runtime behavior."""

    id: str
    session_id: str
    event_type: str
    content: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)
