"""Session-level state models for Alpha Agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

SessionMessageKind = Literal[
    "system_reminder",
    "system_message",
    "user_message",
    "assistant_message",
    "tool_message",
    "compressed_message",
]
LLMRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class SessionRecord:
    """Durable session-level metadata."""

    session_id: str
    timezone: str
    created_at: str
    updated_at: str


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


@dataclass(frozen=True)
class ImportBatchRecord:
    """Durable summary for one external conversation import attempt."""

    id: str
    source_provider: str
    input_name: str | None
    payload_digest: str
    status: str
    conversations_seen: int
    messages_seen: int
    conversations_created: int
    conversations_reused: int
    messages_inserted: int
    messages_deduped: int
    created_at: str
    updated_at: str
    error_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportedConversationRecord:
    """Mapping from one external conversation to one hidden Alpha session."""

    id: str
    source_provider: str
    external_conversation_id: str
    session_id: str
    title: str | None
    external_created_at: str | None
    external_updated_at: str | None
    first_import_batch_id: str
    latest_import_batch_id: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportedMessageRecord:
    """Mapping from one external message identity to one session message."""

    id: str
    source_provider: str
    external_conversation_id: str
    external_message_id: str
    imported_conversation_id: str
    session_message_id: str
    import_batch_id: str
    role: str
    external_created_at: str
    imported_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ImportStatusSummary:
    """Aggregate import and extraction progress for one batch."""

    batch_id: str
    source_provider: str
    status: str
    conversations_seen: int
    messages_seen: int
    conversations_created: int
    conversations_reused: int
    messages_inserted: int
    messages_deduped: int
    extraction_pending: int
    extraction_claimed: int
    extraction_processed: int
    extraction_failed: int
    extraction_skipped: int
    created_at: str
    updated_at: str
    error_summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
