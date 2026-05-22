"""Memory domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MemoryType = Literal["episodic", "semantic", "procedural"]
CandidateType = Literal["episodic", "semantic", "procedural_candidate"]
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
class SessionContextState:
    """Active compressed context projection for a session."""

    session_id: str
    compressed_until_ordinal: int
    summary: str
    summary_source_message_ids: list[str]
    compression_version: str
    created_at: str
    updated_at: str
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


@dataclass(frozen=True)
class EpisodicMemory:
    """Specific event or experience preserved beyond the raw event log."""

    id: str
    content: str
    summary: str
    source_event_ids: list[str]
    people: list[str]
    places: list[str]
    topics: list[str]
    salience: float
    confidence: float
    created_at: str
    last_accessed_at: str | None = None
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticMemory:
    """Stable fact, preference, concept, or user-specific knowledge."""

    id: str
    subject: str
    predicate: str
    object: str
    content: str
    confidence: float
    salience: float
    source_memory_ids: list[str]
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProceduralMemory:
    """Reusable way of doing something."""

    id: str
    name: str
    description: str
    trigger: str
    procedure_markdown: str
    success_count: int
    failure_count: int
    confidence: float
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    """Memory context selected for a turn."""

    episodic_memories: list[EpisodicMemory]
    semantic_memories: list[SemanticMemory]
    procedural_memories: list[ProceduralMemory]
    entity_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExtractedMemoryCandidate:
    """Deterministic memory candidate produced after a turn."""

    type: CandidateType
    content: str
    salience: float
    confidence: float
    subject: str | None = None
    predicate: str | None = None
    object: str | None = None
    source_event_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
