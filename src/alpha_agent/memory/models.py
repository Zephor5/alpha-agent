"""Memory domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MemoryType = Literal["episodic", "semantic", "procedural"]
CandidateType = Literal["episodic", "semantic", "procedural_candidate"]


@dataclass(frozen=True)
class Event:
    """Raw chronological experience."""

    id: str
    session_id: str
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkingMemoryItem:
    """Short-lived active context."""

    id: str
    session_id: str
    content: str
    source_event_id: str | None
    priority: float
    expires_at: str | None
    created_at: str
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

    working_memory: list[WorkingMemoryItem]
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
