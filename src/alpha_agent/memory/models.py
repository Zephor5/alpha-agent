"""Memory domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MemoryType = Literal["episodic", "semantic", "procedural"]
CandidateType = Literal["episodic", "semantic", "procedural_candidate"]
CandidateStatus = Literal["pending", "approved", "auto_approved", "rejected", "edited"]
ConversationRole = Literal["user", "assistant", "tool"]
MemoryCaptureMode = Literal["disabled", "candidate_only", "auto_approve_explicit"]
ScopeKind = Literal["global_user", "platform_user", "chat_thread", "project"]


@dataclass(frozen=True)
class MemoryScope:
    """Explicit read/write scope for long-term memory."""

    kind: ScopeKind
    scope_key: str
    user_id: str | None = None
    platform: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    project_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "MemoryScope":
        """Return the deterministic local CLI scope."""

        return cls(kind="global_user", scope_key="user:default", user_id="default")

    @classmethod
    def from_source_metadata(
        cls,
        *,
        session_id: str,
        source_metadata: dict[str, Any] | None,
    ) -> "MemoryScope":
        """Derive a normalized memory scope from CLI/gateway source metadata."""

        metadata = dict(source_metadata or {})
        explicit = metadata.get("memory_scope")
        if isinstance(explicit, dict):
            return cls.from_record(explicit)

        project_id = _optional_str(metadata.get("project_id") or metadata.get("project"))
        if project_id:
            user_id = _optional_str(metadata.get("user_id")) or "default"
            return cls(
                kind="project",
                scope_key=f"project:{project_id}:user:{user_id}",
                user_id=user_id,
                project_id=project_id,
                metadata={"session_id": session_id},
            )

        platform = _optional_str(metadata.get("platform"))
        user_id = _optional_str(metadata.get("user_id"))
        chat_id = _optional_str(metadata.get("chat_id"))
        thread_id = _optional_str(metadata.get("thread_id"))
        if platform and chat_id and user_id:
            thread_part = thread_id or "main"
            return cls(
                kind="chat_thread",
                scope_key=(
                    f"platform:{platform.lower()}:chat:{chat_id}:"
                    f"thread:{thread_part}:user:{user_id}"
                ),
                user_id=user_id,
                platform=platform.lower(),
                chat_id=chat_id,
                thread_id=thread_id,
                metadata={
                    "chat_type": _optional_str(metadata.get("chat_type")),
                    "session_id": session_id,
                },
            )
        if platform and user_id:
            return cls(
                kind="platform_user",
                scope_key=f"platform:{platform.lower()}:user:{user_id}",
                user_id=user_id,
                platform=platform.lower(),
                metadata={"session_id": session_id},
            )
        return cls.default()

    @classmethod
    def from_record(cls, record: dict[str, Any] | None) -> "MemoryScope":
        """Load a memory scope from stored JSON metadata."""

        if not record:
            return cls.default()
        kind = str(record.get("kind") or "global_user")
        if kind not in {"global_user", "platform_user", "chat_thread", "project"}:
            kind = "global_user"
        scope_key = str(record.get("scope_key") or "user:default")
        metadata = record.get("metadata")
        return cls(
            kind=kind,  # type: ignore[arg-type]
            scope_key=scope_key,
            user_id=_optional_str(record.get("user_id")),
            platform=_optional_str(record.get("platform")),
            chat_id=_optional_str(record.get("chat_id")),
            thread_id=_optional_str(record.get("thread_id")),
            project_id=_optional_str(record.get("project_id")),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def to_record(self) -> dict[str, Any]:
        """Serialize the scope to stable JSON-compatible metadata."""

        return {
            "kind": self.kind,
            "scope_key": self.scope_key,
            "user_id": self.user_id,
            "platform": self.platform,
            "chat_id": self.chat_id,
            "thread_id": self.thread_id,
            "project_id": self.project_id,
            "metadata": self.metadata,
        }

    def allowed_read_scopes(self) -> list["MemoryScope"]:
        """Return scopes visible to the current turn, most-specific first."""

        scopes = [self]
        if self.kind == "chat_thread" and self.platform and self.user_id:
            scopes.append(
                MemoryScope(
                    kind="platform_user",
                    scope_key=f"platform:{self.platform}:user:{self.user_id}",
                    user_id=self.user_id,
                    platform=self.platform,
                )
            )
        if self.kind == "project" and self.user_id:
            scopes.append(
                MemoryScope(
                    kind="global_user",
                    scope_key=f"user:{self.user_id}",
                    user_id=self.user_id,
                )
            )
        return _dedupe_scopes(scopes)


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
    scope: MemoryScope = field(default_factory=MemoryScope.default)


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
    status: str = "active"
    scope: MemoryScope = field(default_factory=MemoryScope.default)


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
    scope: MemoryScope = field(default_factory=MemoryScope.default)


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


@dataclass(frozen=True)
class MemoryCandidate:
    """Stored candidate awaiting or recording a memory decision."""

    id: str
    candidate_type: CandidateType
    proposed_layer: MemoryType
    content: str
    weak_structure: dict[str, Any]
    salience: float
    confidence: float
    scope: MemoryScope
    source_message_ids: list[str]
    status: CandidateStatus
    created_at: str
    updated_at: str
    reviewer_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryDecision:
    """Auditable decision made for a stored candidate."""

    id: str
    candidate_id: str
    action: str
    memory_type: str | None
    memory_id: str | None
    reviewer: str | None
    rationale: str
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


def proposed_layer_for_candidate(candidate_type: CandidateType) -> MemoryType:
    """Map extraction candidate type to its proposed durable layer."""

    if candidate_type == "semantic":
        return "semantic"
    if candidate_type == "procedural_candidate":
        return "procedural"
    return "episodic"


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _dedupe_scopes(scopes: list[MemoryScope]) -> list[MemoryScope]:
    seen: set[tuple[str, str]] = set()
    result: list[MemoryScope] = []
    for scope in scopes:
        key = (scope.kind, scope.scope_key)
        if key in seen:
            continue
        seen.add(key)
        result.append(scope)
    return result
