"""Persistence helpers for extracted memory candidates."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import ExtractedMemoryCandidate, MemoryScope
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class PersistedMemory:
    """Memory item persisted from an extracted candidate."""

    candidate_type: str
    memory_type: str
    memory_id: str
    action: str = "store"


def persist_candidates(
    store: MemoryStore,
    candidates: list[ExtractedMemoryCandidate],
    *,
    scope: MemoryScope | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[PersistedMemory]:
    """Persist approved extracted candidates using the runtime memory mapping."""

    episodic = EpisodicMemoryManager(store)
    semantic = SemanticMemoryManager(store)
    memory_scope = scope or MemoryScope.default()
    persisted: list[PersistedMemory] = []
    for candidate in candidates:
        if candidate.type == "episodic":
            episodic_memory = episodic.create(
                content=candidate.content,
                source_event_ids=candidate.source_event_ids,
                salience=candidate.salience,
                confidence=candidate.confidence,
                scope=memory_scope,
                conn=conn,
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="episodic",
                    memory_id=episodic_memory.id,
                    action="store",
                )
            )
        elif candidate.type == "semantic" and candidate.subject and candidate.predicate:
            decision = semantic.remember_atomic(
                content=candidate.content,
                memory_type=_semantic_memory_type(candidate),
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object or "",
                entities=list(candidate.entities),
                confidence=candidate.confidence,
                salience=candidate.salience,
                stability=candidate.stability,
                source_memory_ids=candidate.source_event_ids,
                scope=memory_scope,
                metadata=dict(candidate.metadata),
                conn=conn,
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="semantic",
                    memory_id=decision.memory.id,
                    action=decision.action,
                )
            )
        elif candidate.type == "procedural_candidate":
            procedural_episode = episodic.create(
                content=f"Procedural candidate: {candidate.content}",
                source_event_ids=candidate.source_event_ids,
                salience=candidate.salience,
                confidence=candidate.confidence,
                scope=memory_scope,
                conn=conn,
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="episodic",
                    memory_id=procedural_episode.id,
                    action="store",
                )
            )
    return persisted


def _semantic_memory_type(candidate: ExtractedMemoryCandidate) -> str:
    metadata_type = candidate.metadata.get("memory_type")
    if isinstance(metadata_type, str) and metadata_type.strip():
        return metadata_type.strip()
    if candidate.predicate in {"prefers", "likes", "dislikes"}:
        return "preference"
    return "fact"
