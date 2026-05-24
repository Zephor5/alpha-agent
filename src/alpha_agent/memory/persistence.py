"""Persistence helpers for extracted memory candidates."""

from __future__ import annotations

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


def persist_candidates(
    store: MemoryStore,
    candidates: list[ExtractedMemoryCandidate],
    *,
    scope: MemoryScope | None = None,
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
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="episodic",
                    memory_id=episodic_memory.id,
                )
            )
        elif candidate.type == "semantic" and candidate.subject and candidate.predicate:
            semantic_memory = semantic.upsert_fact(
                subject=candidate.subject,
                predicate=candidate.predicate,
                object_value=candidate.object or "",
                content=candidate.content,
                confidence=candidate.confidence,
                salience=candidate.salience,
                source_memory_ids=candidate.source_event_ids,
                scope=memory_scope,
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="semantic",
                    memory_id=semantic_memory.id,
                )
            )
        elif candidate.type == "procedural_candidate":
            procedural_episode = episodic.create(
                content=f"Procedural candidate: {candidate.content}",
                source_event_ids=candidate.source_event_ids,
                salience=candidate.salience,
                confidence=candidate.confidence,
                scope=memory_scope,
            )
            persisted.append(
                PersistedMemory(
                    candidate_type=candidate.type,
                    memory_type="episodic",
                    memory_id=procedural_episode.id,
                )
            )
    return persisted
