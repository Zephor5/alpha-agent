"""Semantic memory lifecycle manager."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable

from alpha_agent.memory.models import (
    MemoryScope,
    SemanticDecisionAction,
    SemanticMemory,
    SemanticMemoryDecision,
    SemanticMemoryStatus,
)
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.text import extract_lightweight_entities, tokenize
from alpha_agent.utils.time import utc_now_iso


class SemanticMemoryManager:
    """Store, correct, merge, and retire atomic semantic memories."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def upsert_fact(
        self,
        subject: str,
        predicate: str,
        object_value: str,
        content: str,
        confidence: float = 0.75,
        salience: float = 0.6,
        source_memory_ids: list[str] | None = None,
        scope: MemoryScope | None = None,
        status: str = "active",
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory:
        """Store a structured semantic fact through lifecycle policy."""

        decision = self.remember_atomic(
            content=content,
            memory_type=_memory_type_for(predicate),
            subject=subject,
            predicate=predicate,
            object_value=object_value,
            confidence=confidence,
            salience=salience,
            source_memory_ids=source_memory_ids,
            scope=scope,
            status=status,
            conn=conn,
        )
        return decision.memory

    def remember_atomic(
        self,
        *,
        content: str,
        memory_type: str = "fact",
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        entities: list[str] | None = None,
        confidence: float = 0.75,
        salience: float = 0.6,
        stability: float = 0.6,
        source_memory_ids: list[str] | None = None,
        scope: MemoryScope | None = None,
        status: str = "active",
        valid_from: str | None = None,
        valid_until: str | None = None,
        metadata: dict[str, object] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemoryDecision:
        """Apply duplicate/conflict policy and persist one atomic semantic memory."""

        memory_scope = scope or MemoryScope.default()
        now = utc_now_iso()
        normalized_subject = _normalize_optional(subject)
        normalized_predicate = _normalize_optional(predicate)
        normalized_object = _normalize_optional(object_value)
        memory_entities = _dedupe(
            [
                *(entities or []),
                *extract_lightweight_entities(content),
                *(item for item in [subject, object_value] if item),
            ]
        )
        memory_status: SemanticMemoryStatus
        if status == "deleted":
            memory_status = "deleted"
        elif status == "superseded":
            memory_status = "superseded"
        else:
            memory_status = "active"
        proposed = SemanticMemory(
            id=new_id("sem"),
            content=content.strip(),
            memory_type=memory_type.strip() or "fact",
            subject=normalized_subject,
            predicate=normalized_predicate,
            object=normalized_object,
            entities=memory_entities,
            confidence=_clamp(confidence),
            salience=_clamp(salience),
            stability=_clamp(stability),
            source_memory_ids=_dedupe(source_memory_ids or []),
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
            status=memory_status,
            valid_from=valid_from or now,
            valid_until=valid_until,
            scope=memory_scope,
        )

        if conn is not None:
            return self._remember_atomic_in_conn(proposed, conn=conn)
        with self.store.immediate_transaction() as local:
            return self._remember_atomic_in_conn(proposed, conn=local)

    def retrieve_relevant(self, query: str, limit: int = 8) -> list[SemanticMemory]:
        """Retrieve relevant active facts using non-vector search."""

        return self.store.search_semantic(query, limit, statuses=["active"])

    def _remember_atomic_in_conn(
        self,
        proposed: SemanticMemory,
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemoryDecision:
        if proposed.status != "active":
            saved = self.store.upsert_semantic_memory(proposed, conn=conn)
            return SemanticMemoryDecision(action="store", memory=saved, rationale="forced status")

        exact = self._active_exact_match(proposed, conn=conn)
        if exact is not None:
            action = _merge_action(exact, proposed)
            if action == "skip":
                return SemanticMemoryDecision(
                    action="skip",
                    memory=exact,
                    matched_memory_ids=[exact.id],
                    rationale="duplicate already contains the same source evidence",
                )
            saved = self._merge_into_existing(exact, proposed, action=action, conn=conn)
            return SemanticMemoryDecision(
                action=action,
                memory=saved,
                matched_memory_ids=[exact.id],
                rationale="same weak structure",
            )

        conflicts = self._active_conflicts(proposed, conn=conn)
        if conflicts and proposed.confidence < 0.65:
            saved = self._store_conflict_review(proposed, conflicts, conn=conn)
            return SemanticMemoryDecision(
                action="conflict-review",
                memory=saved,
                matched_memory_ids=[memory.id for memory in conflicts],
                rationale="low-confidence changed object",
            )
        if conflicts:
            saved = self._supersede_conflicts(proposed, conflicts, conn=conn)
            return SemanticMemoryDecision(
                action="supersede",
                memory=saved,
                matched_memory_ids=[memory.id for memory in conflicts],
                rationale="same subject/predicate with changed object",
            )

        content_match = self._active_content_match(proposed, conn=conn)
        if content_match is not None:
            saved = self._merge_into_existing(content_match, proposed, action="merge", conn=conn)
            return SemanticMemoryDecision(
                action="merge",
                memory=saved,
                matched_memory_ids=[content_match.id],
                rationale="same normalized content",
            )

        similar = self._active_similarity_match(proposed, conn=conn)
        if similar is not None:
            saved = self._merge_into_existing(similar, proposed, action="merge", conn=conn)
            return SemanticMemoryDecision(
                action="merge",
                memory=saved,
                matched_memory_ids=[similar.id],
                rationale="high content/entity similarity",
            )

        saved = self.store.upsert_semantic_memory(proposed, conn=conn)
        return SemanticMemoryDecision(action="store", memory=saved, rationale="new atomic memory")

    def _active_exact_match(
        self,
        proposed: SemanticMemory,
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemory | None:
        if proposed.subject is None or proposed.predicate is None or proposed.object is None:
            return None
        matches = self.store.find_semantic_by_structure(
            subject=proposed.subject,
            predicate=proposed.predicate,
            object_value=proposed.object,
            scope=proposed.scope,
            statuses=["active"],
            conn=conn,
        )
        return matches[0] if matches else None

    def _active_content_match(
        self,
        proposed: SemanticMemory,
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemory | None:
        matches = self.store.find_semantic_by_normalized_content(
            content=proposed.content,
            scope=proposed.scope,
            statuses=["active"],
            conn=conn,
        )
        return matches[0] if matches else None

    def _active_similarity_match(
        self,
        proposed: SemanticMemory,
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemory | None:
        active = self.store.list_semantic_memories(
            limit=100,
            scopes=[proposed.scope],
            statuses=["active"],
            conn=conn,
        )
        proposed_tokens = set(tokenize(proposed.content))
        if not proposed_tokens:
            return None
        proposed_entities = {_normalize(entity) for entity in proposed.entities}
        for memory in active:
            if memory.memory_type != proposed.memory_type:
                continue
            memory_tokens = set(tokenize(memory.content))
            if not memory_tokens:
                continue
            similarity = len(proposed_tokens & memory_tokens) / len(
                proposed_tokens | memory_tokens
            )
            entity_overlap = proposed_entities & {_normalize(entity) for entity in memory.entities}
            if similarity >= 0.92 or (entity_overlap and similarity >= 0.75):
                return memory
        return None

    def _active_conflicts(
        self,
        proposed: SemanticMemory,
        *,
        conn: sqlite3.Connection,
    ) -> list[SemanticMemory]:
        if proposed.subject is None or proposed.predicate is None or proposed.object is None:
            return []
        matches = self.store.find_semantic_by_subject_predicate(
            subject=proposed.subject,
            predicate=proposed.predicate,
            scope=proposed.scope,
            statuses=["active"],
            conn=conn,
        )
        return [memory for memory in matches if memory.object != proposed.object]

    def _merge_into_existing(
        self,
        existing: SemanticMemory,
        proposed: SemanticMemory,
        *,
        action: SemanticDecisionAction,
        conn: sqlite3.Connection,
    ) -> SemanticMemory:
        now = utc_now_iso()
        metadata = {**existing.metadata, **proposed.metadata, "lifecycle_action": action}
        merged = SemanticMemory(
            id=existing.id,
            content=proposed.content or existing.content,
            memory_type=proposed.memory_type or existing.memory_type,
            subject=existing.subject or proposed.subject,
            predicate=existing.predicate or proposed.predicate,
            object=existing.object or proposed.object,
            entities=_dedupe([*existing.entities, *proposed.entities]),
            confidence=max(existing.confidence, proposed.confidence),
            salience=max(existing.salience, proposed.salience),
            stability=max(existing.stability, proposed.stability),
            source_memory_ids=_dedupe(
                [*existing.source_memory_ids, *proposed.source_memory_ids]
            ),
            created_at=existing.created_at,
            updated_at=now,
            metadata=metadata,
            status="active",
            valid_from=existing.valid_from or proposed.valid_from,
            valid_until=existing.valid_until,
            supersedes_id=existing.supersedes_id,
            superseded_by_id=existing.superseded_by_id,
            deleted_at=existing.deleted_at,
            scope=existing.scope,
        )
        return self.store.upsert_semantic_memory(merged, conn=conn)

    def _store_conflict_review(
        self,
        proposed: SemanticMemory,
        conflicts: list[SemanticMemory],
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemory:
        metadata = {
            **proposed.metadata,
            "lifecycle_action": "conflict-review",
            "conflicts_with": [memory.id for memory in conflicts],
        }
        conflict = SemanticMemory(
            id=proposed.id,
            content=proposed.content,
            memory_type=proposed.memory_type,
            subject=proposed.subject,
            predicate=proposed.predicate,
            object=proposed.object,
            entities=list(proposed.entities),
            confidence=proposed.confidence,
            salience=proposed.salience,
            stability=proposed.stability,
            source_memory_ids=list(proposed.source_memory_ids),
            created_at=proposed.created_at,
            updated_at=utc_now_iso(),
            metadata=metadata,
            status="conflict_review",
            valid_from=proposed.valid_from,
            valid_until=proposed.valid_until,
            scope=proposed.scope,
        )
        return self.store.upsert_semantic_memory(conflict, conn=conn)

    def _supersede_conflicts(
        self,
        proposed: SemanticMemory,
        conflicts: list[SemanticMemory],
        *,
        conn: sqlite3.Connection,
    ) -> SemanticMemory:
        now = utc_now_iso()
        source_ids = _dedupe(
            [
                *proposed.source_memory_ids,
                *(source for memory in conflicts for source in memory.source_memory_ids),
            ]
        )
        active = SemanticMemory(
            id=proposed.id,
            content=proposed.content,
            memory_type=proposed.memory_type,
            subject=proposed.subject,
            predicate=proposed.predicate,
            object=proposed.object,
            entities=_dedupe(
                [
                    *proposed.entities,
                    *(entity for memory in conflicts for entity in memory.entities),
                ]
            ),
            confidence=proposed.confidence,
            salience=proposed.salience,
            stability=proposed.stability,
            source_memory_ids=source_ids,
            created_at=proposed.created_at,
            updated_at=now,
            metadata={
                **proposed.metadata,
                "lifecycle_action": "supersede",
                "superseded_memory_ids": [memory.id for memory in conflicts],
            },
            status="active",
            valid_from=proposed.valid_from or now,
            valid_until=proposed.valid_until,
            supersedes_id=conflicts[0].id,
            scope=proposed.scope,
        )
        saved = self.store.upsert_semantic_memory(active, conn=conn)
        for conflict in conflicts:
            retired = SemanticMemory(
                id=conflict.id,
                content=conflict.content,
                memory_type=conflict.memory_type,
                subject=conflict.subject,
                predicate=conflict.predicate,
                object=conflict.object,
                entities=list(conflict.entities),
                confidence=conflict.confidence,
                salience=conflict.salience,
                stability=conflict.stability,
                source_memory_ids=list(conflict.source_memory_ids),
                created_at=conflict.created_at,
                updated_at=now,
                metadata={**conflict.metadata, "lifecycle_action": "superseded"},
                status="superseded",
                valid_from=conflict.valid_from,
                valid_until=now,
                supersedes_id=conflict.supersedes_id,
                superseded_by_id=saved.id,
                scope=conflict.scope,
            )
            self.store.upsert_semantic_memory(retired, conn=conn)
        return saved


def _memory_type_for(predicate: str) -> str:
    normalized = _normalize(predicate)
    if normalized in {"prefers", "likes", "dislikes"}:
        return "preference"
    return "fact"


def _merge_action(existing: SemanticMemory, proposed: SemanticMemory) -> SemanticDecisionAction:
    existing_sources = set(existing.source_memory_ids)
    proposed_sources = set(proposed.source_memory_ids)
    same_content = _normalize(existing.content) == _normalize(proposed.content)
    if same_content and proposed_sources.issubset(existing_sources):
        return "skip"
    if same_content:
        return "merge"
    return "update"


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.casefold().split()).strip()
    return normalized or None


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
