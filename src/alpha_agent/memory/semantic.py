"""Semantic memory manager."""

from __future__ import annotations

from alpha_agent.memory.models import MemoryScope, SemanticMemory
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class SemanticMemoryManager:
    """Store and retrieve stable facts, preferences, and concepts."""

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
    ) -> SemanticMemory:
        """Upsert a stable fact by subject, predicate, and object."""

        now = utc_now_iso()
        memory = SemanticMemory(
            id=new_id("sem"),
            subject=subject.strip(),
            predicate=predicate.strip(),
            object=object_value.strip(),
            content=content.strip(),
            confidence=max(0.0, min(1.0, confidence)),
            salience=max(0.0, min(1.0, salience)),
            source_memory_ids=source_memory_ids or [],
            created_at=now,
            updated_at=now,
            metadata={},
            status=status,
            scope=scope or MemoryScope.default(),
        )
        return self.store.upsert_semantic_memory(memory)

    def retrieve_relevant(self, query: str, limit: int = 8) -> list[SemanticMemory]:
        """Retrieve relevant facts using non-vector search."""

        return self.store.search_semantic(query, limit)
