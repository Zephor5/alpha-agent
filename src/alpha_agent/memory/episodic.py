"""Episodic memory manager."""

from __future__ import annotations

from alpha_agent.memory.models import EpisodicMemory, MemoryScope
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.text import extract_lightweight_entities, tokenize
from alpha_agent.utils.time import utc_now_iso


class EpisodicMemoryManager:
    """Create and retrieve specific remembered experiences."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def create(
        self,
        content: str,
        source_event_ids: list[str],
        salience: float,
        confidence: float = 0.7,
        scope: MemoryScope | None = None,
    ) -> EpisodicMemory:
        """Create an episodic memory from important events."""

        topics = sorted(set(tokenize(content)))[:12]
        people = extract_lightweight_entities(content)
        summary = content if len(content) <= 180 else f"{content[:177].rstrip()}..."
        memory = EpisodicMemory(
            id=new_id("epi"),
            content=content,
            summary=summary,
            source_event_ids=source_event_ids,
            people=people,
            places=[],
            topics=topics,
            salience=max(0.0, min(1.0, salience)),
            confidence=max(0.0, min(1.0, confidence)),
            created_at=utc_now_iso(),
            metadata={},
            scope=scope or MemoryScope.default(),
        )
        return self.store.insert_episodic_memory(memory)

    def retrieve_relevant(self, query: str, limit: int = 8) -> list[EpisodicMemory]:
        """Retrieve relevant episodic memories using non-vector search."""

        return self.store.search_episodic(query, limit)
