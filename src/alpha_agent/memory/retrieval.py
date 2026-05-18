"""Memory retrieval and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log1p
from typing import cast

from alpha_agent.memory.models import (
    EpisodicMemory,
    ProceduralMemory,
    RetrievedContext,
    SemanticMemory,
)
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.utils.text import extract_lightweight_entities, keyword_score
from alpha_agent.utils.time import utc_now


@dataclass(frozen=True)
class RankedMemory:
    """Memory plus an explicit retrieval score."""

    memory: EpisodicMemory | SemanticMemory | ProceduralMemory
    memory_type: str
    score: float


class MemoryRetriever:
    """Retrieve context with explicit non-vector ranking."""

    def __init__(self, store: MemoryStore, working_memory: WorkingMemoryManager):
        self.store = store
        self.working_memory = working_memory

    def retrieve_context(self, query: str, session_id: str, limit: int = 8) -> RetrievedContext:
        """Retrieve working, episodic, semantic, and procedural context for a turn."""

        working = self.working_memory.get_active_context(session_id)
        episodic = self._rank_episodic(query, limit)
        semantic = self._rank_semantic(query, limit)
        procedural = self._rank_procedural(query, max(3, limit // 2))

        for ranked in [*episodic, *semantic, *procedural]:
            self.store.log_memory_access(ranked.memory.id, ranked.memory_type, query, ranked.score)

        entity_hints = extract_lightweight_entities(query)
        return RetrievedContext(
            working_memory=working,
            episodic_memories=cast(list[EpisodicMemory], [item.memory for item in episodic]),
            semantic_memories=cast(list[SemanticMemory], [item.memory for item in semantic]),
            procedural_memories=cast(
                list[ProceduralMemory],
                [item.memory for item in procedural],
            ),
            entity_hints=entity_hints,
        )

    def _rank_episodic(self, query: str, limit: int) -> list[RankedMemory]:
        searched = self.store.search_episodic(query, limit=limit * 3)
        recent = self.store.list_episodic_memories(limit=limit * 3)
        candidates = self._dedupe([*searched, *recent])
        ranked = [
            RankedMemory(memory=m, memory_type="episodic", score=self._score(query, m, "episodic"))
            for m in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _rank_semantic(self, query: str, limit: int) -> list[RankedMemory]:
        searched = self.store.search_semantic(query, limit=limit * 3)
        recent = self.store.list_semantic_memories(limit=limit * 3)
        candidates = self._dedupe([*searched, *recent])
        ranked = [
            RankedMemory(memory=m, memory_type="semantic", score=self._score(query, m, "semantic"))
            for m in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _rank_procedural(self, query: str, limit: int) -> list[RankedMemory]:
        searched = self.store.search_procedural(query, limit=limit * 3)
        recent = self.store.list_procedural_memories(limit=limit * 3)
        candidates = self._dedupe([*searched, *recent])
        ranked = [
            RankedMemory(
                memory=m,
                memory_type="procedural",
                score=self._score(query, m, "procedural"),
            )
            for m in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _score(
        self,
        query: str,
        memory: EpisodicMemory | SemanticMemory | ProceduralMemory,
        memory_type: str,
    ) -> float:
        text = self._memory_text(memory)
        kw = keyword_score(query, text)
        salience = getattr(memory, "salience", getattr(memory, "confidence", 0.5))
        recency = self._recency_score(getattr(memory, "updated_at", memory.created_at))
        access = self._access_score(getattr(memory, "access_count", 0))
        type_boost = {"semantic": 0.9, "episodic": 0.75, "procedural": 0.65}.get(memory_type, 0.5)
        return (
            kw * 0.40
            + salience * 0.25
            + recency * 0.20
            + access * 0.10
            + type_boost * 0.05
        )

    def _memory_text(self, memory: EpisodicMemory | SemanticMemory | ProceduralMemory) -> str:
        if isinstance(memory, EpisodicMemory):
            return " ".join([memory.content, memory.summary, *memory.people, *memory.topics])
        if isinstance(memory, SemanticMemory):
            return " ".join([memory.subject, memory.predicate, memory.object, memory.content])
        return " ".join(
            [memory.name, memory.description, memory.trigger, memory.procedure_markdown]
        )

    def _recency_score(self, iso_value: str) -> float:
        try:
            created = datetime.fromisoformat(iso_value)
        except ValueError:
            return 0.0
        age_days = max(0.0, (utc_now() - created).total_seconds() / 86400)
        return exp(-age_days / 30)

    def _access_score(self, access_count: int) -> float:
        return min(1.0, log1p(max(0, access_count)) / log1p(10))

    def _dedupe(self, values: list) -> list:
        seen: set[str] = set()
        result = []
        for value in values:
            if value.id in seen:
                continue
            seen.add(value.id)
            result.append(value)
        return result
