"""Memory retrieval and ranking."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp, log1p
from typing import cast

from alpha_agent.memory.models import (
    EpisodicMemory,
    MemoryScope,
    ProceduralMemory,
    RetrievedContext,
    SemanticMemory,
)
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.text import extract_lightweight_entities, keyword_score
from alpha_agent.utils.time import utc_now


@dataclass(frozen=True)
class RankedMemory:
    """Memory plus an explicit retrieval score."""

    memory: EpisodicMemory | SemanticMemory | ProceduralMemory
    memory_type: str
    score: float


class MemoryRetriever:
    """Retrieve long-term memory with explicit non-vector ranking."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def retrieve_context(
        self,
        query: str,
        session_id: str,
        limit: int = 8,
        *,
        scopes: list[MemoryScope] | None = None,
        record_access: bool = True,
        access_scope: MemoryScope | None = None,
    ) -> RetrievedContext:
        """Retrieve episodic, semantic, and procedural context for a turn."""

        del session_id
        episodic = self._rank_episodic(query, limit, scopes=scopes)
        semantic = self._rank_semantic(query, limit, scopes=scopes)
        procedural = self._rank_procedural(query, max(3, limit // 2), scopes=scopes)

        if record_access:
            for ranked in [*episodic, *semantic, *procedural]:
                self.store.log_memory_access(
                    ranked.memory.id,
                    ranked.memory_type,
                    query,
                    ranked.score,
                    scope=access_scope,
                )

        entity_hints = extract_lightweight_entities(query)
        return RetrievedContext(
            episodic_memories=cast(list[EpisodicMemory], [item.memory for item in episodic]),
            semantic_memories=cast(list[SemanticMemory], [item.memory for item in semantic]),
            procedural_memories=cast(
                list[ProceduralMemory],
                [item.memory for item in procedural],
            ),
            entity_hints=entity_hints,
        )

    def _rank_episodic(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        searched = self.store.search_episodic(query, limit=limit * 3, scopes=scopes)
        recent = self.store.list_episodic_memories(limit=limit * 3, scopes=scopes)
        candidates = self._dedupe([*searched, *recent])
        ranked = [
            RankedMemory(memory=m, memory_type="episodic", score=self._score(query, m, "episodic"))
            for m in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _rank_semantic(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        searched = self.store.search_semantic(
            query,
            limit=limit * 3,
            scopes=scopes,
            statuses=["active"],
        )
        recent = self.store.list_semantic_memories(
            limit=limit * 3,
            scopes=scopes,
            statuses=["active"],
        )
        candidates = self._dedupe([*searched, *recent])
        ranked = [
            RankedMemory(memory=m, memory_type="semantic", score=self._score(query, m, "semantic"))
            for m in candidates
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked[:limit]

    def _rank_procedural(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        searched = self.store.search_procedural(query, limit=limit * 3, scopes=scopes)
        recent = self.store.list_procedural_memories(limit=limit * 3, scopes=scopes)
        candidates = [
            memory
            for memory in self._dedupe([*searched, *recent])
            if self._procedural_text_relevance(query, memory) > 0
        ]
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

    def _procedural_text_relevance(self, query: str, memory: ProceduralMemory) -> float:
        return keyword_score(
            query,
            " ".join([memory.name, memory.description, memory.trigger]),
        )

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
            return " ".join(
                [
                    memory.subject or "",
                    memory.predicate or "",
                    memory.object or "",
                    memory.content,
                    *memory.entities,
                ]
            )
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
