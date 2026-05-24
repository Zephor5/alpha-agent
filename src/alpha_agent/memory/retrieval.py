"""Memory retrieval candidate generation and ranking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import exp, log1p
from typing import cast

from alpha_agent.memory.models import (
    EpisodicMemory,
    MemoryRetrievalExplanation,
    MemoryScope,
    MemoryType,
    ProceduralMemory,
    RetrievedContext,
    SemanticMemory,
)
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.text import extract_lightweight_entities, keyword_score, tokenize
from alpha_agent.utils.time import utc_now


@dataclass(frozen=True)
class QueryExpansion:
    """Additional query terms with their source categories."""

    base_query: str
    terms: list[str]
    sources: list[str]

    @property
    def expanded_query(self) -> str:
        return " ".join([self.base_query, *self.terms]).strip()


@dataclass(frozen=True)
class RetrievalCandidate:
    """Memory candidate before final ranking."""

    memory: EpisodicMemory | SemanticMemory | ProceduralMemory
    memory_type: MemoryType
    channels: set[str] = field(default_factory=set)
    fts: float = 0.0


@dataclass(frozen=True)
class RetrievalCandidateSet:
    """Layer-separated candidates and the query expansion used to find them."""

    episodic: list[RetrievalCandidate]
    semantic: list[RetrievalCandidate]
    procedural: list[RetrievalCandidate]
    expansion: QueryExpansion

    def all(self) -> list[RetrievalCandidate]:
        return [*self.semantic, *self.episodic, *self.procedural]


@dataclass(frozen=True)
class ScoreBreakdown:
    """Normalized retrieval score components."""

    keyword: float
    fts: float
    recency: float
    salience: float
    stability: float
    access: float
    scope_priority: float
    status: float
    source_confidence: float

    def components(self) -> dict[str, float]:
        return {
            "keyword": self.keyword,
            "fts": self.fts,
            "recency": self.recency,
            "salience": self.salience,
            "stability": self.stability,
            "access": self.access,
            "scope_priority": self.scope_priority,
            "status": self.status,
            "source_confidence": self.source_confidence,
        }


@dataclass(frozen=True)
class RankedMemory:
    """Memory plus an explicit retrieval score and explanation."""

    memory: EpisodicMemory | SemanticMemory | ProceduralMemory
    memory_type: MemoryType
    score: float
    breakdown: ScoreBreakdown
    reasons: list[str]


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

        candidates = self.generate_candidates(
            query,
            session_id,
            limit=limit,
            scopes=scopes,
        )
        ranked = self.rank_candidates(candidates, query, scopes=scopes)
        episodic = self._select_ranked(ranked, "episodic", limit)
        semantic = self._select_ranked(ranked, "semantic", limit)
        procedural = self._select_ranked(ranked, "procedural", max(3, limit // 2))

        if record_access:
            for ranked in [*episodic, *semantic, *procedural]:
                self.store.log_memory_access(
                    ranked.memory.id,
                    ranked.memory_type,
                    query,
                    ranked.score,
                    scope=access_scope,
                    metadata={
                        "components": ranked.breakdown.components(),
                        "reasons": ranked.reasons,
                    },
                )

        entity_hints = _dedupe_strings(
            [*extract_lightweight_entities(query), *candidates.expansion.terms]
        )
        explanations = {
            self._explanation_key(item): MemoryRetrievalExplanation(
                memory_type=item.memory_type,
                memory_id=item.memory.id,
                total=item.score,
                components=item.breakdown.components(),
                reasons=list(item.reasons),
            )
            for item in [*episodic, *semantic, *procedural]
        }
        return RetrievedContext(
            episodic_memories=cast(list[EpisodicMemory], [item.memory for item in episodic]),
            semantic_memories=cast(list[SemanticMemory], [item.memory for item in semantic]),
            procedural_memories=cast(
                list[ProceduralMemory],
                [item.memory for item in procedural],
            ),
            entity_hints=entity_hints,
            retrieval_explanations=explanations,
        )

    def expand_query(
        self,
        query: str,
        session_id: str,
        *,
        scopes: list[MemoryScope] | None,
    ) -> QueryExpansion:
        """Expand a query from current entities, session state, and stable preferences."""

        terms: list[str] = []
        sources: list[str] = []
        for entity in extract_lightweight_entities(query):
            _append_term(terms, entity)
            _append_term(sources, "query_entity")

        state = self.store.get_session_context_state(session_id)
        if state is not None and state.summary:
            for term in [
                *extract_lightweight_entities(state.summary),
                *_task_state_terms(state.summary),
            ]:
                if _append_term(terms, term):
                    _append_term(sources, "session_state")

        for memory in self.store.list_semantic_memories(
            limit=50,
            scopes=scopes,
            statuses=["active"],
        ):
            if not _is_high_confidence_profile_or_preference(memory):
                continue
            for term in [*memory.entities, memory.object or ""]:
                if _append_term(terms, term):
                    _append_term(sources, "profile_preference")

        return QueryExpansion(
            base_query=query,
            terms=terms[:16],
            sources=sources,
        )

    def generate_candidates(
        self,
        query: str,
        session_id: str,
        *,
        limit: int,
        scopes: list[MemoryScope] | None,
    ) -> RetrievalCandidateSet:
        """Generate layer-separated candidates without applying final ranking."""

        expansion = self.expand_query(query, session_id, scopes=scopes)
        expanded_query = expansion.expanded_query
        multiplier = 4
        episodic = self._candidate_layer(
            "episodic",
            self.store.search_episodic(expanded_query, limit=limit * multiplier, scopes=scopes),
            self.store.list_episodic_memories(limit=limit * multiplier, scopes=scopes),
            fts_table="episodic_fts",
            query=expanded_query,
        )
        semantic = self._candidate_layer(
            "semantic",
            self.store.search_semantic(
                expanded_query,
                limit=limit * multiplier,
                scopes=scopes,
                statuses=["active"],
            ),
            self.store.list_semantic_memories(
                limit=limit * multiplier,
                scopes=scopes,
                statuses=["active"],
            ),
            fts_table="semantic_fts",
            query=expanded_query,
        )
        procedural = [
            candidate
            for candidate in self._candidate_layer(
                "procedural",
                self.store.search_procedural(
                    expanded_query,
                    limit=max(3, limit // 2) * multiplier,
                    scopes=scopes,
                ),
                self.store.list_procedural_memories(
                    limit=max(3, limit // 2) * multiplier,
                    scopes=scopes,
                ),
                fts_table="procedural_fts",
                query=expanded_query,
            )
            if isinstance(candidate.memory, ProceduralMemory)
            and self._procedural_text_relevance(expanded_query, candidate.memory) > 0
        ]
        return RetrievalCandidateSet(
            episodic=episodic,
            semantic=semantic,
            procedural=procedural,
            expansion=expansion,
        )

    def rank_candidates(
        self,
        candidates: RetrievalCandidateSet,
        query: str,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        """Rank generated candidates with auditable score components."""

        ranking_query = candidates.expansion.expanded_query or query
        ranked = [
            self._rank_candidate(candidate, ranking_query, scopes=scopes)
            for candidate in candidates.all()
            if self._is_prompt_eligible(candidate.memory, candidate.memory_type, scopes)
        ]
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ranked

    def _rank_episodic(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        candidates = self.generate_candidates(query, "", limit=limit, scopes=scopes)
        return self._select_ranked(
            self.rank_candidates(candidates, query, scopes=scopes),
            "episodic",
            limit,
        )

    def _rank_semantic(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        candidates = self.generate_candidates(query, "", limit=limit, scopes=scopes)
        return self._select_ranked(
            self.rank_candidates(candidates, query, scopes=scopes),
            "semantic",
            limit,
        )

    def _rank_procedural(
        self,
        query: str,
        limit: int,
        *,
        scopes: list[MemoryScope] | None,
    ) -> list[RankedMemory]:
        candidates = self.generate_candidates(query, "", limit=limit, scopes=scopes)
        return self._select_ranked(
            self.rank_candidates(candidates, query, scopes=scopes),
            "procedural",
            limit,
        )

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
        candidate = RetrievalCandidate(
            memory=memory,
            memory_type=cast(MemoryType, memory_type),
            channels={"direct"},
        )
        return self._rank_candidate(candidate, query, scopes=None).score

    def _candidate_layer(
        self,
        memory_type: MemoryType,
        searched: list[EpisodicMemory | SemanticMemory | ProceduralMemory],
        recent: list[EpisodicMemory | SemanticMemory | ProceduralMemory],
        *,
        fts_table: str,
        query: str,
    ) -> list[RetrievalCandidate]:
        fts_available = self.store.has_fts_index(fts_table) and bool(query.strip())
        by_id: dict[str, RetrievalCandidate] = {}
        for memory in searched:
            by_id[memory.id] = RetrievalCandidate(
                memory=memory,
                memory_type=memory_type,
                channels={"fts" if fts_available else "like"},
                fts=1.0 if fts_available else 0.0,
            )
        for memory in recent:
            existing = by_id.get(memory.id)
            if existing is None:
                by_id[memory.id] = RetrievalCandidate(
                    memory=memory,
                    memory_type=memory_type,
                    channels={"recent"},
                    fts=0.0,
                )
            else:
                by_id[memory.id] = RetrievalCandidate(
                    memory=existing.memory,
                    memory_type=memory_type,
                    channels={*existing.channels, "recent"},
                    fts=existing.fts,
                )
        return list(by_id.values())

    def _rank_candidate(
        self,
        candidate: RetrievalCandidate,
        query: str,
        *,
        scopes: list[MemoryScope] | None,
    ) -> RankedMemory:
        memory = candidate.memory
        breakdown = ScoreBreakdown(
            keyword=keyword_score(query, self._memory_text(memory)),
            fts=candidate.fts,
            recency=self._recency_score(getattr(memory, "updated_at", memory.created_at)),
            salience=float(getattr(memory, "salience", getattr(memory, "confidence", 0.5))),
            stability=float(getattr(memory, "stability", getattr(memory, "confidence", 0.6))),
            access=self._access_component(memory, candidate.memory_type),
            scope_priority=self._scope_priority(memory.scope, scopes),
            status=self._status_component(memory, candidate.memory_type),
            source_confidence=self._source_confidence(memory),
        )
        score = (
            breakdown.keyword * 0.30
            + breakdown.fts * 0.12
            + breakdown.recency * 0.16
            + breakdown.salience * 0.14
            + breakdown.stability * 0.10
            + breakdown.access * 0.08
            + breakdown.scope_priority * 0.08
            + breakdown.status * 0.06
            + breakdown.source_confidence * 0.06
        )
        return RankedMemory(
            memory=memory,
            memory_type=candidate.memory_type,
            score=score,
            breakdown=breakdown,
            reasons=self._reasons(breakdown, candidate.channels),
        )

    def _access_component(
        self,
        memory: EpisodicMemory | SemanticMemory | ProceduralMemory,
        memory_type: MemoryType,
    ) -> float:
        if isinstance(memory, EpisodicMemory):
            return self._access_score(memory.access_count)
        return self._access_score(self.store.count_memory_access(memory.id, memory_type))

    def _scope_priority(self, scope: MemoryScope, scopes: list[MemoryScope] | None) -> float:
        if not scopes:
            return 1.0
        keys = [item.scope_key for item in scopes]
        if scope.scope_key not in keys:
            return 0.0
        if len(keys) == 1:
            return 1.0
        return max(0.0, 1.0 - (keys.index(scope.scope_key) / (len(keys) - 1)) * 0.5)

    def _status_component(
        self,
        memory: EpisodicMemory | SemanticMemory | ProceduralMemory,
        memory_type: MemoryType,
    ) -> float:
        if memory_type == "semantic":
            return 1.0 if isinstance(memory, SemanticMemory) and memory.status == "active" else 0.0
        return 1.0

    def _source_confidence(
        self,
        memory: EpisodicMemory | SemanticMemory | ProceduralMemory,
    ) -> float:
        metadata_value = memory.metadata.get("source_confidence")
        if isinstance(metadata_value, int | float):
            return _clamp(float(metadata_value))
        return _clamp(float(getattr(memory, "confidence", 0.6)))

    def _reasons(self, breakdown: ScoreBreakdown, channels: set[str]) -> list[str]:
        reasons = []
        if breakdown.keyword > 0:
            reasons.append(f"keyword={breakdown.keyword:.2f}")
        if "fts" in channels:
            reasons.append("fts_match")
        if "like" in channels:
            reasons.append("like_match")
        if "recent" in channels:
            reasons.append(f"recency={breakdown.recency:.2f}")
        if breakdown.salience >= 0.7:
            reasons.append(f"salience={breakdown.salience:.2f}")
        if breakdown.stability >= 0.7:
            reasons.append(f"stability={breakdown.stability:.2f}")
        if breakdown.scope_priority > 0:
            reasons.append(f"scope_priority={breakdown.scope_priority:.2f}")
        if breakdown.source_confidence >= 0.7:
            reasons.append(f"source_confidence={breakdown.source_confidence:.2f}")
        return reasons or ["fallback_rank"]

    def _select_ranked(
        self,
        ranked: list[RankedMemory],
        memory_type: MemoryType,
        limit: int,
    ) -> list[RankedMemory]:
        return [item for item in ranked if item.memory_type == memory_type][:limit]

    def _is_prompt_eligible(
        self,
        memory: EpisodicMemory | SemanticMemory | ProceduralMemory,
        memory_type: MemoryType,
        scopes: list[MemoryScope] | None,
    ) -> bool:
        if scopes and memory.scope.scope_key not in {scope.scope_key for scope in scopes}:
            return False
        if memory_type == "semantic":
            return isinstance(memory, SemanticMemory) and memory.status == "active"
        return True

    def _explanation_key(self, ranked: RankedMemory) -> str:
        return f"{ranked.memory_type}:{ranked.memory.id}"

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

def _append_term(values: list[str], value: str) -> bool:
    normalized = " ".join(value.split()).strip()
    if not normalized or normalized.casefold() in {item.casefold() for item in values}:
        return False
    values.append(normalized)
    return True


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_term(result, value)
    return result


def _task_state_terms(summary: str) -> list[str]:
    terms: list[str] = []
    for raw in summary.replace("\n", ". ").split("."):
        line = raw.strip()
        lower = line.casefold()
        if any(marker in lower for marker in ("current task", "goal", "decision", "pending")):
            for token in tokenize(line):
                if len(token) >= 4:
                    terms.append(token)
    return terms[:8]


def _is_high_confidence_profile_or_preference(memory: SemanticMemory) -> bool:
    if memory.confidence < 0.85 or memory.stability < 0.75:
        return False
    if memory.memory_type in {"preference", "profile"}:
        return True
    return memory.predicate in {"prefers", "likes", "uses"}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
