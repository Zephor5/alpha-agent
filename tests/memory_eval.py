from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import ExtractedMemoryCandidate, MemoryScope
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class ExpectedCandidate:
    candidate_type: str
    content_contains: str
    subject: str | None = None
    predicate: str | None = None
    object_value: str | None = None
    min_confidence: float = 0.0


@dataclass(frozen=True)
class RetrievalRow:
    memory_type: str
    memory_id: str
    score: float
    keyword: float
    salience: float
    access: float
    scope_key: str
    content: str


def assert_extraction_candidates(
    message: str,
    expected: list[ExpectedCandidate],
) -> list[ExtractedMemoryCandidate]:
    candidates = MemoryExtractor().extract(message, "", ["source-1"])
    missing: list[str] = []
    for item in expected:
        match = next(
            (
                candidate
                for candidate in candidates
                if candidate.type == item.candidate_type
                and item.content_contains in candidate.content
                and (item.subject is None or candidate.subject == item.subject)
                and (item.predicate is None or candidate.predicate == item.predicate)
                and (item.object_value is None or candidate.object == item.object_value)
                and candidate.confidence >= item.min_confidence
            ),
            None,
        )
        if match is None:
            missing.append(repr(item))
    assert not missing, (
        "Missing extracted candidates:\n"
        + "\n".join(missing)
        + "\nActual candidates:\n"
        + "\n".join(_candidate_debug(candidate) for candidate in candidates)
    )
    return candidates


def assert_retrieves_ids(
    store: MemoryStore,
    *,
    query: str,
    scope: MemoryScope,
    expected_semantic_ids: list[str],
) -> None:
    retriever = MemoryRetriever(store)
    context = retriever.retrieve_context(
        query,
        "eval-session",
        scopes=scope.allowed_read_scopes(),
        record_access=False,
    )
    actual = [memory.id for memory in context.semantic_memories]
    rows = [
        RetrievalRow(
            memory_type=ranked.memory_type,
            memory_id=ranked.memory.id,
            score=ranked.score,
            keyword=retriever._score(query, ranked.memory, ranked.memory_type),
            salience=getattr(ranked.memory, "salience", getattr(ranked.memory, "confidence", 0)),
            access=getattr(ranked.memory, "access_count", 0),
            scope_key=ranked.memory.scope.scope_key,
            content=getattr(ranked.memory, "content", ""),
        )
        for ranked in retriever._rank_semantic(
            query,
            20,
            scopes=scope.allowed_read_scopes(),
        )
    ]
    assert actual[: len(expected_semantic_ids)] == expected_semantic_ids, (
        f"Expected semantic ids {expected_semantic_ids}, got {actual}.\n"
        + "\n".join(str(row) for row in rows)
    )


def _candidate_debug(candidate: ExtractedMemoryCandidate) -> str:
    return (
        f"type={candidate.type} content={candidate.content!r} "
        f"subject={candidate.subject!r} predicate={candidate.predicate!r} "
        f"object={candidate.object!r} confidence={candidate.confidence}"
    )
