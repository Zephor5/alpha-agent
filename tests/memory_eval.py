from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import (
    ExtractedMemoryCandidate,
    MemoryScope,
    ProceduralMemory,
    SessionContextState,
)
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
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


@dataclass(frozen=True)
class MemoryBehaviorCase:
    case_id: str
    category: str
    user_message: str
    expected_candidates: tuple[ExpectedCandidate, ...]
    retrieval_query: str | None = None


@dataclass(frozen=True)
class SeededMemoryBehaviorFixture:
    session_id: str
    scope: MemoryScope
    semantic_ids: dict[str, str]
    procedural_ids: dict[str, str]
    retrieval_queries: dict[str, str]
    prompt_query: str


MEMORY_BEHAVIOR_FIXTURE_CASES: tuple[MemoryBehaviorCase, ...] = (
    MemoryBehaviorCase(
        case_id="preference-answer-style",
        category="preference",
        user_message="I prefer concise answers",
        expected_candidates=(
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="User prefers: concise answers",
                subject="user",
                predicate="prefers",
                object_value="concise answers",
                min_confidence=0.7,
            ),
        ),
        retrieval_query="answer style concise",
    ),
    MemoryBehaviorCase(
        case_id="fact-favorite-editor",
        category="fact",
        user_message="my favorite editor is neovim",
        expected_candidates=(
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="user.favorite_editor is neovim",
                subject="user.favorite_editor",
                predicate="is",
                object_value="neovim",
            ),
        ),
        retrieval_query="favorite editor neovim",
    ),
    MemoryBehaviorCase(
        case_id="correction-test-runner",
        category="correction",
        user_message="actually I prefer uv run pytest for test runs",
        expected_candidates=(
            ExpectedCandidate(
                candidate_type="semantic",
                content_contains="User prefers: uv run pytest for test runs",
                subject="user",
                predicate="prefers",
                object_value="uv run pytest for test runs",
                min_confidence=0.7,
            ),
        ),
        retrieval_query="test runner command",
    ),
    MemoryBehaviorCase(
        case_id="project-state-memory-phase",
        category="project_state",
        user_message=(
            "remember that Project Alpha Agent is finishing memory behavior "
            "fixture coverage"
        ),
        expected_candidates=(
            ExpectedCandidate(
                candidate_type="episodic",
                content_contains=(
                    "Project Alpha Agent is finishing memory behavior fixture coverage"
                ),
                min_confidence=0.7,
            ),
        ),
        retrieval_query="Alpha Agent memory fixture coverage",
    ),
    MemoryBehaviorCase(
        case_id="procedure-debug-tests",
        category="procedure_hint",
        user_message="when I ask you to debug tests, reproduce the failure first",
        expected_candidates=(
            ExpectedCandidate(
                candidate_type="procedural_candidate",
                content_contains="reproduce the failure first",
                subject="user",
                predicate="procedure",
                object_value="debug tests",
            ),
        ),
        retrieval_query="debug tests failure",
    ),
    MemoryBehaviorCase(
        case_id="do-not-remember-tea",
        category="do_not_remember",
        user_message="do not remember that I prefer tea",
        expected_candidates=(),
    ),
)


def assert_memory_behavior_extraction_cases(
    cases: tuple[MemoryBehaviorCase, ...] = MEMORY_BEHAVIOR_FIXTURE_CASES,
) -> None:
    for case in cases:
        candidates = assert_extraction_candidates(
            case.user_message,
            list(case.expected_candidates),
        )
        if not case.expected_candidates:
            assert candidates == [], (
                f"Expected no extracted candidates for {case.case_id}, got:\n"
                + "\n".join(_candidate_debug(candidate) for candidate in candidates)
            )


def seed_memory_behavior_fixture(
    store: MemoryStore,
    *,
    scope: MemoryScope | None = None,
    session_id: str = "memory-behavior-fixture-session",
) -> SeededMemoryBehaviorFixture:
    memory_scope = scope or MemoryScope.default()
    semantic = SemanticMemoryManager(store)
    preference = semantic.remember_atomic(
        subject="user.answer_style",
        predicate="prefers",
        object_value="concise answers",
        content="User prefers concise answers for routine replies",
        memory_type="preference",
        confidence=0.93,
        salience=0.88,
        stability=0.86,
        source_memory_ids=["fixture-msg-preference"],
        scope=memory_scope,
        metadata={"fixture_case": "preference-answer-style"},
    ).memory
    fact = semantic.remember_atomic(
        subject="user.favorite_editor",
        predicate="is",
        object_value="neovim",
        content="User favorite editor is neovim",
        memory_type="fact",
        confidence=0.94,
        salience=0.95,
        stability=0.95,
        source_memory_ids=["fixture-msg-fact"],
        scope=memory_scope,
        metadata={"fixture_case": "fact-favorite-editor"},
    ).memory
    old_correction = semantic.remember_atomic(
        subject="user.test_runner",
        predicate="uses",
        object_value="pytest",
        content="User test runner command was pytest",
        memory_type="fact",
        confidence=0.72,
        salience=0.66,
        stability=0.68,
        source_memory_ids=["fixture-msg-old-correction"],
        scope=memory_scope,
        metadata={"fixture_case": "correction-test-runner-old"},
    ).memory
    correction = semantic.remember_atomic(
        subject="user.test_runner",
        predicate="uses",
        object_value="uv run pytest",
        content="Correction: user test runner command is uv run pytest",
        memory_type="fact",
        confidence=0.91,
        salience=0.86,
        stability=0.7,
        source_memory_ids=["fixture-msg-correction"],
        scope=memory_scope,
        metadata={"fixture_case": "correction-test-runner"},
    ).memory
    project_state = semantic.remember_atomic(
        subject="project.alpha_agent",
        predicate="current_state",
        object_value="memory behavior fixture coverage",
        content="Project Alpha Agent is finishing memory behavior fixture coverage",
        memory_type="project_state",
        confidence=0.88,
        salience=0.9,
        stability=0.78,
        source_memory_ids=["fixture-msg-project-state"],
        scope=memory_scope,
        metadata={"fixture_case": "project-state-memory-phase"},
    ).memory

    now = "2026-01-01T00:00:00+00:00"
    procedure = store.upsert_procedural_memory(
        ProceduralMemory(
            id="fixture-proc-debug-tests",
            name="Debug tests fixture",
            description="Reproduce failing tests before editing",
            trigger="debug tests failure",
            procedure_markdown=(
                "1. Reproduce the failing test\n"
                "2. Inspect the smallest relevant code path\n"
                "3. Fix the root cause and rerun focused tests"
            ),
            success_count=0,
            failure_count=0,
            confidence=0.82,
            created_at=now,
            updated_at=now,
            metadata={"fixture_case": "procedure-debug-tests"},
            scope=memory_scope,
        )
    )
    store.upsert_session_context_state(
        SessionContextState(
            session_id=session_id,
            compressed_until_ordinal=1,
            summary="Current task: Project Alpha Agent memory behavior fixture coverage.",
            summary_source_message_ids=["fixture-msg-project-state"],
            compression_version="fixture",
            created_at=now,
            updated_at=now,
            metadata={"fixture": "memory_behavior"},
        )
    )

    return SeededMemoryBehaviorFixture(
        session_id=session_id,
        scope=memory_scope,
        semantic_ids={
            "preference": preference.id,
            "fact": fact.id,
            "correction_old": old_correction.id,
            "correction": correction.id,
            "project_state": project_state.id,
        },
        procedural_ids={"procedure_hint": procedure.id},
        retrieval_queries={
            "preference": "answer style concise",
            "fact": "favorite editor neovim",
            "correction": "test runner command uv run pytest",
            "project_state": "Alpha Agent memory fixture coverage",
            "procedure_hint": "debug tests failure",
        },
        prompt_query=(
            "Debug tests for Alpha Agent memory fixture coverage and keep the "
            "answer concise."
        ),
    )


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
