from __future__ import annotations

from pathlib import Path

import pytest

from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import MemoryScope, SessionContextState
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from tests.memory_eval import assert_retrieves_ids, seed_memory_behavior_fixture


def test_insert_and_search_episodic_memories(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    episodic = EpisodicMemoryManager(store)
    episodic.create(
        content="User debugged a SQLite retrieval issue",
        source_event_ids=["evt1"],
        salience=0.8,
    )

    results = store.search_episodic("SQLite retrieval")

    assert len(results) == 1
    assert "SQLite" in results[0].content


def test_retrieval_ranking_without_vectors(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    episodic = EpisodicMemoryManager(store)
    semantic = SemanticMemoryManager(store)
    episodic.create("Casual chat about lunch", ["evt1"], salience=0.2)
    episodic.create("Important decision about SQLite memory retrieval", ["evt2"], salience=0.9)
    semantic.upsert_fact(
        "user",
        "prefers",
        "sqlite memory",
        "User prefers SQLite memory retrieval for the MVP",
        salience=0.85,
    )
    retriever = MemoryRetriever(store)

    context = retriever.retrieve_context("sqlite memory retrieval", "session-1", limit=3)

    assert context.semantic_memories[0].object == "sqlite memory"
    assert context.episodic_memories[0].summary.startswith("Important decision")


def test_retrieval_can_skip_access_recording(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    episode = EpisodicMemoryManager(store).create(
        "Important decision about SQLite memory retrieval",
        ["evt1"],
        salience=0.9,
    )
    retriever = MemoryRetriever(store)

    context = retriever.retrieve_context(
        "sqlite memory retrieval",
        "session-1",
        limit=3,
        record_access=False,
    )

    assert [memory.id for memory in context.episodic_memories] == [episode.id]
    assert store.list_episodic_memories(limit=1)[0].access_count == 0
    with store.connect() as conn:
        access_logs = conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
    assert access_logs == 0


def test_procedural_retrieval_requires_textual_relevance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    ProceduralMemoryManager(store).load_builtin_skills()
    retriever = MemoryRetriever(store)

    unrelated = retriever.retrieve_context("What tools do you have?", "session-1", limit=3)
    matched = retriever.retrieve_context("debug this failing command", "session-1", limit=3)

    assert unrelated.procedural_memories == []
    assert [memory.name for memory in matched.procedural_memories] == ["Debug Loop"]


def test_retrieval_filters_same_query_by_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    alice_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:alice",
        platform="telegram",
        user_id="alice",
    )
    bob_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:bob",
        platform="telegram",
        user_id="bob",
    )
    alice = semantic.upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers tea",
        salience=0.9,
        scope=alice_scope,
    )
    bob = semantic.upsert_fact(
        "user",
        "prefers",
        "coffee",
        "User prefers coffee",
        salience=0.9,
        scope=bob_scope,
    )

    assert_retrieves_ids(
        store,
        query="what does the user prefer",
        scope=alice_scope,
        expected_semantic_ids=[alice.id],
    )
    assert_retrieves_ids(
        store,
        query="what does the user prefer",
        scope=bob_scope,
        expected_semantic_ids=[bob.id],
    )


def test_shared_channel_scope_does_not_read_default_or_platform_user_memory(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    default_scope = MemoryScope.default()
    platform_user_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:alice",
        platform="telegram",
        user_id="alice",
    )
    shared_channel_scope = MemoryScope(
        kind="chat_thread",
        scope_key="platform:telegram:chat:shared-chat:thread:main",
        platform="telegram",
        chat_id="shared-chat",
        thread_id=None,
        user_id=None,
    )
    semantic.upsert_fact(
        "user",
        "prefers",
        "default tea",
        "User prefers default tea",
        salience=0.9,
        scope=default_scope,
    )
    semantic.upsert_fact(
        "user",
        "prefers",
        "private coffee",
        "User prefers private coffee",
        salience=0.9,
        scope=platform_user_scope,
    )
    shared = semantic.upsert_fact(
        "channel",
        "prefers",
        "shared agenda",
        "Channel prefers shared agenda",
        salience=0.9,
        scope=shared_channel_scope,
    )

    assert [scope.scope_key for scope in shared_channel_scope.allowed_read_scopes()] == [
        shared_channel_scope.scope_key,
    ]

    assert_retrieves_ids(
        store,
        query="what does the user or channel prefer",
        scope=shared_channel_scope,
        expected_semantic_ids=[shared.id],
    )


def test_retrieval_filters_inactive_semantic_status(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    active = semantic.upsert_fact(
        "user",
        "prefers",
        "concise answers",
        "User prefers concise answers",
    )
    semantic.upsert_fact(
        "user",
        "prefers",
        "stale answers",
        "User prefers stale answers",
        status="deleted",
    )

    context = MemoryRetriever(store).retrieve_context(
        "user preferences answers",
        "session-1",
        scopes=MemoryScope.default().allowed_read_scopes(),
        record_access=False,
    )

    assert [memory.id for memory in context.semantic_memories] == [active.id]


def test_retrieval_uses_corrected_active_memory_only(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    semantic.upsert_fact("user", "prefers", "tea", "User prefers tea")
    corrected = semantic.upsert_fact(
        "user",
        "prefers",
        "coffee",
        "User now prefers coffee",
    )

    context = MemoryRetriever(store).retrieve_context(
        "what does the user prefer",
        "session-1",
        scopes=MemoryScope.default().allowed_read_scopes(),
        record_access=False,
    )

    assert [memory.id for memory in context.semantic_memories] == [corrected.id]
    assert "tea" not in [memory.object for memory in context.semantic_memories]


def test_retrieval_splits_candidates_from_ranking_with_score_breakdown(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    memory = semantic.remember_atomic(
        subject="user",
        predicate="prefers",
        object_value="concise retrieval explanations",
        content="User prefers concise retrieval explanations with source confidence",
        memory_type="preference",
        confidence=0.91,
        salience=0.86,
        stability=0.82,
        source_memory_ids=["msg-source"],
        metadata={"source_confidence": 0.77},
    ).memory
    retriever = MemoryRetriever(store)

    candidates = retriever.generate_candidates(
        "retrieval explanations",
        "session-1",
        limit=5,
        scopes=MemoryScope.default().allowed_read_scopes(),
    )
    ranked = retriever.rank_candidates(
        candidates,
        "retrieval explanations",
        scopes=MemoryScope.default().allowed_read_scopes(),
    )
    context = retriever.retrieve_context(
        "retrieval explanations",
        "session-1",
        limit=5,
        scopes=MemoryScope.default().allowed_read_scopes(),
        record_access=True,
    )

    assert candidates.semantic
    assert ranked[0].memory.id == memory.id
    assert ranked[0].breakdown.keyword > 0
    assert ranked[0].breakdown.fts >= 0
    assert ranked[0].breakdown.recency > 0
    assert ranked[0].breakdown.salience == memory.salience
    assert ranked[0].breakdown.stability == memory.stability
    assert ranked[0].breakdown.access == 0
    assert ranked[0].breakdown.scope_priority == 1
    assert ranked[0].breakdown.status == 1
    assert ranked[0].breakdown.source_confidence == 0.77
    assert any("keyword" in reason for reason in ranked[0].reasons)
    explanation = context.retrieval_explanations[f"semantic:{memory.id}"]
    assert explanation.total == pytest.approx(ranked[0].score)
    assert explanation.components["source_confidence"] == 0.77
    assert explanation.reasons
    with store.connect() as conn:
        row = conn.execute(
            "SELECT metadata FROM memory_access_log WHERE memory_id = ?",
            (memory.id,),
        ).fetchone()
    assert "source_confidence" in row["metadata"]


def test_memory_behavior_fixture_explains_retrieval_cases(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    fixture = seed_memory_behavior_fixture(store)

    assert_retrieves_ids(
        store,
        query=fixture.retrieval_queries["preference"],
        scope=fixture.scope,
        expected_semantic_ids=[fixture.semantic_ids["preference"]],
    )
    assert_retrieves_ids(
        store,
        query=fixture.retrieval_queries["fact"],
        scope=fixture.scope,
        expected_semantic_ids=[fixture.semantic_ids["fact"]],
    )
    assert_retrieves_ids(
        store,
        query=fixture.retrieval_queries["correction"],
        scope=fixture.scope,
        expected_semantic_ids=[fixture.semantic_ids["correction"]],
    )
    assert_retrieves_ids(
        store,
        query=fixture.retrieval_queries["project_state"],
        scope=fixture.scope,
        expected_semantic_ids=[fixture.semantic_ids["project_state"]],
    )

    context = MemoryRetriever(store).retrieve_context(
        fixture.retrieval_queries["procedure_hint"],
        fixture.session_id,
        scopes=fixture.scope.allowed_read_scopes(),
        record_access=False,
    )

    assert [memory.id for memory in context.procedural_memories] == [
        fixture.procedural_ids["procedure_hint"]
    ]
    assert fixture.semantic_ids["correction_old"] not in [
        memory.id for memory in context.semantic_memories
    ]


def test_query_expansion_uses_session_state_entities_and_profile_preferences(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    semantic.remember_atomic(
        subject="user",
        predicate="prefers",
        object_value="SQLite",
        content="User prefers SQLite for local memory storage",
        memory_type="preference",
        confidence=0.95,
        stability=0.9,
    )
    store.upsert_session_context_state(
        SessionContextState(
            session_id="session-1",
            compressed_until_ordinal=1,
            summary="Current task: Project Atlas retrieval cleanup.",
            summary_source_message_ids=["msg-1"],
            compression_version="test",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    expansion = MemoryRetriever(store).expand_query(
        "what should we use?",
        "session-1",
        scopes=MemoryScope.default().allowed_read_scopes(),
    )

    assert any("Atlas" in term for term in expansion.terms)
    assert "SQLite" in expansion.terms
    assert "session_state" in expansion.sources
    assert "profile_preference" in expansion.sources
