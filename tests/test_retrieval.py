from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import MemoryScope
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from tests.memory_eval import assert_retrieves_ids


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
