from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager


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
    retriever = MemoryRetriever(store, WorkingMemoryManager(store))

    context = retriever.retrieve_context("sqlite memory retrieval", "session-1", limit=3)

    assert context.semantic_memories[0].object == "sqlite memory"
    assert context.episodic_memories[0].summary.startswith("Important decision")


def test_procedural_retrieval_requires_textual_relevance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    ProceduralMemoryManager(store).load_builtin_skills()
    retriever = MemoryRetriever(store, WorkingMemoryManager(store))

    unrelated = retriever.retrieve_context("What tools do you have?", "session-1", limit=3)
    matched = retriever.retrieve_context("debug this failing command", "session-1", limit=3)

    assert unrelated.procedural_memories == []
    assert [memory.name for memory in matched.procedural_memories] == ["Debug Loop"]
