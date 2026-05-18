from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.models import Event
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


def test_database_initialization(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    stats = store.stats()

    assert stats["events"] == 0
    assert stats["episodic"] == 0
    assert stats["semantic"] == 0


def test_insert_and_read_events(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    event = Event(
        id=new_id("evt"),
        session_id="s1",
        role="user",
        content="hello",
        created_at=utc_now_iso(),
        metadata={"source": "test"},
    )

    store.insert_event(event)
    events = store.list_events(session_id="s1")

    assert len(events) == 1
    assert events[0].content == "hello"
    assert events[0].metadata == {"source": "test"}


def test_semantic_upsert_merges_same_fact(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    first = semantic.upsert_fact(
        "user",
        "prefers",
        "concise answers",
        "User prefers concise answers",
    )
    second = semantic.upsert_fact("User", "Prefers", "Concise Answers", "Updated content")

    memories = store.list_semantic_memories()
    assert len(memories) == 1
    assert first.id == second.id
    assert memories[0].content == "Updated content"


def test_negative_preference_is_stored_as_dislike() -> None:
    from alpha_agent.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor()

    candidates = extractor.extract("I don't like verbose answers", "", ["evt1"])

    semantic = [candidate for candidate in candidates if candidate.type == "semantic"]
    assert semantic[0].predicate == "dislikes"
    assert semantic[0].object == "verbose answers"


def test_prune_low_priority_working_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    working.add_active_context("s1", "low", priority=0.1)
    working.add_active_context("s1", "high", priority=0.8)

    pruned = store.prune_low_priority_working_memory(priority_below=0.25)

    active = store.list_working_memory("s1")
    assert pruned == 1
    assert [item.content for item in active] == ["high"]
