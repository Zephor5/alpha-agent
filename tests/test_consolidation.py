from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager


def test_consolidation_promotes_explicit_durable_facts(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    episodic = EpisodicMemoryManager(store)
    episodic.create(
        content="User said: remember that my favorite color is blue",
        source_event_ids=["evt1"],
        salience=0.9,
        confidence=0.8,
    )

    report = ConsolidationService(store).consolidate()
    semantic = store.list_semantic_memories()

    assert report.promoted_facts == 1
    assert len(semantic) == 1
    assert semantic[0].subject == "user.favorite_color"
    assert semantic[0].object == "blue"


def test_consolidation_prunes_low_priority_working_memory(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    working.add_active_context("s1", "low", priority=0.1)
    working.add_active_context("s1", "high", priority=0.8)

    report = ConsolidationService(store).consolidate()

    active = store.list_working_memory("s1")
    assert report.pruned_working_memory == 1
    assert [item.content for item in active] == ["high"]
