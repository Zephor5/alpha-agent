from __future__ import annotations

from datetime import timedelta

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind, LoopPriority
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore


def test_busy_rejected_new_source_is_not_observed_until_later_success(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())
    holder = LoopAcquireRequest(
        "consolidation",
        LoopPriority.CONSOLIDATION,
        timedelta(seconds=30),
    )
    source = {"platform": "test", "user_id": "u-new"}

    with agent.coordinator.acquire(holder):
        busy = agent.respond("hello", session_id="s1", source_metadata=source)

    assert busy.debug["busy"] is True
    assert list(SQLiteEventLog(store).iter()) == []

    agent.respond("hello again", session_id="s1", source_metadata=source)
    first_observed = [
        event
        for event in SQLiteEventLog(store).iter(
            kinds=[CognitiveEventKind.COUNTERPART_FIRST_OBSERVED]
        )
    ]
    assert len(first_observed) == 1
