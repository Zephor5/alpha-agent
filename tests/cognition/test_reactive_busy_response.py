from __future__ import annotations

from datetime import timedelta

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore


def test_busy_response_writes_no_events_or_conversation(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())
    holder = LoopAcquireRequest(
        "consolidation",
        LoopPriority.CONSOLIDATION,
        timedelta(seconds=30),
    )

    with agent.coordinator.acquire(holder):
        result = agent.respond("hello", session_id="s1")

    assert result.response.startswith("Agent is currently")
    assert result.debug["busy"] is True
    assert result.debug["holder"] == "consolidation"
    assert result.debug["since"]
    assert list(SQLiteEventLog(store).iter()) == []
    assert store.list_conversation_messages("s1") == []
