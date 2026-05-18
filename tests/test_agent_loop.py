from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.memory.models import RetrievedContext
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.runtime.agent import AlphaAgent


def test_mock_agent_loop_stores_user_and_assistant_events(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    ProceduralMemoryManager(store).load_builtin_skills()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
    )

    result = agent.respond("remember that I prefer concise answers", session_id="s1")

    events = store.list_events(session_id="s1")
    semantic = store.list_semantic_memories()
    assert "Mock response" in result.response
    assert len(events) == 2
    assert len(semantic) == 1
    assert result.debug["extracted_memory_count"] >= 1


def test_agent_honors_configured_retrieval_limit(tmp_path: Path) -> None:
    class RecordingRetriever(MemoryRetriever):
        def __init__(self, store: MemoryStore, working: WorkingMemoryManager):
            super().__init__(store, working)
            self.seen_limit: int | None = None

        def retrieve_context(
            self,
            query: str,
            session_id: str,
            limit: int = 8,
        ) -> RetrievedContext:
            self.seen_limit = limit
            return super().retrieve_context(query, session_id, limit)

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    retriever = RecordingRetriever(store, working)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=retriever,
        retrieval_limit=2,
    )

    agent.respond("hello", session_id="s1")

    assert retriever.seen_limit == 2


def test_cli_basic_commands_with_mock_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "alpha.db"
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    runner = CliRunner()

    init_result = runner.invoke(app, ["init"])
    ask_result = runner.invoke(app, ["ask", "hello"])

    assert init_result.exit_code == 0
    assert ask_result.exit_code == 0
    assert "Mock response" in ask_result.output
