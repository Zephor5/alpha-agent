from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.gateway.models import ConversationSource
from alpha_agent.gateway.session import GatewaySessionStore, SessionMode
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.store import MemoryStore


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


def _store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def test_debug_prompt_loads_gateway_source_from_session_and_prints_retrieval_trace(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    source = ConversationSource(
        platform="feishu",
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
        user_name="Ada",
        thread_id="thread-9",
    )
    mapping = GatewaySessionStore(store).get_or_create(source, SessionMode.THREAD_PER_USER)
    episode = EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision",
        source_event_ids=[],
        salience=0.9,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "debug",
            "prompt",
            "sqlite memory",
            "--session",
            mapping.session_id,
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert mapping.session_id in result.output
    assert "feishu" in result.output
    assert "chat-1" in result.output
    assert "user-1" in result.output
    assert "thread-9" in result.output
    assert episode.id in result.output
    assert "retrieval_score=" in result.output
    assert "access_count=" in result.output
    assert "access_count=1" not in result.output
    assert store.list_episodic_memories(limit=1)[0].access_count == 0
    with store.connect() as conn:
        access_logs = conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
    assert access_logs == 0


def test_memory_search_does_not_record_memory_access(tmp_path: Path) -> None:
    store = _store(tmp_path)
    EpisodicMemoryManager(store).create(
        content="SQLite memory retrieval decision",
        source_event_ids=[],
        salience=0.9,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["memory", "search", "sqlite memory"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "episodic" in result.output
    assert store.list_episodic_memories(limit=1)[0].access_count == 0
    with store.connect() as conn:
        access_logs = conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
    assert access_logs == 0


def test_skills_list_does_not_load_builtin_skills(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["skills", "list"], env=_env(tmp_path))

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert store.stats()["procedural"] == 0
    assert "Debug Loop" not in result.output


def test_debug_prompt_does_not_load_builtin_skills(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "debug this failing command"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert store.stats()["procedural"] == 0
    assert "Debug Loop" not in result.output


def test_debug_prompt_manual_source_flags_override_gateway_mapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    source = ConversationSource(
        platform="feishu",
        chat_id="chat-1",
        chat_type="group",
        user_id="user-1",
        thread_id="thread-9",
    )
    mapping = GatewaySessionStore(store).get_or_create(source, SessionMode.THREAD_PER_USER)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "debug",
            "prompt",
            "hello",
            "--session",
            mapping.session_id,
            "--platform",
            "telegram",
            "--chat-id",
            "chat-override",
        ],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "telegram" in result.output
    assert "chat-override" in result.output
    assert "thread-9" in result.output


def test_memory_review_preview_does_not_store_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["memory", "review", "remember that I prefer tea", "--session", "session-review"],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Candidate" in result.output
    assert "User prefers: tea" in result.output
    assert store.list_semantic_memories() == []
    assert store.list_episodic_memories() == []


def test_memory_review_approve_all_stores_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve-all",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Approved 2 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert len(store.list_episodic_memories()) == 1


def test_memory_review_reject_all_does_not_store_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--reject-all",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Rejected 2 candidate" in result.output
    assert store.list_semantic_memories() == []
    assert store.list_episodic_memories() == []


def test_memory_review_approve_one_unedited_candidate(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve",
            "1",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Approved 1 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert store.list_episodic_memories() == []


def test_memory_review_mixed_approve_and_reject_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "remember that I prefer tea",
            "--session",
            "session-review",
            "--approve",
            "1",
            "--reject",
            "2",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    assert result.exit_code == 0
    assert "Rejected 1 candidate" in result.output
    assert "Approved 1 candidate" in result.output
    assert [memory.object for memory in store.list_semantic_memories()] == ["tea"]
    assert store.list_episodic_memories() == []


def test_memory_review_edit_one_while_approving_selected_candidates(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "memory",
            "review",
            "I prefer coffee",
            "--session",
            "session-review",
            "--approve",
            "1",
            "--candidate",
            "1",
            "--edit-content",
            "User prefers: tea",
            "--edit-object",
            "tea",
        ],
        env=_env(tmp_path),
    )

    store = _store(tmp_path)
    semantic = store.list_semantic_memories()
    assert result.exit_code == 0
    assert "Edited candidate 1" in result.output
    assert "Approved 1 candidate" in result.output
    assert len(semantic) == 1
    assert semantic[0].content == "User prefers: tea"
    assert semantic[0].object == "tea"
