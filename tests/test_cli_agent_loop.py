from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


def test_init_creates_state_database_without_loading_long_term_records(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["init"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Initialized Alpha Agent database" in result.output
    with StateStore(tmp_path / "alpha.db").connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert tables == {
            "conversation_messages",
            "runtime_traces",
            "gateway_session_mappings",
            "gateway_dedup",
            "cognitive_events",
            "counterpart_view",
            "belief_view",
            "belief_entity_index",
            "belief_about_index",
            "context_window_view",
            "context_window_background",
            "reflection_view",
            "procedure_view",
            "cognition_worker_checkpoint",
        }


def test_debug_prompt_renders_minimal_prompt_for_existing_session(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_conversation_message(session_id="s1", role="user", raw_content="hello")
    store.append_conversation_message(session_id="s1", role="assistant", raw_content="hi")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Message 1 [system]" in result.output
    assert "Identity: Alpha Agent" in result.output
    assert "hello" in result.output
    assert "hi" in result.output
    assert "continue" in result.output


def test_debug_prompt_trace_renders_recent_cognitive_events(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    AlphaAgent(store=store, llm_provider=MockLLMProvider()).respond("hello", session_id="s1")
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["debug", "prompt", "continue", "--session", "s1", "--trace"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Cognitive Trace" in result.output
    for kind in [
        "perceived",
        "attended",
        "interpreted",
        "judged",
        "decided",
        "acted",
        "received_feedback",
        "reflected",
        "revised",
    ]:
        assert f"kind={kind}" in result.output
    assert "tick_id=" in result.output


def test_skills_list_reads_builtin_skills_without_state_store(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["skills", "list"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Skill:" in result.output
