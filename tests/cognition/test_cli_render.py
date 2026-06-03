from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_debug_prompt_previews_runtime_session_history_prompt(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["debug", "prompt", "hello"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Message 1 [system]" in result.output
    assert "Identity: Alpha Agent" in result.output
    assert "Message 2 [user]" in result.output
    assert "hello" in result.output


def test_cognition_graph_diff_and_evidence_commands(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    event_ids = id_factory()
    clock = clock_factory()
    emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={
            "turn_id": "turn_a",
            "session_id": "s1",
            "belief": belief("belief:old", "Old preference.").to_record(),
        },
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={
            "turn_id": "turn_b",
            "session_id": "s1",
            "belief": belief("belief:new", "New preference.").to_record(),
        },
        event_ids=event_ids,
        clock=clock,
    )

    runner = CliRunner()
    graph = runner.invoke(app, ["cognition", "graph", "--format", "mermaid"], env=_env(tmp_path))
    diff = runner.invoke(app, ["cognition", "diff", "turn_a", "turn_b"], env=_env(tmp_path))
    evidence = runner.invoke(app, ["cognition", "evidence", "belief:old"], env=_env(tmp_path))

    assert graph.exit_code == 0
    assert "graph TD" in graph.output
    assert diff.exit_code == 0
    assert "+ belief_formed:belief:new" in diff.output
    assert evidence.exit_code == 0
    assert "Evidence for belief:old" in evidence.output
    assert "belief_formed" in evidence.output


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }
