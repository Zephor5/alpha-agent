from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from alpha_agent.cli import app


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }


class _FakeDaemonClient:
    requests: list[dict[str, Any]] = []
    response: dict[str, Any] = {"ok": True, "session_id": "s1", "response": "daemon response"}
    responses: list[dict[str, Any]] = []

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        if self.responses:
            return dict(self.responses.pop(0))
        return dict(self.response)

    def status(self) -> dict[str, Any]:
        return dict(self.response)

    def stop(self) -> dict[str, Any]:
        return dict(self.response)


def test_ask_sends_ipc_request_to_daemon(tmp_path: Path, monkeypatch) -> None:
    _FakeDaemonClient.requests = []
    _FakeDaemonClient.responses = []
    _FakeDaemonClient.response = {"ok": True, "session_id": "s1", "response": "from daemon"}
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(app, ["ask", "hello"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "from daemon" in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "ask",
            "message": "hello",
            "session_id": None,
            "source_metadata": {"channel": "cli", "command": "ask"},
        }
    ]


def test_ask_reports_daemon_not_running(tmp_path: Path, monkeypatch) -> None:
    _FakeDaemonClient.requests = []
    _FakeDaemonClient.responses = []
    _FakeDaemonClient.response = {
        "ok": False,
        "error": {"code": "DAEMON_NOT_RUNNING", "message": "socket missing"},
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(app, ["ask", "hello"], env=_env(tmp_path))

    assert result.exit_code == 1
    assert "Daemon is not running. Run alpha daemon run." in result.output


def test_chat_sends_turns_and_consolidation_over_ipc(tmp_path: Path, monkeypatch) -> None:
    _FakeDaemonClient.requests = []
    _FakeDaemonClient.responses = [
        {"ok": True, "session_id": "daemon-s1", "response": "first response"},
        {"ok": True, "response": "consolidated"},
        {"ok": True, "session_id": "daemon-s2", "response": "second response"},
    ]
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", "local-s1"],
        input="hello\n/consolidate\nagain\n/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "first response" in result.output
    assert "consolidated" in result.output
    assert "second response" in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "chat_turn",
            "message": "hello",
            "session_id": "local-s1",
            "source_metadata": {"channel": "cli", "command": "chat"},
        },
        {"type": "consolidate_memory"},
        {
            "type": "chat_turn",
            "message": "again",
            "session_id": "daemon-s1",
            "source_metadata": {"channel": "cli", "command": "chat"},
        },
    ]


def test_chat_reports_daemon_not_running(tmp_path: Path, monkeypatch) -> None:
    _FakeDaemonClient.requests = []
    _FakeDaemonClient.responses = []
    _FakeDaemonClient.response = {
        "ok": False,
        "error": {"code": "DAEMON_NOT_RUNNING", "message": "socket missing"},
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", "s1"],
        input="hello\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Daemon is not running. Run alpha daemon run." in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "chat_turn",
            "message": "hello",
            "session_id": "s1",
            "source_metadata": {"channel": "cli", "command": "chat"},
        }
    ]
