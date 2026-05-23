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
    status_responses: list[dict[str, Any]] = []
    stop_policies: list[str] = []

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(payload)
        if self.responses:
            return dict(self.responses.pop(0))
        return dict(self.response)

    def status(self) -> dict[str, Any]:
        if self.status_responses:
            return dict(self.status_responses.pop(0))
        return dict(self.response)

    def stop(self, *, policy: str = "graceful") -> dict[str, Any]:
        self.stop_policies.append(policy)
        return dict(self.response)


class _FakeProcess:
    def __init__(self, pid: int = 12345, returncode: int | None = None):
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


def _reset_fake_client() -> None:
    _FakeDaemonClient.requests = []
    _FakeDaemonClient.responses = []
    _FakeDaemonClient.status_responses = []
    _FakeDaemonClient.stop_policies = []
    _FakeDaemonClient.response = {"ok": True, "session_id": "s1", "response": "daemon response"}


def test_ask_sends_ipc_request_to_daemon(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
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
    _reset_fake_client()
    _FakeDaemonClient.response = {
        "ok": False,
        "error": {"code": "DAEMON_NOT_RUNNING", "message": "socket missing"},
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(app, ["ask", "hello"], env=_env(tmp_path))

    assert result.exit_code == 1
    assert "Daemon is not running. Run alpha daemon start." in result.output


def test_chat_sends_turns_and_consolidation_over_ipc(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
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
    _reset_fake_client()
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
    assert "Daemon is not running. Run alpha daemon start." in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "chat_turn",
            "message": "hello",
            "session_id": "s1",
            "source_metadata": {"channel": "cli", "command": "chat"},
        }
    ]


def test_daemon_start_spawns_background_run_and_waits_for_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    _FakeDaemonClient.status_responses = [
        {"ok": False, "error": {"code": "DAEMON_NOT_RUNNING", "message": "missing"}},
        {
            "ok": True,
            "status": {
                "running": True,
                "state": "running",
                "pid": 12345,
                "socket_path": str(tmp_path / "daemon.sock"),
                "status_path": str(tmp_path / "daemon-status.json"),
                "adapters": [],
            },
        },
    ]
    popen_calls: list[dict[str, Any]] = []

    def fake_popen(command, **kwargs):
        popen_calls.append({"command": command, **kwargs})
        return _FakeProcess(pid=12345)

    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    monkeypatch.setattr("alpha_agent.cli.subprocess.Popen", fake_popen)
    runner = CliRunner()

    result = runner.invoke(app, ["daemon", "start"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Daemon started" in result.output
    assert popen_calls
    command = popen_calls[0]["command"]
    assert command[-3:] == ["alpha_agent.cli", "daemon", "run"]
    assert popen_calls[0]["start_new_session"] is True


def test_daemon_start_does_not_spawn_when_already_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    _FakeDaemonClient.status_responses = [
        {
            "ok": True,
            "status": {
                "running": True,
                "state": "running",
                "pid": 12345,
                "socket_path": str(tmp_path / "daemon.sock"),
                "status_path": str(tmp_path / "daemon-status.json"),
                "adapters": [],
            },
        }
    ]
    popen_calls: list[Any] = []

    def fake_popen(command, **kwargs):
        popen_calls.append((command, kwargs))
        return _FakeProcess(pid=12345)

    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    monkeypatch.setattr("alpha_agent.cli.subprocess.Popen", fake_popen)
    runner = CliRunner()

    result = runner.invoke(app, ["daemon", "start"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Daemon is already running" in result.output
    assert popen_calls == []


def test_daemon_stop_supports_immediate_policy(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
    _FakeDaemonClient.response = {
        "ok": True,
        "status": {"message": "Daemon is stopping immediately."},
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(app, ["daemon", "stop", "--immediate"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Daemon is stopping immediately." in result.output
    assert _FakeDaemonClient.stop_policies == ["immediate"]
