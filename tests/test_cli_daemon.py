from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from alpha_agent import cli
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


class _Stream:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


def test_chat_prompt_uses_width_aware_terminal_editor_for_tty(monkeypatch) -> None:
    calls: list[str] = []

    def fake_terminal_prompt(message: str) -> str:
        calls.append(message)
        return "中文"

    monkeypatch.setattr(cli, "_terminal_prompt", fake_terminal_prompt)
    monkeypatch.setattr(cli.sys, "stdin", _Stream(True))
    monkeypatch.setattr(cli.sys, "stdout", _Stream(True))

    assert cli._read_chat_message() == "中文"
    assert calls == ["You: "]


def test_chat_prompt_keeps_typer_prompt_for_non_tty(monkeypatch) -> None:
    def fake_typer_prompt(message: str) -> str:
        return f"typed through {message}"

    def fail_terminal_prompt(message: str) -> str:
        raise AssertionError(f"terminal prompt should not run for {message}")

    monkeypatch.setattr(cli, "_terminal_prompt", fail_terminal_prompt)
    monkeypatch.setattr(cli.typer, "prompt", fake_typer_prompt)
    monkeypatch.setattr(cli.sys, "stdin", _Stream(False))
    monkeypatch.setattr(cli.sys, "stdout", _Stream(False))

    assert cli._read_chat_message() == "typed through You"


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


def test_chat_sends_turns_over_ipc(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
    _FakeDaemonClient.responses = [
        {"ok": True, "session_id": "daemon-s1", "response": "first response"},
        {"ok": True, "session_id": "daemon-s2", "response": "second response"},
    ]
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", "local-s1"],
        input="hello\nagain\n/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "first response" in result.output
    assert "second response" in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "chat_turn",
            "message": "hello",
            "session_id": "local-s1",
            "source_metadata": {"channel": "cli", "command": "chat"},
        },
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


def test_daemon_restart_stops_running_daemon_then_starts_new_process(
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
                "pid": 11111,
                "socket_path": str(tmp_path / "daemon.sock"),
                "status_path": str(tmp_path / "daemon-status.json"),
                "adapters": [],
            },
        },
        {"ok": False, "error": {"code": "DAEMON_NOT_RUNNING", "message": "missing"}},
        {
            "ok": True,
            "status": {
                "running": True,
                "state": "running",
                "pid": 22222,
                "socket_path": str(tmp_path / "daemon.sock"),
                "status_path": str(tmp_path / "daemon-status.json"),
                "adapters": [],
            },
        },
    ]
    _FakeDaemonClient.response = {
        "ok": True,
        "status": {"message": "Daemon is draining the current request before stopping."},
    }
    popen_calls: list[dict[str, Any]] = []

    def fake_popen(command, **kwargs):
        popen_calls.append({"command": command, **kwargs})
        return _FakeProcess(pid=22222)

    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    monkeypatch.setattr("alpha_agent.cli.subprocess.Popen", fake_popen)
    runner = CliRunner()

    result = runner.invoke(app, ["daemon", "restart"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Daemon restarted with PID 22222." in result.output
    assert _FakeDaemonClient.stop_policies == ["graceful"]
    assert len(popen_calls) == 1


def test_daemon_restart_starts_when_daemon_is_not_running(
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

    result = runner.invoke(app, ["daemon", "restart"], env=_env(tmp_path))

    assert result.exit_code == 0
    assert "Daemon is not running; starting it." in result.output
    assert "Daemon started with PID 12345." in result.output
    assert _FakeDaemonClient.stop_policies == []
    assert len(popen_calls) == 1


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
