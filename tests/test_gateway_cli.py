from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.gateway.logging import GatewayLogContext, append_gateway_log, hash_identifier
from alpha_agent.gateway.models import ConversationSource, DeliveryResult, InboundMessage


def _set_gateway_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    db_path = tmp_path / "runtime" / "alpha.db"
    log_dir = tmp_path / "runtime" / "logs"
    status_path = tmp_path / "runtime" / "gateway-status.json"
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(tmp_path / "runtime" / "config.toml"))
    monkeypatch.setenv("ALPHA_LOG_DIR", str(log_dir))
    monkeypatch.setenv("ALPHA_GATEWAY_STATUS_PATH", str(status_path))
    monkeypatch.setenv("ALPHA_DAEMON_SOCKET_PATH", str(tmp_path / "runtime" / "daemon.sock"))
    monkeypatch.setenv(
        "ALPHA_DAEMON_STATUS_PATH",
        str(tmp_path / "runtime" / "daemon-status.json"),
    )
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    return db_path, log_dir, status_path


def test_gateway_status_reports_idle_without_status_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, log_dir, status_path = _set_gateway_env(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["gateway", "status"])

    assert result.exit_code == 0
    assert "idle" in result.output
    assert "not running" in result.output
    assert str(db_path) in result.output
    assert str(log_dir) in result.output
    assert not status_path.exists()


def test_gateway_doctor_initializes_db_and_runtime_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path, log_dir, _status_path = _set_gateway_env(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["gateway", "doctor"])

    assert result.exit_code == 0
    assert str(db_path) in result.output
    assert "gateway_session_mappings" in result.output
    assert "gateway_dedup" in result.output
    assert "mock" in result.output
    assert "No real platform adapters configured" in result.output
    assert db_path.exists()
    assert (log_dir / "agent.log").exists()
    assert (log_dir / "gateway.log").exists()
    assert (log_dir / "errors.log").exists()
    assert "gateway.doctor" in (log_dir / "gateway.log").read_text(encoding="utf-8")


def test_gateway_run_command_is_not_exposed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _db_path, _log_dir, status_path = _set_gateway_env(tmp_path, monkeypatch)
    runner = CliRunner()

    result = runner.invoke(app, ["gateway", "run", "--once"])

    assert result.exit_code != 0
    assert "No such command" in result.output
    assert not status_path.exists()


def test_gateway_log_context_hashes_external_ids(tmp_path: Path) -> None:
    log_path = tmp_path / "gateway.log"

    append_gateway_log(
        log_path,
        event="gateway.message.received",
        message="Inbound message normalized.",
        context=GatewayLogContext(
            session_id="session-1",
            platform="feishu",
            chat_id="chat-secret",
            user_id="user-secret",
        ),
    )

    entry = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert entry["context"]["session_id"] == "session-1"
    assert entry["context"]["platform"] == "feishu"
    assert entry["context"]["chat_id_hash"] == hash_identifier("chat-secret")
    assert entry["context"]["user_id_hash"] == hash_identifier("user-secret")
    assert "chat-secret" not in log_path.read_text(encoding="utf-8")
    assert "user-secret" not in log_path.read_text(encoding="utf-8")


def test_gateway_stop_command_is_not_exposed_without_pid_lock_support() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["gateway", "stop"])

    assert result.exit_code != 0


class _CliFakeAdapter:
    name = "cli-fake"

    def __init__(
        self,
        messages: list[InboundMessage],
        *,
        connect_error: Exception | None = None,
    ):
        self.messages = messages
        self.connect_error = connect_error
        self.connected = False
        self.disconnected = False
        self.sent: list[str] = []

    def connect(self, handler):
        self.connected = True
        if self.connect_error:
            raise self.connect_error
        for message in self.messages:
            handler(message)

    def disconnect(self) -> None:
        self.disconnected = True

    def send(self, source: ConversationSource, outbound):
        self.sent.append(outbound.text)
        return DeliveryResult(success=True, message_id="sent-1")

    def send_typing(self, source: ConversationSource) -> None:
        return None
