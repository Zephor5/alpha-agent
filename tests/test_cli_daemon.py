from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Any

from typer.testing import CliRunner

from alpha_agent import cli
from alpha_agent.cli import app
from alpha_agent.daemon.conversation_import import ConversationImportService
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


def test_chat_renders_current_turn_tool_rounds(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()

    class WritingFakeClient:
        def __init__(self, socket_path: Path):
            self.socket_path = socket_path

        def request(self, payload: dict[str, Any]) -> dict[str, Any]:
            session_id = str(payload["session_id"])
            store.append_session_message(
                session_id=session_id,
                kind="user_message",
                llm_role="user",
                raw_content=str(payload["message"]),
            )
            store.append_session_message(
                session_id=session_id,
                kind="assistant_message",
                llm_role="assistant",
                raw_content="I will check the first source.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"first"}',
                        },
                    }
                ],
            )
            store.append_session_message(
                session_id=session_id,
                kind="tool_message",
                llm_role="tool",
                raw_content='{"result":"first"}',
                tool_call_id="call_1",
                provider_metadata={"tool_name": "lookup"},
            )
            store.append_session_message(
                session_id=session_id,
                kind="assistant_message",
                llm_role="assistant",
                raw_content="I will verify with the second source.",
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"second"}',
                        },
                    }
                ],
            )
            store.append_session_message(
                session_id=session_id,
                kind="tool_message",
                llm_role="tool",
                raw_content='{"result":"second"}',
                tool_call_id="call_2",
                provider_metadata={"tool_name": "lookup"},
            )
            store.append_session_message(
                session_id=session_id,
                kind="assistant_message",
                llm_role="assistant",
                raw_content="final answer",
            )
            return {"ok": True, "session_id": session_id, "response": "final answer"}

    monkeypatch.setattr("alpha_agent.cli.DaemonClient", WritingFakeClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat"],
        input="hello\n/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "I will check the first source." in result.output
    assert "Tool call: lookup" in result.output
    assert '{"query":"first"}' in result.output
    assert "Tool result: lookup" in result.output
    assert '{"result":"first"}' in result.output
    assert "I will verify with the second source." in result.output
    assert '{"query":"second"}' in result.output
    assert '{"result":"second"}' in result.output
    assert "final answer" in result.output
    assert result.output.count("final answer") == 1


def test_chat_progress_renders_before_ipc_response_returns(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    progress_rendered = Event()
    progress_seen_before_response: list[bool] = []
    rendered_contents: list[str] = []

    class BlockingFakeClient:
        def __init__(self, socket_path: Path):
            self.socket_path = socket_path

        def request(self, payload: dict[str, Any]) -> dict[str, Any]:
            session_id = str(payload["session_id"])
            store.append_session_message(
                session_id=session_id,
                kind="user_message",
                llm_role="user",
                raw_content=str(payload["message"]),
            )
            store.append_session_message(
                session_id=session_id,
                kind="assistant_message",
                llm_role="assistant",
                raw_content="I am checking that now.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"progress"}',
                        },
                    }
                ],
            )
            progress_seen_before_response.append(progress_rendered.wait(timeout=1.0))
            store.append_session_message(
                session_id=session_id,
                kind="assistant_message",
                llm_role="assistant",
                raw_content="done",
            )
            return {"ok": True, "session_id": session_id, "response": "done"}

    def capture_rendered_messages(
        messages: list[Any],
        *,
        fallback_response: str,
    ) -> None:
        for message in messages:
            content = cli._chat_turn_content(message)
            rendered_contents.append(content)
            if "I am checking that now." in content:
                progress_rendered.set()

    monkeypatch.setattr(cli, "_render_chat_turn_messages", capture_rendered_messages)
    monkeypatch.setattr(cli, "CHAT_TURN_PROGRESS_POLL_INTERVAL_SECONDS", 0.01)

    response, rendered_after_ordinal = cli._request_chat_turn_with_progress(
        BlockingFakeClient(tmp_path / "daemon.sock"),
        {"type": "chat_turn", "message": "hello", "session_id": "s1"},
        store=store,
        session_id="s1",
        after_ordinal=0,
    )

    assert response == {"ok": True, "session_id": "s1", "response": "done"}
    assert progress_seen_before_response == [True]
    assert any("I am checking that now." in content for content in rendered_contents)
    assert rendered_after_ordinal == 3


def test_chat_with_existing_session_renders_recent_history(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_message(
        session_id="local-s1",
        kind="user_message",
        llm_role="user",
        raw_content="old user message",
    )
    store.append_session_message(
        session_id="local-s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="old assistant message",
    )
    store.append_session_message(
        session_id="local-s1",
        kind="user_message",
        llm_role="user",
        raw_content="recent user message",
    )
    store.append_session_message(
        session_id="local-s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="recent assistant message",
    )
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    monkeypatch.setattr(cli, "CHAT_HISTORY_PREVIEW_LIMIT", 2)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", "local-s1"],
        input="/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Alpha Chat" in result.output
    assert "local-s1" in result.output
    assert "Recent Session Context" in result.output
    assert "recent user message" in result.output
    assert "recent assistant message" in result.output
    assert "old user message" not in result.output
    assert "old assistant message" not in result.output
    assert _FakeDaemonClient.requests == []


def test_chat_history_table_draws_message_separators(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_message(
        session_id="local-s1",
        kind="user_message",
        llm_role="user",
        raw_content="first",
    )
    store.append_session_message(
        session_id="local-s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="second",
    )

    table = cli._build_chat_history_table(
        cli._displayable_session_messages(store, "local-s1")
    )

    assert table.show_lines is True


def test_chat_with_existing_session_renders_tool_round(tmp_path: Path, monkeypatch) -> None:
    _reset_fake_client()
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_session_message(
        session_id="local-s1",
        kind="user_message",
        llm_role="user",
        raw_content="check docs",
    )
    store.append_session_message(
        session_id="local-s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
            }
        ],
    )
    store.append_session_message(
        session_id="local-s1",
        kind="tool_message",
        llm_role="tool",
        raw_content='{"ok": true}',
        tool_call_id="call_1",
        provider_metadata={"tool_name": "lookup"},
    )
    store.append_session_message(
        session_id="local-s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="done",
    )
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", "local-s1"],
        input="/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Tool call: lookup" in result.output
    assert '{"query":"alpha"}' in result.output
    assert "Tool result: lookup" in result.output
    assert '{"ok": true}' in result.output
    assert _FakeDaemonClient.requests == []


def test_chat_with_import_session_rejects_without_rendering_history(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    ConversationImportService(store).import_payload(
        '{"source_provider":"chatgpt","conversations":[{"external_conversation_id":"conv_1",'
        '"messages":[{"external_message_id":"msg_1","role":"user",'
        '"content":"DO_NOT_SHOW_IMPORTED_HISTORY",'
        '"created_at":"2026-01-01T00:00:00Z"}]}]}',
        input_name="external.json",
    )
    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["chat", "--session", imported.session_id],
        input="/exit\n",
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Import sessions are hidden source material" in result.output
    assert "DO_NOT_SHOW_IMPORTED_HISTORY" not in result.output
    assert _FakeDaemonClient.requests == []


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


def test_cognition_import_conversations_sends_filename_only_and_renders_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    payload_path = tmp_path / "nested" / "external.json"
    payload_path.parent.mkdir()
    payload_path.write_text('{"source_provider":"chatgpt","conversations":[]}', encoding="utf-8")
    _FakeDaemonClient.response = {
        "ok": True,
        "summary": {
            "batch_id": None,
            "source_provider": "chatgpt",
            "dry_run": True,
            "status": "completed",
            "input_name": "external.json",
            "conversations_seen": 1,
            "messages_seen": 3,
            "conversations_created": 1,
            "conversations_reused": 0,
            "messages_inserted": 3,
            "messages_deduped": 0,
        },
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "conversations", str(payload_path), "--dry-run"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Conversation Import" in result.output
    assert "dry_run=true" in result.output
    assert "source_provider=chatgpt" in result.output
    assert "messages_inserted=3" in result.output
    assert "background_cognition=eligible" in result.output
    assert "background_cognition=available" not in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "conversation_import",
            "input_name": "external.json",
            "payload_json": '{"source_provider":"chatgpt","conversations":[]}',
            "dry_run": True,
        }
    ]
    assert str(payload_path) not in str(_FakeDaemonClient.requests[0]["input_name"])


def test_cognition_import_conversations_rejects_large_file_before_ipc(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    payload_path = tmp_path / "large.json"
    payload_path.write_text("12345", encoding="utf-8")
    monkeypatch.setattr("alpha_agent.cli.MAX_CONVERSATION_IMPORT_PAYLOAD_BYTES", 4)
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "conversations", str(payload_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "exceeds the 50 MB conversation import limit" in result.output
    assert _FakeDaemonClient.requests == []


def test_cognition_import_conversations_reports_daemon_not_running(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    payload_path = tmp_path / "external.json"
    payload_path.write_text("{}", encoding="utf-8")
    _FakeDaemonClient.response = {
        "ok": False,
        "error": {"code": "DAEMON_NOT_RUNNING", "message": "socket missing"},
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "conversations", str(payload_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Daemon is not running. Run alpha daemon start." in result.output


def test_cognition_import_conversations_renders_validation_details(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    payload_path = tmp_path / "bad.json"
    payload_path.write_text("{}", encoding="utf-8")
    _FakeDaemonClient.response = {
        "ok": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Invalid conversation import payload.",
            "details": [
                {
                    "path": "conversations[0].messages[0].created_at",
                    "message": "timestamp must include an explicit timezone offset or Z",
                    "code": "naive_timestamp",
                }
            ],
        },
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "conversations", str(payload_path)],
        env=_env(tmp_path),
    )

    assert result.exit_code == 1
    assert "Invalid conversation import payload." in result.output
    assert "conversations[0].messages[0].created_at" in result.output
    assert "timestamp must include an explicit timezone offset or Z" in result.output


def test_cognition_import_status_renders_summary_and_verbose_conversations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _reset_fake_client()
    _FakeDaemonClient.response = {
        "ok": True,
        "status": {
            "batch_id": "import_batch_1",
            "source_provider": "chatgpt",
            "status": "completed",
            "conversations_seen": 1,
            "messages_seen": 3,
            "conversations_created": 1,
            "conversations_reused": 0,
            "messages_inserted": 3,
            "messages_deduped": 0,
            "extraction_pending": 2,
            "extraction_claimed": 0,
            "extraction_processed": 1,
            "extraction_failed": 0,
            "extraction_skipped": 0,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "error_summary": None,
        },
        "conversations": [
            {
                "external_conversation_id": "conv_1",
                "title": "Imported design discussion",
                "session_id": "session_hidden",
                "messages_inserted": 3,
                "messages_deduped": 0,
                "session_reused": False,
                "extraction_pending": 2,
                "extraction_claimed": 0,
                "extraction_processed": 1,
                "extraction_failed": 0,
                "extraction_skipped": 0,
            }
        ],
    }
    monkeypatch.setattr("alpha_agent.cli.DaemonClient", _FakeDaemonClient)
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["cognition", "import", "status", "import_batch_1", "--verbose"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "Conversation Import Status" in result.output
    assert "batch_id=import_batch_1" in result.output
    assert "extraction_pending=2" in result.output
    assert "conv_1" in result.output
    assert "session_hidden" in result.output
    assert _FakeDaemonClient.requests == [
        {
            "type": "conversation_import_status",
            "batch_id": "import_batch_1",
            "verbose": True,
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
