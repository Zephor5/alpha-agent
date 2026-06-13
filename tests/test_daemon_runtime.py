from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import MethodType

import pytest

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.config import AlphaConfig, CognitionBackgroundConfig
from alpha_agent.daemon.runtime import AlphaDaemon, DaemonAlreadyRunningError
from alpha_agent.daemon.status import DaemonRuntimeConfig, running_status, write_daemon_status
from alpha_agent.gateway.runner import ActiveTurnGuard
from alpha_agent.state.store import StateStore


class _AgentResult:
    def __init__(self, response: str, session_id: str):
        self.response = response
        self.session_id = session_id


class _FakeAgent:
    def __init__(self):
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def respond(
        self,
        message: str,
        *,
        session_id: str,
        source_metadata: dict[str, object] | None = None,
    ) -> _AgentResult:
        self.calls.append((message, session_id, source_metadata or {}))
        return _AgentResult(f"response to {message}", session_id)


class _FakeManager:
    def __init__(self, agent: _FakeAgent):
        self.agent = agent
        self.session_ids: list[str] = []

    def get_or_create(self, session_id: str) -> _FakeAgent:
        self.session_ids.append(session_id)
        return self.agent

    def evict_all(self) -> None:
        return None


class _FailingAdapter:
    name = "failing"

    def __init__(self):
        self.connected = False
        self.disconnected = False

    def connect(self, _handler) -> None:
        self.connected = True
        raise RuntimeError("connect failed")

    def disconnect(self) -> None:
        self.disconnected = True

    def send(self, _source, _outbound):
        raise AssertionError("send should not be called")

    def send_typing(self, _source) -> None:
        return None


class _RecordingBackgroundWake:
    def __init__(self) -> None:
        self.wake_calls = 0

    def wake(self) -> bool:
        self.wake_calls += 1
        return True

    def tick_once(self) -> list[object]:
        raise AssertionError("conversation import must not synchronously drain background work")


def _config(tmp_path: Path) -> AlphaConfig:
    return AlphaConfig(
        db_path=tmp_path / "alpha.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway-status.json",
        daemon_socket_path=tmp_path / "daemon.sock",
        daemon_status_path=tmp_path / "daemon-status.json",
    )


def _import_payload() -> dict[str, object]:
    return {
        "source_provider": "chatgpt",
        "timezone": "Asia/Shanghai",
        "conversations": [
            {
                "external_conversation_id": "conv_1",
                "title": "Imported design discussion",
                "messages": [
                    {
                        "external_message_id": "msg_1",
                        "role": "system",
                        "content": "External assistant policy.",
                        "created_at": "2026-01-01T10:00:00+08:00",
                    },
                    {
                        "external_message_id": "msg_2",
                        "role": "user",
                        "content": "I prefer direct feedback.",
                        "created_at": "2026-01-01T10:01:00+08:00",
                    },
                    {
                        "external_message_id": "msg_3",
                        "role": "assistant",
                        "content": "Understood.",
                        "created_at": "2026-01-01T10:02:00+08:00",
                    },
                ],
            }
        ],
    }


def test_daemon_handles_ask_with_session_guard_and_source_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    agent = _FakeAgent()
    daemon = AlphaDaemon(
        config,
        store=store,
        agent_manager=_FakeManager(agent),  # type: ignore[arg-type]
        runtime=DaemonRuntimeConfig(
            socket_path=config.daemon_socket_path,
            status_path=config.daemon_status_path,
            log_dir=config.log_dir,
        ),
    )

    response = daemon.handle_payload(
        {
            "type": "ask",
            "message": "hello",
            "session_id": "s1",
            "source_metadata": {"channel": "spoofed", "request_id": "req-1"},
        }
    )

    assert response == {
        "ok": True,
        "session_id": "s1",
        "response": "response to hello",
    }
    assert agent.calls == [
        (
            "hello",
            "s1",
            {
                "channel": "cli",
                "command": "ask",
                "client": {
                    "channel": "spoofed",
                    "request_id": "req-1",
                },
            },
        )
    ]


def test_daemon_shares_single_llm_trace_logger_across_runtime_services(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path), llm_debug_logging=True)
    store = StateStore(config.db_path)
    store.initialize()

    daemon = AlphaDaemon(config, store=store)
    factory_logger = daemon.agent_manager.factory.llm_trace_logger
    agent = daemon.agent_manager.factory.create()

    assert factory_logger is daemon.background_service.llm_trace_logger
    assert factory_logger is daemon.direct_compact_extraction.llm_trace_logger
    assert factory_logger is daemon.feedback_attribution.llm_trace_logger
    assert agent.llm_trace_logger is factory_logger
    submitter = agent.feedback_attribution_submitter
    assert isinstance(submitter, MethodType)
    assert submitter.__self__ is daemon.feedback_attribution
    assert factory_logger.enabled
    assert factory_logger.trace_log_path == config.log_dir / "llm.jsonl"


def test_daemon_returns_unknown_request_type_for_invalid_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "missing"})

    assert response["ok"] is False
    assert response["error"]["code"] == "UNKNOWN_REQUEST_TYPE"


def test_daemon_conversation_import_dry_run_returns_summary_without_writes(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)
    wake = _RecordingBackgroundWake()
    daemon.background_service = wake  # type: ignore[assignment]

    response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "input_name": "conversation_exports/export.json",
            "payload_json": json.dumps(_import_payload()),
            "dry_run": True,
        }
    )

    assert response["ok"] is True
    assert response["summary"] == {
        "batch_id": None,
        "source_provider": "chatgpt",
        "dry_run": True,
        "status": "completed",
        "input_name": "export.json",
        "conversations_seen": 1,
        "messages_seen": 3,
        "conversations_created": 1,
        "conversations_reused": 0,
        "messages_inserted": 3,
        "messages_deduped": 0,
    }
    assert wake.wake_calls == 0
    assert store.list_import_batches() == []
    assert store.get_imported_conversation("chatgpt", "conv_1") is None


def test_daemon_conversation_import_wakes_background_after_inserted_messages(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)
    wake = _RecordingBackgroundWake()
    daemon.background_service = wake  # type: ignore[assignment]

    response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "input_name": "export.json",
            "payload_json": json.dumps(_import_payload()),
        }
    )

    assert response["ok"] is True
    assert response["summary"]["messages_inserted"] == 3
    assert wake.wake_calls == 1


def test_daemon_conversation_import_real_run_and_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    import_response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "input_name": "export.json",
            "payload_json": json.dumps(_import_payload()),
        }
    )

    assert import_response["ok"] is True
    summary = import_response["summary"]
    assert summary["batch_id"]
    assert summary["messages_inserted"] == 3
    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    assert store.is_import_session(imported.session_id) is True

    status_response = daemon.handle_payload(
        {
            "type": "conversation_import_status",
            "batch_id": summary["batch_id"],
            "verbose": True,
        }
    )

    assert status_response["ok"] is True
    assert status_response["status"]["batch_id"] == summary["batch_id"]
    assert status_response["status"]["messages_inserted"] == 3
    assert status_response["status"]["extraction_pending"] == 3
    assert status_response["status"]["extraction_processed"] == 0
    assert status_response["conversations"] == [
        {
            "external_conversation_id": "conv_1",
            "title": "Imported design discussion",
            "session_id": imported.session_id,
            "messages_inserted": 3,
            "messages_deduped": 0,
            "session_reused": False,
            "extraction_pending": 3,
            "extraction_claimed": 0,
            "extraction_processed": 0,
            "extraction_failed": 0,
            "extraction_skipped": 0,
        }
    ]


def test_daemon_conversation_import_returns_validation_details(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)
    wake = _RecordingBackgroundWake()
    daemon.background_service = wake  # type: ignore[assignment]
    payload = _import_payload()
    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    first_conversation = conversations[0]
    assert isinstance(first_conversation, dict)
    messages = first_conversation["messages"]
    assert isinstance(messages, list)
    first_message = messages[0]
    assert isinstance(first_message, dict)
    first_message["created_at"] = "2026-01-01T10:00:00"

    response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "input_name": "bad.json",
            "payload_json": json.dumps(payload),
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "VALIDATION_ERROR"
    assert response["error"]["message"] == "Invalid conversation import payload."
    assert response["error"]["details"] == [
        {
            "path": "conversations[0].messages[0].created_at",
            "message": "timestamp must include an explicit timezone offset or Z",
            "code": "naive_timestamp",
        }
    ]
    assert wake.wake_calls == 0


def test_daemon_conversation_import_rejects_malformed_boundary_fields(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "input_name": "export.json",
            "payload_json": {"not": "a string"},
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_REQUEST"
    assert response["error"]["message"] == "payload_json must be a string."


def test_daemon_conversation_import_does_not_use_turn_guard(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    guard = ActiveTurnGuard()
    assert guard.begin("s1", "already running").accepted is True
    daemon = AlphaDaemon(config, store=store, turn_guard=guard)

    response = daemon.handle_payload(
        {
            "type": "conversation_import",
            "payload_json": json.dumps(_import_payload()),
            "dry_run": True,
        }
    )

    assert response["ok"] is True
    assert guard.is_active("s1") is True


def test_daemon_rejects_ordinary_turn_for_import_session(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    import_response = AlphaDaemon(config, store=store).handle_payload(
        {
            "type": "conversation_import",
            "payload_json": json.dumps(_import_payload()),
        }
    )
    assert import_response["ok"] is True
    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    agent = _FakeAgent()
    daemon = AlphaDaemon(
        config,
        store=store,
        agent_manager=_FakeManager(agent),  # type: ignore[arg-type]
    )

    response = daemon.handle_payload(
        {
            "type": "chat_turn",
            "message": "continue this",
            "session_id": imported.session_id,
        }
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "IMPORT_SESSION_NOT_CHAT"
    assert "cannot be continued" in response["error"]["message"]
    assert agent.calls == []


def test_daemon_status_response_includes_runtime_paths(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "status"})

    assert response["ok"] is True
    assert response["status"]["state"] == "running"
    assert response["status"]["socket_path"] == str(config.daemon_socket_path)
    assert response["status"]["status_path"] == str(config.daemon_status_path)
    assert response["status"]["background_enabled"] is True
    assert response["status"]["background_state"] == "stopped"
    assert response["status"]["background_last_tick"] is None
    assert response["status"]["background_last_success"] is None
    assert response["status"]["background_last_error"] is None
    assert response["status"]["background_next_tick"] is None


def test_daemon_disabled_background_status_has_no_ticks(tmp_path: Path) -> None:
    config = AlphaConfig(
        db_path=tmp_path / "alpha.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway-status.json",
        daemon_socket_path=tmp_path / "daemon.sock",
        daemon_status_path=tmp_path / "daemon-status.json",
        cognition_background=CognitionBackgroundConfig(enabled=False),
    )
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    daemon.background_service.start()
    response = daemon.handle_payload({"type": "status"})

    assert response["ok"] is True
    assert response["status"]["background_enabled"] is False
    assert response["status"]["background_state"] == "disabled"
    assert response["status"]["background_last_tick"] is None
    assert response["status"]["background_next_tick"] is None


def test_daemon_created_agents_and_background_share_loop_coordinator(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    first = daemon.agent_manager.get_or_create("s1")
    second = daemon.agent_manager.get_or_create("s2")

    assert first.coordinator is daemon.loop_coordinator
    assert second.coordinator is daemon.loop_coordinator
    assert daemon.background_service.coordinator is daemon.loop_coordinator


def test_background_holder_makes_daemon_foreground_turn_busy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)
    request = LoopAcquireRequest(
        loop_name="background:test",
        priority=LoopPriority.CONSOLIDATION,
        max_chunk_duration=timedelta(seconds=30),
    )

    with daemon.loop_coordinator.try_acquire(request):
        response = daemon.handle_payload(
            {"type": "ask", "message": "hello", "session_id": "s1"}
        )
        should_yield = daemon.loop_coordinator.yield_to_higher_priority()

    assert response["ok"] is True
    assert response["session_id"] == "s1"
    assert "Agent is currently background:test" in response["response"]
    assert should_yield is True
    assert store.list_session_messages("s1") == []


def test_daemon_stop_response_uses_current_graceful_stopping_status(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "stop"})

    assert response["ok"] is True
    assert response["status"]["state"] == "stopping"
    assert (
        response["status"]["message"] == "Daemon is draining the current request before stopping."
    )
    assert daemon.feedback_attribution._closed is True


def test_daemon_stop_response_accepts_immediate_policy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "stop", "policy": "immediate"})

    assert response["ok"] is True
    assert response["status"]["state"] == "stopping"
    assert response["status"]["message"] == "Daemon is stopping immediately."


def test_daemon_stop_rejects_unknown_policy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "stop", "policy": "eventually"})

    assert response["ok"] is False
    assert response["error"]["code"] == "INVALID_REQUEST"
    assert response["error"]["message"] == "Stop policy must be one of: graceful, immediate."


def test_daemon_refuses_to_start_when_status_pid_is_alive(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    runtime = DaemonRuntimeConfig(
        socket_path=config.daemon_socket_path,
        status_path=config.daemon_status_path,
        log_dir=config.log_dir,
    )
    write_daemon_status(
        runtime.status_path,
        running_status(config=config, runtime=runtime, message="already running"),
    )
    daemon = AlphaDaemon(config, store=store, runtime=runtime)

    with pytest.raises(DaemonAlreadyRunningError):
        daemon._assert_single_owner()


def test_daemon_disconnects_adapter_when_startup_connect_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    adapter = _FailingAdapter()
    monkeypatch.setattr("alpha_agent.daemon.runtime.configured_adapters", lambda: (adapter,))
    daemon = AlphaDaemon(config, store=store)

    with pytest.raises(RuntimeError, match="connect failed"):
        daemon.run()

    assert adapter.connected is True
    assert adapter.disconnected is True
    assert daemon.feedback_attribution._closed is True
