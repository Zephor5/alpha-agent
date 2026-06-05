from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.models import LoopPriority
from alpha_agent.config import AlphaConfig, CognitionBackgroundConfig
from alpha_agent.daemon.runtime import AlphaDaemon, DaemonAlreadyRunningError
from alpha_agent.daemon.status import DaemonRuntimeConfig, running_status, write_daemon_status
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


def _config(tmp_path: Path) -> AlphaConfig:
    return AlphaConfig(
        db_path=tmp_path / "alpha.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway-status.json",
        daemon_socket_path=tmp_path / "daemon.sock",
        daemon_status_path=tmp_path / "daemon-status.json",
    )


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


def test_daemon_returns_unknown_request_type_for_invalid_payload(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = StateStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store)

    response = daemon.handle_payload({"type": "missing"})

    assert response["ok"] is False
    assert response["error"]["code"] == "UNKNOWN_REQUEST_TYPE"


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
