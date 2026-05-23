from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from alpha_agent.config import AlphaConfig
from alpha_agent.daemon.runtime import AlphaDaemon, DaemonAlreadyRunningError, StopPolicy
from alpha_agent.daemon.status import (
    DaemonRuntimeConfig,
    daemon_lock_path,
    read_daemon_status,
)
from alpha_agent.memory.store import MemoryStore


class _FakeManager:
    def __init__(self):
        self.evicted = False

    def evict_all(self) -> None:
        self.evicted = True


class _FailingServer:
    instances: list[_FailingServer] = []

    def __init__(self, socket_path: Path, handler: Any):
        self.socket_path = socket_path
        self.handler = handler
        self.stopped = False
        self.stopped_immediately = False
        _FailingServer.instances.append(self)

    def serve_forever(self) -> None:
        raise RuntimeError("server failed")

    def stop(self) -> None:
        self.stopped = True

    def stop_immediately(self) -> None:
        self.stopped_immediately = True


class _StoppingServer:
    def __init__(self, socket_path: Path, handler: Any):
        self.socket_path = socket_path
        self.handler = handler
        self.stopped = False

    def serve_forever(self) -> None:
        self.socket_path.write_text("stale socket", encoding="utf-8")

    def stop(self) -> None:
        self.stopped = True


class _FakeSignalModule:
    SIGTERM = 15
    SIGINT = 2

    def __init__(self):
        self.handlers: dict[int, Any] = {
            self.SIGTERM: "old-term",
            self.SIGINT: "old-int",
        }

    def getsignal(self, signum: int) -> Any:
        return self.handlers[signum]

    def signal(self, signum: int, handler: Any) -> Any:
        previous = self.handlers[signum]
        self.handlers[signum] = handler
        return previous


def _config(tmp_path: Path) -> AlphaConfig:
    return AlphaConfig(
        db_path=tmp_path / "alpha.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway-status.json",
        daemon_socket_path=tmp_path / "daemon.sock",
        daemon_status_path=tmp_path / "daemon-status.json",
    )


def _runtime(config: AlphaConfig) -> DaemonRuntimeConfig:
    return DaemonRuntimeConfig(
        socket_path=config.daemon_socket_path,
        status_path=config.daemon_status_path,
        log_dir=config.log_dir,
    )


def test_daemon_runtime_lock_prevents_double_start_without_status_file(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    runtime = _runtime(config)
    owner = AlphaDaemon(config, store=store, runtime=runtime)
    contender = AlphaDaemon(config, store=store, runtime=runtime)

    lock = owner._acquire_runtime_lock()
    try:
        with pytest.raises(DaemonAlreadyRunningError):
            contender._acquire_runtime_lock()
    finally:
        lock.release()


def test_daemon_runtime_lock_replaces_stale_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    runtime = _runtime(config)
    lock_path = daemon_lock_path(runtime)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("999999\n", encoding="utf-8")
    monkeypatch.setattr("alpha_agent.daemon.status.is_pid_running", lambda _pid: False)
    daemon = AlphaDaemon(config, store=store, runtime=runtime)

    lock = daemon._acquire_runtime_lock()
    try:
        assert lock.path == lock_path
        assert lock_path.read_text(encoding="utf-8").strip() == str(lock.pid)
    finally:
        lock.release()


@pytest.mark.parametrize("contents", ["", "not-a-pid\n"])
def test_daemon_runtime_lock_replaces_stale_invalid_owner(
    tmp_path: Path,
    contents: str,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    runtime = _runtime(config)
    lock_path = daemon_lock_path(runtime)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(contents, encoding="utf-8")
    old_timestamp = time.time() - 60
    os.utime(lock_path, (old_timestamp, old_timestamp))
    daemon = AlphaDaemon(config, store=store, runtime=runtime)

    lock = daemon._acquire_runtime_lock()
    try:
        assert lock_path.read_text(encoding="utf-8").strip() == str(lock.pid)
    finally:
        lock.release()


@pytest.mark.parametrize("contents", ["", "not-a-pid\n"])
def test_daemon_runtime_lock_preserves_fresh_invalid_owner(
    tmp_path: Path,
    contents: str,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    runtime = _runtime(config)
    lock_path = daemon_lock_path(runtime)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(contents, encoding="utf-8")
    daemon = AlphaDaemon(config, store=store, runtime=runtime)

    with pytest.raises(DaemonAlreadyRunningError):
        daemon._acquire_runtime_lock()

    assert lock_path.read_text(encoding="utf-8") == contents


def test_daemon_runtime_lock_preserves_valid_live_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    runtime = _runtime(config)
    lock_path = daemon_lock_path(runtime)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("12345\n", encoding="utf-8")
    monkeypatch.setattr("alpha_agent.daemon.status.is_pid_running", lambda pid: pid == 12345)
    daemon = AlphaDaemon(config, store=store, runtime=runtime)

    with pytest.raises(DaemonAlreadyRunningError):
        daemon._acquire_runtime_lock()

    assert lock_path.read_text(encoding="utf-8") == "12345\n"


def test_daemon_runtime_lock_is_released_after_server_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    manager = _FakeManager()
    monkeypatch.setattr("alpha_agent.daemon.runtime.configured_adapters", lambda: ())
    monkeypatch.setattr("alpha_agent.daemon.runtime.JsonLineDaemonServer", _FailingServer)
    daemon = AlphaDaemon(
        config,
        store=store,
        agent_manager=manager,  # type: ignore[arg-type]
        runtime=_runtime(config),
    )

    with pytest.raises(RuntimeError, match="server failed"):
        daemon.run()

    status = read_daemon_status(config.daemon_status_path)
    assert status is not None
    assert status.state == "error"
    assert status.running is False
    assert "server failed" in status.message
    assert manager.evicted is True
    assert not daemon_lock_path(_runtime(config)).exists()


def test_daemon_stop_policy_is_explicit_and_graceful_by_default(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store, runtime=_runtime(config))
    server = _FailingServer(config.daemon_socket_path, daemon.handle_payload)
    daemon._server = server  # type: ignore[assignment]

    daemon.stop()

    status = read_daemon_status(config.daemon_status_path)
    assert status is not None
    assert status.state == "stopping"
    assert status.message == "Daemon is draining the current request before stopping."
    assert daemon.stop_policy is StopPolicy.GRACEFUL
    assert server.stopped is True
    assert server.stopped_immediately is False


def test_daemon_stop_policy_can_be_immediate(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store, runtime=_runtime(config))
    server = _FailingServer(config.daemon_socket_path, daemon.handle_payload)
    daemon._server = server  # type: ignore[assignment]

    daemon.stop(StopPolicy.IMMEDIATE)

    status = read_daemon_status(config.daemon_status_path)
    assert status is not None
    assert status.state == "stopping"
    assert status.message == "Daemon is stopping immediately."
    assert daemon.stop_policy is StopPolicy.IMMEDIATE
    assert server.stopped is False
    assert server.stopped_immediately is True


def test_daemon_signal_handlers_request_stop_and_restore_previous_handlers(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    daemon = AlphaDaemon(config, store=store, runtime=_runtime(config))
    fake_signals = _FakeSignalModule()

    restore = daemon._install_signal_handlers(fake_signals)  # type: ignore[arg-type]
    fake_signals.handlers[fake_signals.SIGTERM](fake_signals.SIGTERM, None)

    assert daemon.stop_policy is StopPolicy.GRACEFUL

    restore()

    assert fake_signals.handlers == {
        fake_signals.SIGTERM: "old-term",
        fake_signals.SIGINT: "old-int",
    }


def test_daemon_cleanup_runs_before_signal_restore_error_is_raised(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.db_path)
    store.initialize()
    manager = _FakeManager()
    monkeypatch.setattr("alpha_agent.daemon.runtime.configured_adapters", lambda: ())
    monkeypatch.setattr("alpha_agent.daemon.runtime.JsonLineDaemonServer", _StoppingServer)
    daemon = AlphaDaemon(
        config,
        store=store,
        agent_manager=manager,  # type: ignore[arg-type]
        runtime=_runtime(config),
    )

    def restore() -> None:
        raise RuntimeError("restore failed")

    monkeypatch.setattr(daemon, "_install_signal_handlers", lambda: restore)

    with pytest.raises(RuntimeError, match="restore failed"):
        daemon.run()

    status = read_daemon_status(config.daemon_status_path)
    assert status is not None
    assert status.state == "idle"
    assert manager.evicted is True
    assert not config.daemon_socket_path.exists()
    assert not daemon_lock_path(_runtime(config)).exists()
