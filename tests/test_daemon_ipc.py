from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from alpha_agent.daemon.ipc import DaemonIpcServer, request_daemon
from alpha_agent.daemon.server import JsonLineDaemonServer
from alpha_agent.daemon.status import DaemonStatus, read_daemon_status, write_daemon_status


def test_daemon_status_round_trips_json(tmp_path: Path) -> None:
    status_path = tmp_path / "daemon-status.json"
    socket_path = tmp_path / "daemon.sock"
    status = DaemonStatus(
        state="running",
        running=True,
        pid=12345,
        socket_path=str(socket_path),
        status_path=str(status_path),
        updated_at="2026-01-01T00:00:00+00:00",
        adapters=["telegram", "feishu"],
    )

    write_daemon_status(status_path, status)

    assert read_daemon_status(status_path) == status


def test_daemon_status_missing_or_invalid_file_returns_none(tmp_path: Path) -> None:
    assert read_daemon_status(tmp_path / "missing.json") is None

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("not-json", encoding="utf-8")

    assert read_daemon_status(invalid_path) is None


def test_daemon_ipc_returns_single_json_line_response(short_socket_path: Path) -> None:
    socket_path = short_socket_path
    server = DaemonIpcServer(
        socket_path,
        handlers={"ping": lambda request: {"echo": request["payload"]}},
    )
    thread = threading.Thread(target=server.serve_once, daemon=True)
    thread.start()
    _wait_for_socket(socket_path)

    response = request_daemon(socket_path, {"type": "ping", "payload": "hello"})
    thread.join(timeout=2)

    assert response == {"ok": True, "echo": "hello"}
    assert not thread.is_alive()


def test_daemon_ipc_unknown_request_type_is_validated_at_boundary(
    short_socket_path: Path,
) -> None:
    socket_path = short_socket_path
    server = DaemonIpcServer(socket_path, handlers={"ping": lambda _request: {"pong": True}})
    thread = threading.Thread(target=server.serve_once, daemon=True)
    thread.start()
    _wait_for_socket(socket_path)

    response = request_daemon(socket_path, {"type": "missing"})
    thread.join(timeout=2)

    assert response["ok"] is False
    assert response["error"]["code"] == "UNKNOWN_REQUEST_TYPE"


def test_daemon_client_reports_not_running_for_missing_socket(tmp_path: Path) -> None:
    response = request_daemon(tmp_path / "missing.sock", {"type": "status"})

    assert response["ok"] is False
    assert response["error"]["code"] == "DAEMON_NOT_RUNNING"


def test_daemon_client_reports_not_running_when_connection_fails(tmp_path: Path) -> None:
    stale_socket_path = tmp_path / "stale.sock"
    stale_socket_path.write_text("not a socket", encoding="utf-8")

    response = request_daemon(stale_socket_path, {"type": "status"})

    assert response["ok"] is False
    assert response["error"]["code"] == "DAEMON_NOT_RUNNING"


def test_json_line_server_wraps_handler_exception(short_socket_path: Path) -> None:
    def fail(_request):
        raise RuntimeError("boom")

    server = JsonLineDaemonServer(short_socket_path, fail)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _wait_for_socket(short_socket_path)

    response = request_daemon(short_socket_path, {"type": "ask"})
    server.stop()
    thread.join(timeout=2)

    assert response["ok"] is False
    assert response["error"]["code"] == "INTERNAL_ERROR"


def test_json_line_server_immediate_stop_removes_socket(short_socket_path: Path) -> None:
    server = JsonLineDaemonServer(short_socket_path, lambda _request: {"ok": True})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _wait_for_socket(short_socket_path)

    server.stop_immediately()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert not short_socket_path.exists()


def _wait_for_socket(path: Path) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"socket was not created: {path}")


@pytest.fixture
def short_socket_path() -> Iterator[Path]:
    with TemporaryDirectory(prefix="alpha-daemon-", dir="/tmp") as directory:
        yield Path(directory) / "daemon.sock"
