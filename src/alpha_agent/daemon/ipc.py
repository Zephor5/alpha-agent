"""Unix socket JSON-lines IPC for the daemon."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]
DaemonHandler = Callable[[JsonObject], JsonObject]

DAEMON_NOT_RUNNING = "DAEMON_NOT_RUNNING"
INVALID_JSON = "INVALID_JSON"
INVALID_REQUEST = "INVALID_REQUEST"
UNKNOWN_REQUEST_TYPE = "UNKNOWN_REQUEST_TYPE"
INTERNAL_ERROR = "INTERNAL_ERROR"


class DaemonIpcServer:
    """Single-request JSON-lines daemon IPC server."""

    def __init__(self, socket_path: Path, handlers: Mapping[str, DaemonHandler]):
        self.socket_path = socket_path
        self._handlers = dict(handlers)

    def serve_once(self) -> None:
        """Serve one request on a Unix socket, then close and remove the socket."""

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(str(self.socket_path))
                server.listen(1)
                connection, _address = server.accept()
                with connection:
                    request_line = _read_line(connection)
                    response = self._handle_request_line(request_line)
                    connection.sendall(_encode_line(response))
        finally:
            if self.socket_path.exists():
                self.socket_path.unlink()

    def _handle_request_line(self, request_line: bytes) -> JsonObject:
        try:
            raw = json.loads(request_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return error_response(INVALID_JSON, "Request must be a valid JSON line.")

        if not isinstance(raw, dict):
            return error_response(INVALID_REQUEST, "Request must be a JSON object.")

        request_type = raw.get("type")
        if not isinstance(request_type, str) or not request_type:
            return error_response(UNKNOWN_REQUEST_TYPE, "Request type is required.")

        handler = self._handlers.get(request_type)
        if handler is None:
            return error_response(UNKNOWN_REQUEST_TYPE, f"Unknown request type: {request_type}")

        try:
            handler_response = handler(raw)
        except Exception as exc:
            return error_response(INTERNAL_ERROR, str(exc))

        response = dict(handler_response)
        response["ok"] = bool(response.get("ok", True))
        return response


def request_daemon(
    socket_path: Path,
    request: JsonObject,
    *,
    timeout: float | None = None,
) -> JsonObject:
    """Send one JSON request to the daemon and return one JSON response."""

    if not socket_path.exists():
        return error_response(DAEMON_NOT_RUNNING, "Daemon socket does not exist.")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            if timeout is not None:
                client.settimeout(timeout)
            client.connect(str(socket_path))
            client.sendall(_encode_line(request))
            response_line = _read_line(client)
    except TimeoutError:
        return error_response("DAEMON_REQUEST_TIMEOUT", "Daemon request timed out.")
    except OSError:
        return error_response(DAEMON_NOT_RUNNING, "Daemon socket is not accepting connections.")

    try:
        response = json.loads(response_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_response(INVALID_JSON, "Daemon response must be a valid JSON line.")

    if not isinstance(response, dict):
        return error_response(INVALID_REQUEST, "Daemon response must be a JSON object.")

    response.setdefault("ok", False)
    return response


def error_response(code: str, message: str) -> JsonObject:
    """Build a stable daemon IPC error response."""

    return {"ok": False, "error": {"code": code, "message": message}}


def _encode_line(payload: JsonObject) -> bytes:
    return json.dumps(payload, sort_keys=True).encode("utf-8") + b"\n"


def _read_line(connection: socket.socket) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(4096)
        if not chunk:
            break
        if b"\n" in chunk:
            before_newline, _newline, _after_newline = chunk.partition(b"\n")
            chunks.append(before_newline)
            break
        chunks.append(chunk)
    return b"".join(chunks)
