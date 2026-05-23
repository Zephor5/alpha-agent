"""Unix socket JSON-lines server for the daemon."""

from __future__ import annotations

import json
import socketserver
from collections.abc import Callable
from pathlib import Path
from threading import Event
from typing import Any

JsonObject = dict[str, Any]
DaemonRequestHandler = Callable[[JsonObject], JsonObject]


class _ThreadedUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class JsonLineDaemonServer:
    """Small Unix socket server that handles one JSON object per line."""

    def __init__(self, socket_path: Path, handler: DaemonRequestHandler):
        self.socket_path = socket_path
        self.handler = handler
        self._stop_event = Event()
        self._server: _ThreadedUnixServer | None = None

    def serve_forever(self) -> None:
        """Serve daemon IPC until shutdown is requested."""

        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()

        outer = self

        class RequestHandler(socketserver.StreamRequestHandler):
            def handle(self) -> None:
                line = self.rfile.readline()
                try:
                    request = json.loads(line.decode("utf-8"))
                    if not isinstance(request, dict):
                        request = {"type": None}
                except (UnicodeDecodeError, json.JSONDecodeError):
                    response = {
                        "ok": False,
                        "error": {
                            "code": "INVALID_JSON",
                            "message": "Request must be a valid JSON line.",
                        },
                    }
                else:
                    try:
                        response = outer.handler(request)
                    except Exception as exc:
                        response = {
                            "ok": False,
                            "error": {
                                "code": "INTERNAL_ERROR",
                                "message": str(exc),
                            },
                        }
                self.wfile.write(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")

        try:
            with _ThreadedUnixServer(str(self.socket_path), RequestHandler) as server:
                server.timeout = 0.5
                self._server = server
                while not self._stop_event.is_set():
                    server.handle_request()
        finally:
            self._server = None
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def stop(self) -> None:
        """Request shutdown after the current request."""

        self._stop_event.set()

    def stop_immediately(self) -> None:
        """Stop accepting new requests and close the listening socket now."""

        self._stop_event.set()
        server = self._server
        if server is not None:
            server.server_close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
