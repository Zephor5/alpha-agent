"""CLI IPC client for the Alpha daemon."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alpha_agent.daemon.ipc import request_daemon


class DaemonClient:
    """Thin JSON-lines client for one daemon request at a time."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request to the daemon."""

        return request_daemon(self.socket_path, payload)

    def status(self) -> dict[str, Any]:
        """Read daemon status through IPC."""

        return self.request({"type": "status"})

    def stop(self, *, policy: str = "graceful") -> dict[str, Any]:
        """Request daemon shutdown."""

        return self.request({"type": "stop", "policy": policy})
