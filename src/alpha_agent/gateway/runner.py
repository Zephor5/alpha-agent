"""Small gateway coordination helpers."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True, slots=True)
class TurnStartResult:
    """Decision returned when a platform turn asks to enter a session."""

    accepted: bool
    bypassed: bool = False
    queued: bool = False
    reason: str | None = None


class ActiveTurnGuard:
    """In-memory guard that allows at most one active non-control turn per session."""

    def __init__(self, bypass_commands: set[str] | None = None):
        self._active_session_ids: set[str] = set()
        self._bypass_commands = bypass_commands or {"/stop", "/reset", "/status"}
        self._lock = Lock()

    def begin(self, session_id: str, text: str, *, allow_queue: bool = False) -> TurnStartResult:
        """Try to mark a session active for processing."""

        if self._is_bypass_command(text):
            return TurnStartResult(accepted=True, bypassed=True)
        with self._lock:
            if session_id in self._active_session_ids:
                return TurnStartResult(
                    accepted=False,
                    queued=allow_queue,
                    reason="active_turn",
                )
            self._active_session_ids.add(session_id)
        return TurnStartResult(accepted=True)

    def complete(self, session_id: str) -> None:
        """Release a previously active session."""

        with self._lock:
            self._active_session_ids.discard(session_id)

    def is_active(self, session_id: str) -> bool:
        """Return whether a session currently has an active non-control turn."""

        with self._lock:
            return session_id in self._active_session_ids

    def _is_bypass_command(self, text: str) -> bool:
        first_token = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
        return first_token in self._bypass_commands
