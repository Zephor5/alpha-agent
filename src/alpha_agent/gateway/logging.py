"""Small JSONL logging helpers for gateway runtime events."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from alpha_agent.utils.time import utc_now_iso

LogLevel = Literal["debug", "info", "warning", "error"]


@dataclass(frozen=True, slots=True)
class GatewayLogContext:
    """Gateway log context with raw external identifiers redacted by default."""

    session_id: str | None = None
    platform: str | None = None
    chat_id: str | None = None
    user_id: str | None = None

    def to_redacted_dict(self) -> dict[str, str]:
        """Return log-safe context fields with external IDs hashed."""

        data: dict[str, str] = {}
        if self.session_id:
            data["session_id"] = self.session_id
        if self.platform:
            data["platform"] = self.platform
        if self.chat_id:
            data["chat_id_hash"] = hash_identifier(self.chat_id)
        if self.user_id:
            data["user_id_hash"] = hash_identifier(self.user_id)
        return data


def hash_identifier(value: str) -> str:
    """Return a short stable hash for an external platform identifier."""

    normalized = value.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:16]


def append_gateway_log(
    path: Path,
    *,
    event: str,
    message: str,
    level: LogLevel = "info",
    context: GatewayLogContext | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Append one inspectable JSONL gateway log entry."""

    path.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "timestamp": utc_now_iso(),
        "level": level,
        "event": event,
        "message": message,
        "context": context.to_redacted_dict() if context else {},
        "metadata": metadata or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
