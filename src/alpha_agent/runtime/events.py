"""Event creation helpers."""

from __future__ import annotations

import json
from typing import Any, Literal

from alpha_agent.memory.models import Event
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

EventRole = Literal["user", "assistant", "system", "tool"]


def create_event(
    session_id: str,
    role: EventRole,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a raw event with generated id and timestamp."""

    return Event(
        id=new_id("evt"),
        session_id=session_id,
        role=role,
        content=content,
        created_at=utc_now_iso(),
        metadata=metadata or {},
    )


def create_runtime_event(
    session_id: str,
    event_type: str,
    content: str,
    *,
    role: EventRole = "system",
    metadata: dict[str, Any] | None = None,
) -> Event:
    """Create a structured runtime event stored in the normal event log."""

    event_metadata = {"event_type": event_type}
    event_metadata.update(metadata or {})
    return create_event(
        session_id=session_id,
        role=role,
        content=content,
        metadata=event_metadata,
    )


def deterministic_json(value: Any) -> str:
    """Serialize event payloads deterministically for replayable logs."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
