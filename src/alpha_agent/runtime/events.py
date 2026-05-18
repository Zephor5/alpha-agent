"""Event creation helpers."""

from __future__ import annotations

from typing import Literal

from alpha_agent.memory.models import Event
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


def create_event(
    session_id: str,
    role: Literal["user", "assistant", "system", "tool"],
    content: str,
) -> Event:
    """Create a raw event with generated id and timestamp."""

    return Event(
        id=new_id("evt"),
        session_id=session_id,
        role=role,
        content=content,
        created_at=utc_now_iso(),
        metadata={},
    )
