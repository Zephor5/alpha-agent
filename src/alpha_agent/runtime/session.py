"""Session helpers."""

from __future__ import annotations

from alpha_agent.utils.ids import new_id


def new_session_id() -> str:
    """Generate a new chat session id."""

    return new_id("session")
