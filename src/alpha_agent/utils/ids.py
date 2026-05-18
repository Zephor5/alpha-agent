"""Identifier helpers."""

from __future__ import annotations

from uuid import uuid4


def new_id(prefix: str) -> str:
    """Generate a readable unique identifier with a domain prefix."""

    return f"{prefix}_{uuid4().hex}"
