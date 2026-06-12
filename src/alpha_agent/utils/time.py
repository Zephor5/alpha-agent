"""Time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def local_now() -> datetime:
    """Return the current timezone-aware local datetime."""

    return datetime.now().astimezone()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return utc_now().isoformat()
