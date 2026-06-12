"""Time helpers."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_FIXED_OFFSET_RE = re.compile(r"^[+-](?:[01]\d|2[0-3]):[0-5]\d$")


def local_now() -> datetime:
    """Return the current timezone-aware local datetime."""

    return datetime.now().astimezone()


def utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""

    return datetime.now(UTC)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""

    return utc_now().isoformat()


def validate_timezone(value: str) -> str:
    """Return a normalized valid IANA name or fixed UTC offset."""

    timezone = value.strip() if value else ""
    if not timezone:
        raise ValueError("timezone must be a non-empty IANA name or fixed offset")
    if _FIXED_OFFSET_RE.fullmatch(timezone):
        return timezone
    try:
        ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"invalid timezone {value!r}") from exc
    return timezone


def local_timezone_identifier() -> str:
    """Return the best available local timezone identifier."""

    env_timezone = _timezone_from_env()
    if env_timezone is not None:
        return env_timezone
    localtime_timezone = _timezone_from_localtime()
    if localtime_timezone is not None:
        return localtime_timezone
    return _format_fixed_offset(local_now().utcoffset() or timedelta(0))


def _timezone_from_env() -> str | None:
    value = os.environ.get("TZ")
    if not value:
        return None
    candidate = value[1:] if value.startswith(":") else value
    try:
        return validate_timezone(candidate)
    except ValueError:
        return None


def _timezone_from_localtime() -> str | None:
    try:
        resolved = Path("/etc/localtime").resolve()
    except OSError:
        return None
    parts = resolved.parts
    for index, part in enumerate(parts):
        if part != "zoneinfo":
            continue
        candidate = "/".join(parts[index + 1 :])
        if not candidate:
            continue
        try:
            return validate_timezone(candidate)
        except ValueError:
            return None
    return None


def _format_fixed_offset(offset: timedelta) -> str:
    total_minutes = round(offset.total_seconds() / 60)
    sign = "+" if total_minutes >= 0 else "-"
    absolute_minutes = abs(total_minutes)
    hours, minutes = divmod(absolute_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"
