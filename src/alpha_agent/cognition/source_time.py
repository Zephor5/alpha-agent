"""Resolve source-message evidence time for cognition prompts and metadata."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Literal
from zoneinfo import ZoneInfo

from alpha_agent.cognition.models import BeliefRecord, Reference
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore
from alpha_agent.utils.time import validate_timezone

SOURCE_TIME_BASIS_SESSION_MESSAGE: Literal["session_message"] = "session_message"

_NON_EVIDENCE_SESSION_MESSAGE_KINDS = frozenset({"system_reminder", "compressed_message"})


@dataclass(frozen=True)
class SourceTimeRange:
    """UTC source-time range resolved from persisted session-message evidence."""

    start: datetime
    end: datetime
    session_id: str
    basis: Literal["session_message"] = SOURCE_TIME_BASIS_SESSION_MESSAGE

    def __post_init__(self) -> None:
        start = _normalize_utc_datetime(self.start, "start")
        end = _normalize_utc_datetime(self.end, "end")
        if end < start:
            raise ValueError("source time end must not be before start")
        if not self.session_id.strip():
            raise ValueError("source time session_id must be non-empty")
        if self.basis != SOURCE_TIME_BASIS_SESSION_MESSAGE:
            raise ValueError("source time basis must be session_message")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def source_time_start(self) -> str:
        return self.start.isoformat()

    @property
    def source_time_end(self) -> str:
        return self.end.isoformat()

    @property
    def source_time_basis(self) -> Literal["session_message"]:
        return self.basis

    def to_metadata(self) -> dict[str, str]:
        return {
            "source_time_start": self.source_time_start,
            "source_time_end": self.source_time_end,
            "source_time_basis": self.source_time_basis,
        }


def resolve_belief_source_time_range(
    store: StateStore,
    belief: BeliefRecord,
) -> SourceTimeRange | None:
    """Resolve source time from a belief's session-message sources only."""

    return resolve_source_time_range(store, belief.sources)


def resolve_source_time_range(
    store: StateStore,
    refs: Iterable[Reference],
) -> SourceTimeRange | None:
    """Resolve a UTC source-time range from session-message references.

    Non-session references are ignored. Referenced reminder and compressed
    messages are fetched so missing refs fail fast, but they are not evidence
    and therefore do not contribute to the resolved time range.
    """

    message_ids = tuple(ref.id for ref in refs if ref.kind == SOURCE_TIME_BASIS_SESSION_MESSAGE)
    if not message_ids:
        return None

    messages = _fetch_session_messages(store, message_ids)
    evidence_messages = tuple(
        message for message in messages if message.kind not in _NON_EVIDENCE_SESSION_MESSAGE_KINDS
    )
    if not evidence_messages:
        return None

    session_ids = sorted({message.session_id for message in evidence_messages})
    if len(session_ids) != 1:
        raise ValueError(
            "source message time range spans multiple sessions: " + ", ".join(session_ids)
        )

    timestamps = tuple(_parse_session_message_timestamp(message) for message in evidence_messages)
    return SourceTimeRange(
        start=min(timestamps),
        end=max(timestamps),
        session_id=session_ids[0],
    )


def render_source_time_line(store: StateStore, source_time: SourceTimeRange) -> str:
    """Render a prompt-facing source-time line in the stored session timezone."""

    session_record = store.get_session_record(source_time.session_id)
    if session_record is None:
        raise KeyError(f"missing session record for source-time session: {source_time.session_id}")
    local_timezone = _timezone_info(session_record.timezone)
    start = source_time.start.astimezone(local_timezone)
    end = source_time.end.astimezone(local_timezone)
    if source_time.start == source_time.end:
        return f"Source message time: {_format_prompt_time(start)} ({session_record.timezone})."
    return (
        "Source message time range: "
        f"{_format_prompt_time(start)} to {_format_prompt_time(end)} "
        f"({session_record.timezone})."
    )


def _fetch_session_messages(
    store: StateStore,
    message_ids: tuple[str, ...],
) -> tuple[SessionMessage, ...]:
    messages = tuple(store.list_session_messages_by_ids(list(message_ids)))
    found_ids = {message.id for message in messages}
    missing_ids = tuple(
        dict.fromkeys(message_id for message_id in message_ids if message_id not in found_ids)
    )
    if missing_ids:
        raise KeyError("missing session_message source refs: " + ", ".join(missing_ids))
    return messages


def _parse_session_message_timestamp(message: SessionMessage) -> datetime:
    raw_value = message.created_at.strip()
    if not raw_value:
        raise ValueError(f"invalid created_at for session_message {message.id}: empty timestamp")
    if raw_value.endswith("Z"):
        raw_value = f"{raw_value[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"invalid created_at for session_message {message.id}: must be ISO datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(
            f"invalid created_at for session_message {message.id}: timezone is required"
        )
    return parsed.astimezone(UTC)


def _normalize_utc_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"source time {field_name} must include timezone")
    return value.astimezone(UTC)


def _timezone_info(identifier: str) -> tzinfo:
    timezone_name = validate_timezone(identifier)
    if timezone_name[0] in "+-":
        hours = int(timezone_name[1:3])
        minutes = int(timezone_name[4:6])
        offset = timedelta(hours=hours, minutes=minutes)
        if timezone_name[0] == "-":
            offset = -offset
        return timezone(offset)
    return ZoneInfo(timezone_name)


def _format_prompt_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M")


__all__ = [
    "SOURCE_TIME_BASIS_SESSION_MESSAGE",
    "SourceTimeRange",
    "render_source_time_line",
    "resolve_belief_source_time_range",
    "resolve_source_time_range",
]
