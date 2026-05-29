"""Shared cognition test helpers."""

from __future__ import annotations

from collections.abc import Callable

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    CounterpartId,
    CounterpartRef,
    CounterpartRole,
    Instant,
    StimulusKind,
    ThreadId,
    counterpart_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def id_factory(prefix: str = "evt") -> Callable[[], str]:
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}-{counter:04d}"

    return next_id


def clock_factory() -> Callable[[], str]:
    counter = 0

    def now() -> str:
        nonlocal counter
        counter += 1
        return f"2026-01-01T00:00:{counter:02d}+00:00"

    return now


def emit(
    log: EventLog,
    kind: CognitiveEventKind = CognitiveEventKind.PERCEIVED,
    *,
    payload: dict[str, object] | None = None,
    event_ids: Callable[[], str] | None = None,
    clock: Callable[[], str] | None = None,
) -> CognitiveEvent:
    emitter = EventEmitter(
        log,
        id_factory=event_ids or id_factory(),
        clock=clock or clock_factory(),
    )
    return emitter.emit(kind, payload=payload or default_payload(kind))


def default_payload(kind: CognitiveEventKind) -> dict[str, object]:
    if kind == CognitiveEventKind.PERCEIVED:
        return perceived_payload()
    if kind == CognitiveEventKind.JUDGED:
        return {"claim": "test claim"}
    return {}


def perceived_payload(
    *,
    index: object = 1,
    session_id: str = "s1",
    raw: object | None = None,
    tick_id: str | None = None,
    counterpart: CounterpartRef | None = None,
) -> dict[str, object]:
    counterpart_record = counterpart.to_record() if counterpart is not None else None
    return {
        "tick_id": tick_id or f"tick-{index}",
        "index": index,
        "stimulus_kind": StimulusKind.USER_MESSAGE.value,
        "payload_digest": f"digest-{index}",
        "thread_id": ThreadId.from_session(session_id).to_record(),
        "perception": {
            "id": f"perception:{index}",
            "source_kind": StimulusKind.USER_MESSAGE.value,
            "from_counterpart": counterpart_record,
            "raw": raw if raw is not None else f"message-{index}",
            "surface_intent": [],
            "raised_entities": [],
            "subject": {"kind": "subject", "id": str(SUBJECT_SELF)},
            "situation": {"kind": "situation", "id": f"situation:{index}"},
            "received_at": "2026-01-01T00:00:00+00:00",
        },
        "source_refs": [],
        "from_counterpart": counterpart_record,
        "present_counterparts": [counterpart_record] if counterpart_record else [],
    }


def counterpart_payload(
    counterpart_id: str = "counterpart:user-a",
    *,
    role: CounterpartRole = CounterpartRole.USER,
) -> dict[str, object]:
    return {
        "counterpart_id": counterpart_id,
        "role": role.value,
        "identity": {"display_name": "User A"},
        "metadata": {"source": "test"},
    }


def counterpart_output(counterpart_id: str = "counterpart:user-a"):
    return counterpart_ref(CounterpartId(counterpart_id))


def instant(value: str) -> Instant:
    return Instant(value)
