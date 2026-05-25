"""Shared cognition test helpers."""

from __future__ import annotations

from collections.abc import Callable

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    CounterpartId,
    CounterpartRole,
    Instant,
    counterpart_ref,
)


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
    return emitter.emit(kind, payload=payload or {})


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
