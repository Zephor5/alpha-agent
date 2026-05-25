"""In-memory event log implementation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    EventId,
    Instant,
    SubjectRef,
)


class InMemoryEventLog(EventLog):
    """Append-only event log for tests and lightweight local use."""

    def __init__(self) -> None:
        self._events: list[CognitiveEvent] = []
        self._by_id: dict[EventId, CognitiveEvent] = {}

    def append(self, event: CognitiveEvent) -> EventId:
        if event.id in self._by_id:
            raise ValueError(f"event already exists: {event.id}")
        self._events.append(event)
        self._by_id[event.id] = event
        return event.id

    def get(self, event_id: EventId) -> CognitiveEvent:
        try:
            return self._by_id[event_id]
        except KeyError as exc:
            raise KeyError(f"unknown cognitive event: {event_id}") from exc

    def iter(
        self,
        *,
        subject: SubjectRef | None = None,
        kinds: Iterable[CognitiveEventKind] | None = None,
        since: Instant | None = None,
        until: Instant | None = None,
    ) -> Iterator[CognitiveEvent]:
        kind_set = set(kinds) if kinds is not None else None
        for event in self._events:
            if subject is not None and event.subject != subject:
                continue
            if kind_set is not None and event.kind not in kind_set:
                continue
            if since is not None and event.timestamp < since:
                continue
            if until is not None and event.timestamp > until:
                continue
            yield event

    def length(self, *, subject: SubjectRef | None = None) -> int:
        if subject is None:
            return len(self._events)
        return sum(1 for event in self._events if event.subject == subject)
