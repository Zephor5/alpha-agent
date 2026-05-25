"""Event log protocol."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    EventId,
    Instant,
    SubjectRef,
)


class EventLog(Protocol):
    """Append-only cognitive event log."""

    def append(self, event: CognitiveEvent) -> EventId:
        """Append one event and return its id."""

    def get(self, event_id: EventId) -> CognitiveEvent:
        """Return one event by id."""

    def iter(
        self,
        *,
        subject: SubjectRef | None = None,
        kinds: Iterable[CognitiveEventKind] | None = None,
        since: Instant | None = None,
        until: Instant | None = None,
    ) -> Iterator[CognitiveEvent]:
        """Iterate events in stable replay order."""

    def length(self, *, subject: SubjectRef | None = None) -> int:
        """Return event count."""
