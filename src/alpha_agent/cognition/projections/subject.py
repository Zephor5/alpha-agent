"""Stub subject projection for Phase 02 reactive ticks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, Subject
from alpha_agent.cognition.projections.base import Projection


@dataclass(frozen=True)
class SubjectProjectionView:
    subject: Subject
    status: str = "stub"


class SubjectProjection(Projection):
    """Rebuild the single subject from self-model events, or return default."""

    name = "subject"
    handles = frozenset({CognitiveEventKind.SELF_MODEL_UPDATED})

    def __init__(self, event_log: EventLog):
        self.event_log = event_log

    def current(self) -> Subject:
        latest: Subject | None = None
        for event in self.event_log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]):
            latest = self._subject_from_event(event) or latest
        return latest or Subject()

    def apply(self, event: CognitiveEvent) -> None:
        return None

    def reset(self) -> None:
        return None

    def view(self) -> SubjectProjectionView:
        return SubjectProjectionView(subject=self.current())

    def _subject_from_event(self, event: CognitiveEvent) -> Subject | None:
        raw = event.payload.get("subject")
        if isinstance(raw, dict):
            return Subject.from_record(raw)
        fields: dict[str, Any] = {}
        for key in ("role", "capabilities", "declared_needs", "held_at"):
            if key in event.payload:
                fields[key] = event.payload[key]
        return Subject.from_record(fields) if fields else None
