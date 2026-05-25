"""Subject projection with persisted value-lens state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, Subject
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.cognition.value.lens import (
    default_value_lens,
    ensure_lens_schema,
    load_lens,
    upsert_lens_event,
)
from alpha_agent.state.store import StateStore


@dataclass(frozen=True)
class SubjectProjectionView:
    subject: Subject
    status: str = "materialized"


class SubjectProjection(Projection):
    """Rebuild the single subject and current value lens."""

    name = "subject"
    handles = frozenset(
        {
            CognitiveEventKind.SELF_MODEL_UPDATED,
            CognitiveEventKind.VALUE_LENS_SHIFTED,
        }
    )

    def __init__(self, event_log: EventLog, store: StateStore | None = None):
        self.event_log = event_log
        self.store = store or getattr(event_log, "store", None)
        if self.store is not None:
            ensure_lens_schema(self.store)

    def current(self) -> Subject:
        latest: Subject | None = None
        for event in self.event_log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]):
            latest = self._subject_from_event(event) or latest
        subject = latest or Subject()
        if self.store is not None:
            lens = load_lens(self.store, str(subject.id))
        else:
            lens = self._lens_from_events() or default_value_lens()
        return Subject.from_record({**subject.to_record(), "value_lens": lens.to_record()})

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.VALUE_LENS_SHIFTED and self.store is not None:
            upsert_lens_event(self.store, event)
        return None

    def reset(self) -> None:
        if self.store is not None:
            ensure_lens_schema(self.store)
            with self.store.transaction() as conn:
                conn.execute("DELETE FROM subject_value_lens")
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

    def _lens_from_events(self):
        latest = None
        for event in self.event_log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED]):
            if str(event.payload.get("subject_id") or SUBJECT_SELF) != str(SUBJECT_SELF):
                continue
            raw = event.payload.get("after")
            if isinstance(raw, dict):
                latest = raw
        if latest is None:
            return None
        from alpha_agent.cognition.models.value import ValueLens

        return ValueLens.from_record(latest)
