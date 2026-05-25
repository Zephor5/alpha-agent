"""Subject projection with persisted value-lens state."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, SelfModel, Subject
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.cognition.value.lens import (
    default_value_lens,
    ensure_lens_schema,
    load_lens,
    upsert_lens_event,
)
from alpha_agent.state.store import StateStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_view (
    id TEXT PRIMARY KEY,
    role TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    declared_needs TEXT NOT NULL DEFAULT '[]',
    value_lens_id TEXT,
    self_model TEXT NOT NULL DEFAULT '{}',
    served_counterparts TEXT NOT NULL DEFAULT '[]',
    known_biases TEXT NOT NULL DEFAULT '[]',
    held_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
)
"""


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
            self._ensure_schema()
            self._rebuild_if_empty()

    def current(self) -> Subject:
        subject = self._stored_subject() if self.store is not None else self._scan_subject()
        subject = subject or Subject()
        if self.store is not None:
            lens = load_lens(self.store, str(subject.id))
        else:
            lens = self._lens_from_events() or default_value_lens()
        return Subject.from_record({**subject.to_record(), "value_lens": lens.to_record()})

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.VALUE_LENS_SHIFTED and self.store is not None:
            upsert_lens_event(self.store, event)
        elif event.kind == CognitiveEventKind.SELF_MODEL_UPDATED and self.store is not None:
            self._upsert_self_model_event(event)
        return None

    def reset(self) -> None:
        if self.store is not None:
            ensure_lens_schema(self.store)
            self._ensure_schema()
            with self.store.transaction() as conn:
                conn.execute("DELETE FROM subject_value_lens")
                conn.execute("DELETE FROM subject_view")
        return None

    def view(self) -> SubjectProjectionView:
        return SubjectProjectionView(subject=self.current())

    @staticmethod
    def subject_with_self_model(subject: Subject, self_model: SelfModel) -> Subject:
        return Subject.from_record({**subject.to_record(), "self_model": self_model.to_record()})

    def _ensure_schema(self) -> None:
        if self.store is None:
            return
        with self.store.transaction() as conn:
            conn.execute(_SCHEMA)

    def _rebuild_if_empty(self) -> None:
        if self.store is None:
            return
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM subject_view LIMIT 1").fetchone()
        if row is not None:
            return
        for event in self.event_log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]):
            self._upsert_self_model_event(event)

    def _stored_subject(self) -> Subject | None:
        if self.store is None:
            return None
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM subject_view WHERE id = ?",
                (str(SUBJECT_SELF),),
            ).fetchone()
        if row is None:
            return None
        return Subject.from_record(
            {
                "id": row["id"],
                "role": row["role"] or "agent",
                "capabilities": _loads(row["capabilities"], []),
                "declared_needs": _loads(row["declared_needs"], []),
                "self_model": _loads(row["self_model"], {}),
                "served_counterparts": _loads(row["served_counterparts"], []),
                "known_biases": _loads(row["known_biases"], []),
                "held_at": row["held_at"],
            }
        )

    def _scan_subject(self) -> Subject | None:
        latest: Subject | None = None
        for event in self.event_log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]):
            latest = self._subject_from_event(event) or latest
        return latest

    def _subject_from_event(self, event: CognitiveEvent) -> Subject | None:
        raw = event.payload.get("subject")
        if isinstance(raw, dict):
            return Subject.from_record(raw)
        after = event.payload.get("after")
        if isinstance(after, dict):
            return self.subject_with_self_model(Subject(), SelfModel.from_record(after))
        fields: dict[str, Any] = {}
        for key in ("role", "capabilities", "declared_needs", "held_at"):
            if key in event.payload:
                fields[key] = event.payload[key]
        return Subject.from_record(fields) if fields else None

    def _upsert_self_model_event(self, event: CognitiveEvent) -> None:
        if self.store is None:
            return
        subject = self._subject_from_event(event)
        if subject is None:
            return
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO subject_view
                    (id, role, capabilities, declared_needs, value_lens_id, self_model,
                     served_counterparts, known_biases, held_at, last_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    role = excluded.role,
                    capabilities = excluded.capabilities,
                    declared_needs = excluded.declared_needs,
                    value_lens_id = excluded.value_lens_id,
                    self_model = excluded.self_model,
                    served_counterparts = excluded.served_counterparts,
                    known_biases = excluded.known_biases,
                    held_at = excluded.held_at,
                    last_event_id = excluded.last_event_id
                """,
                (
                    str(subject.id),
                    str(subject.role),
                    _dumps(subject.capabilities),
                    _dumps(subject.declared_needs),
                    str(subject.id),
                    _dumps(subject.self_model.to_record()),
                    _dumps([item.to_record() for item in subject.served_counterparts]),
                    _dumps(subject.known_biases),
                    str(event.timestamp),
                    str(event.id),
                ),
            )

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


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default
