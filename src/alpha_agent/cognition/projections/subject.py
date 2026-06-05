"""Subject identity projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import Subject
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.state.store import StateStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS subject_view (
    id TEXT PRIMARY KEY,
    role TEXT,
    capabilities TEXT NOT NULL DEFAULT '[]',
    declared_needs TEXT NOT NULL DEFAULT '[]',
    membership TEXT NOT NULL DEFAULT '[]',
    served_counterparts TEXT NOT NULL DEFAULT '[]',
    held_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
)
"""


@dataclass(frozen=True)
class SubjectProjectionView:
    subject: Subject
    status: str = "materialized"


class SubjectProjection(Projection):
    """Expose the single subject identity without deterministic self-model state."""

    name = "subject"
    handles = frozenset()

    def __init__(self, event_log: EventLog, store: StateStore | None = None):
        self.event_log = event_log
        self.store = store or getattr(event_log, "store", None)
        if self.store is not None:
            self._ensure_schema()

    def current(self) -> Subject:
        return self._stored_subject() or Subject()

    def reset(self) -> None:
        if self.store is not None:
            self._ensure_schema()
            with self.store.transaction() as conn:
                conn.execute("DELETE FROM subject_view")

    def view(self) -> SubjectProjectionView:
        return SubjectProjectionView(subject=self.current())

    def _ensure_schema(self) -> None:
        if self.store is None:
            return
        with self.store.transaction() as conn:
            conn.execute(_SCHEMA)

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
                "membership": _loads(row["membership"], []),
                "served_counterparts": _loads(row["served_counterparts"], []),
                "held_at": row["held_at"],
            }
        )


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default
