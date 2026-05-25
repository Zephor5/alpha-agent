"""SQLite-backed cognitive event log."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    EventId,
    Instant,
    Reference,
    SubjectRef,
)
from alpha_agent.state.store import StateStore


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str) -> Any:
    return json.loads(value)


class SQLiteEventLog(EventLog):
    """Append-only event log using the shared StateStore database."""

    def __init__(self, store: StateStore):
        self.store = store

    def append(self, event: CognitiveEvent) -> EventId:
        with self.store.immediate_transaction() as conn:
            ordinal = self._next_ordinal(conn, event.subject)
            conn.execute(
                """
                INSERT INTO cognitive_events
                    (id, kind, subject_id, subject_version, situation_id, actor, rationale,
                     inputs, outputs, causal_parents, payload, timestamp, ordinal, schema_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.id),
                    event.kind.value,
                    event.subject.id,
                    event.subject_version,
                    event.situation.id if event.situation is not None else None,
                    _dumps(event.actor.to_record()),
                    str(event.rationale),
                    _dumps([ref.to_record() for ref in event.inputs]),
                    _dumps([ref.to_record() for ref in event.outputs]),
                    _dumps([str(parent) for parent in event.causal_parents]),
                    _dumps(event.payload),
                    str(event.timestamp),
                    ordinal,
                    event.schema_version,
                ),
            )
        return event.id

    def get(self, event_id: EventId) -> CognitiveEvent:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognitive_events WHERE id = ?",
                (str(event_id),),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown cognitive event: {event_id}")
        return self._event_from_row(row)

    def iter(
        self,
        *,
        subject: SubjectRef | None = None,
        kinds: Iterable[CognitiveEventKind] | None = None,
        since: Instant | None = None,
        until: Instant | None = None,
    ) -> Iterator[CognitiveEvent]:
        conditions: list[str] = []
        params: list[Any] = []
        if subject is not None:
            conditions.append("subject_id = ?")
            params.append(subject.id)
        if kinds is not None:
            kind_values = [kind.value for kind in kinds]
            if not kind_values:
                return
            placeholders = ",".join("?" for _ in kind_values)
            conditions.append(f"kind IN ({placeholders})")
            params.extend(kind_values)
        if since is not None:
            conditions.append("timestamp >= ?")
            params.append(str(since))
        if until is not None:
            conditions.append("timestamp <= ?")
            params.append(str(until))
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT *
            FROM cognitive_events
            {where}
            ORDER BY subject_id ASC, ordinal ASC
        """
        with self.store.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        for row in rows:
            yield self._event_from_row(row)

    def length(self, *, subject: SubjectRef | None = None) -> int:
        if subject is None:
            query = "SELECT COUNT(*) AS count FROM cognitive_events"
            params: tuple[str, ...] = ()
        else:
            query = "SELECT COUNT(*) AS count FROM cognitive_events WHERE subject_id = ?"
            params = (subject.id,)
        with self.store.connect() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["count"])

    def _next_ordinal(self, conn: sqlite3.Connection, subject: SubjectRef) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(ordinal), 0) AS ordinal
            FROM cognitive_events
            WHERE subject_id = ?
            """,
            (subject.id,),
        ).fetchone()
        return int(row["ordinal"]) + 1

    def _event_from_row(self, row: sqlite3.Row) -> CognitiveEvent:
        record = {
            "id": row["id"],
            "kind": row["kind"],
            "subject": {"kind": "subject", "id": row["subject_id"]},
            "subject_version": int(row["subject_version"]),
            "situation": (
                {"kind": "situation", "id": row["situation_id"]}
                if row["situation_id"] is not None
                else None
            ),
            "inputs": [Reference.from_record(item) for item in _loads(row["inputs"])],
            "outputs": [Reference.from_record(item) for item in _loads(row["outputs"])],
            "rationale": row["rationale"],
            "timestamp": row["timestamp"],
            "actor": Reference.from_record(_loads(row["actor"])),
            "causal_parents": _loads(row["causal_parents"]),
            "payload": _loads(row["payload"]),
            "schema_version": int(row["schema_version"]),
        }
        return CognitiveEvent.from_record(record)
