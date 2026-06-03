"""SQLite-backed reflection projection."""

from __future__ import annotations

import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, Reflection
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.state.store import StateStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reflection_view (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'L1',
    kind TEXT NOT NULL,
    severity TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    finding TEXT NOT NULL,
    suggested_remedy TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_reflection_severity
    ON reflection_view(severity, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reflection_kind
    ON reflection_view(kind, created_at DESC);
"""


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-reflection-{uuid.uuid4().hex}.db"
    return StateStore(path)


@dataclass(frozen=True)
class ReflectionProjectionView:
    reflections: tuple[Reflection, ...] = ()


class ReflectionProjection(Projection):
    """Materialize reflected events into queryable reflection rows."""

    name = "reflection"
    handles = frozenset({CognitiveEventKind.REFLECTED})

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        event_log: EventLog | None = None,
        auto_rebuild: bool = False,
    ):
        self.store = store or _temporary_store()
        self.store.initialize()
        self._ensure_schema()
        if auto_rebuild and event_log is not None:
            self._rebuild_if_empty(event_log)

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind not in self.handles:
            return
        turn_id = str(event.payload.get("turn_id") or "")
        records = event.payload.get("reflections") or []
        if not isinstance(records, list):
            return
        with self.store.transaction() as conn:
            for record in records:
                if not isinstance(record, dict):
                    continue
                reflection = Reflection.from_record(record)
                target_kind, target_id = target_to_parts(reflection.target)
                conn.execute(
                    """
                    INSERT INTO reflection_view
                        (id, turn_id, level, kind, severity, target_kind, target_id,
                         finding, suggested_remedy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        turn_id = excluded.turn_id,
                        level = excluded.level,
                        kind = excluded.kind,
                        severity = excluded.severity,
                        target_kind = excluded.target_kind,
                        target_id = excluded.target_id,
                        finding = excluded.finding,
                        suggested_remedy = excluded.suggested_remedy,
                        created_at = excluded.created_at
                    """,
                    (
                        str(reflection.id),
                        turn_id,
                        reflection.level,
                        str(reflection.kind),
                        str(reflection.severity),
                        target_kind,
                        target_id,
                        str(reflection.finding),
                        str(reflection.suggested_remedy),
                        str(reflection.created_at),
                    ),
                )

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM reflection_view")

    def view(self) -> ReflectionProjectionView:
        return ReflectionProjectionView(reflections=tuple(self.list_recent(last=100)))

    def list_recent(
        self,
        *,
        last: int = 20,
        severity: str | None = None,
        kind: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> list[Reflection]:
        conditions: list[str] = []
        params: list[Any] = []
        if severity is not None:
            conditions.append("severity = ?")
            params.append(severity)
        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)
        if since is not None:
            conditions.append("created_at >= ?")
            params.append(since)
        if until is not None:
            conditions.append("created_at <= ?")
            params.append(until)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit = max(1, last)
        with self.store.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM reflection_view
                {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def by_severity(self, severity: str, *, last: int = 20) -> list[Reflection]:
        return self.list_recent(last=last, severity=severity)

    def by_kind(self, kind: str, *, last: int = 20) -> list[Reflection]:
        return self.list_recent(last=last, kind=kind)

    def for_target(self, target_kind: str, target_id: str, *, last: int = 20) -> list[Reflection]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM reflection_view
                WHERE target_kind = ? AND target_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (target_kind, target_id, max(1, last)),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _rebuild_if_empty(self, event_log: EventLog) -> None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM reflection_view LIMIT 1").fetchone()
        if row is not None:
            return
        for event in event_log.iter(kinds=self.handles):
            self.apply(event)

    def _from_row(self, row: Any) -> Reflection:
        return Reflection.from_record(
            {
                "id": row["id"],
                "level": row["level"],
                "kind": row["kind"],
                "severity": row["severity"],
                "target": f"{row['target_kind']}:{row['target_id']}",
                "finding": row["finding"],
                "suggested_remedy": row["suggested_remedy"],
                "created_at": row["created_at"],
            }
        )


def target_to_parts(target: object) -> tuple[str, str]:
    value = str(target)
    if ":" not in value:
        return "unknown", value
    kind, target_id = value.split(":", 1)
    return kind, target_id
