"""SQLite-backed goal projection."""

from __future__ import annotations

import json
import tempfile
import uuid
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, CounterpartRef, Goal
from alpha_agent.cognition.models._ids import BeliefId, GoalId, Instant
from alpha_agent.cognition.projections.base import EventProjection
from alpha_agent.state.store import StateStore

ACTIVE_GOAL_LIMIT = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS goal_view (
    id TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    target_outcome TEXT NOT NULL DEFAULT '',
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'user',
    for_counterpart TEXT,
    linked_belief_ids TEXT NOT NULL DEFAULT '[]',
    last_drive_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_goal_status_priority
    ON goal_view(status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_goal_for_counterpart
    ON goal_view(for_counterpart, status);
"""


class GoalProjection(EventProjection):
    """Materialize goal lifecycle events into a deterministic view."""

    name = "goal"
    handles = frozenset(
        {
            CognitiveEventKind.GOAL_SET,
            CognitiveEventKind.GOAL_SATISFIED,
            CognitiveEventKind.GOAL_ABANDONED,
            CognitiveEventKind.GOAL_PROGRESSED,
        }
    )

    def __init__(
        self,
        store: StateStore | None = None,
        *,
        event_log: EventLog | None = None,
        auto_rebuild: bool = False,
        active_limit: int = ACTIVE_GOAL_LIMIT,
    ):
        self.store = store or _temporary_store()
        self.active_limit = active_limit
        self.store.initialize()
        self._ensure_schema()
        if auto_rebuild and event_log is not None:
            self._rebuild_if_empty(event_log)

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.GOAL_SET:
            goal = self._goal_from_event(event)
            if goal is not None:
                self._upsert(event, goal)
        elif event.kind == CognitiveEventKind.GOAL_SATISFIED:
            self._mark_status(event, "satisfied")
        elif event.kind == CognitiveEventKind.GOAL_ABANDONED:
            self._mark_status(event, "abandoned")
        elif event.kind == CognitiveEventKind.GOAL_PROGRESSED:
            self._progress(event)

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM goal_view")

    def view(self) -> tuple[Goal, ...]:
        return tuple(self.list_all())

    def active(self) -> list[Goal]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM goal_view
                WHERE status = 'active'
                ORDER BY priority DESC,
                         CASE WHEN last_drive_at IS NULL THEN 0 ELSE 1 END ASC,
                         COALESCE(last_drive_at, '') ASC,
                         updated_at ASC,
                         id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def list_all(self) -> list[Goal]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM goal_view
                ORDER BY created_at ASC, id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, goal_id: GoalId | str) -> Goal | None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT * FROM goal_view WHERE id = ?", (str(goal_id),)).fetchone()
        return self._from_row(row) if row is not None else None

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _rebuild_if_empty(self, event_log: EventLog) -> None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM goal_view LIMIT 1").fetchone()
        if row is not None:
            return
        for event in event_log.iter(kinds=self.handles):
            self.apply(event)

    def _goal_from_event(self, event: CognitiveEvent) -> Goal | None:
        raw = event.payload.get("goal")
        if not isinstance(raw, dict):
            return None
        return Goal.from_record(raw)

    def _upsert(self, event: CognitiveEvent, goal: Goal) -> None:
        with self.store.transaction() as conn:
            if self._would_exceed_active_limit(conn, goal):
                return
            conn.execute(
                """
                INSERT INTO goal_view
                    (id, description, target_outcome, priority, status, source,
                     for_counterpart, linked_belief_ids, last_drive_at, created_at,
                     updated_at, last_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    description = excluded.description,
                    target_outcome = excluded.target_outcome,
                    priority = excluded.priority,
                    status = excluded.status,
                    source = excluded.source,
                    for_counterpart = excluded.for_counterpart,
                    linked_belief_ids = excluded.linked_belief_ids,
                    last_drive_at = excluded.last_drive_at,
                    updated_at = excluded.updated_at,
                    last_event_id = excluded.last_event_id
                """,
                (
                    str(goal.id),
                    goal.description,
                    goal.target_outcome,
                    int(goal.priority),
                    goal.status,
                    goal.source,
                    goal.for_counterpart.id if goal.for_counterpart else None,
                    _dumps([str(item) for item in goal.linked_belief_ids]),
                    str(goal.last_drive_at) if goal.last_drive_at else None,
                    str(goal.created_at),
                    str(goal.updated_at),
                    str(event.id),
                ),
            )

    def _would_exceed_active_limit(self, conn: Any, goal: Goal) -> bool:
        if goal.status != "active":
            return False
        existing = conn.execute(
            "SELECT status FROM goal_view WHERE id = ?",
            (str(goal.id),),
        ).fetchone()
        if existing is not None and existing["status"] == "active":
            return False
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM goal_view WHERE status = 'active'"
        ).fetchone()
        return int(row["count"]) >= self.active_limit

    def _mark_status(self, event: CognitiveEvent, status: str) -> None:
        goal_id = event.payload.get("goal_id")
        if goal_id is None:
            return
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE goal_view
                SET status = ?,
                    updated_at = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (status, str(event.timestamp), str(event.id), str(goal_id)),
            )

    def _progress(self, event: CognitiveEvent) -> None:
        goal_id = event.payload.get("goal_id")
        if goal_id is None:
            return
        drive_progress = bool(event.payload.get("drive_progress"))
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE goal_view
                SET updated_at = ?,
                    last_drive_at = CASE WHEN ? THEN ? ELSE last_drive_at END,
                    last_event_id = ?
                WHERE id = ?
                """,
                (
                    str(event.timestamp),
                    1 if drive_progress else 0,
                    str(event.timestamp),
                    str(event.id),
                    str(goal_id),
                ),
            )

    def _from_row(self, row: Any) -> Goal:
        counterpart = (
            CounterpartRef("counterpart", row["for_counterpart"])
            if row["for_counterpart"] is not None
            else None
        )
        return Goal(
            id=GoalId(row["id"]),
            description=row["description"],
            target_outcome=row["target_outcome"],
            priority=int(row["priority"]),
            status=row["status"],
            source=row["source"],
            linked_belief_ids=[
                BeliefId(str(item)) for item in _loads(row["linked_belief_ids"], [])
            ],
            for_counterpart=counterpart,
            created_at=Instant(row["created_at"]),
            updated_at=Instant(row["updated_at"]),
            last_drive_at=Instant(row["last_drive_at"]) if row["last_drive_at"] else None,
        )


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-goal-{uuid.uuid4().hex}.db"
    return StateStore(path)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default
