"""SQLite-backed strategy override projection."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    CounterpartRef,
    Instant,
    StrategyId,
    StrategyOverride,
)
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.state.store import StateStore
from alpha_agent.utils.time import utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    target_stages TEXT NOT NULL DEFAULT '[]',
    for_counterpart TEXT,
    set_by TEXT NOT NULL,
    set_at TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_strategy_status_validity
    ON strategy_view(status, valid_until);
CREATE INDEX IF NOT EXISTS idx_strategy_for_counterpart
    ON strategy_view(for_counterpart, status);
"""


def strategy_applies_to_counterpart(
    strategy: StrategyOverride,
    counterpart: CounterpartRef | None,
) -> bool:
    if strategy.for_counterpart is None:
        return True
    if counterpart is None:
        return False
    return (
        strategy.for_counterpart.kind == counterpart.kind
        and strategy.for_counterpart.id == counterpart.id
    )


def strategy_is_active_for_stage(
    strategies: list[StrategyOverride],
    name: str,
    stage: str,
) -> bool:
    return any(
        strategy.name == name
        and (not strategy.target_stages or stage in strategy.target_stages)
        for strategy in strategies
    )


class StrategyProjection(Projection):
    """Materialize active and expired strategy overrides."""

    name = "strategy"
    handles = frozenset(
        {
            CognitiveEventKind.STRATEGY_CHANGED,
            CognitiveEventKind.STRATEGY_EXPIRED,
        }
    )

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
        if event.kind == CognitiveEventKind.STRATEGY_CHANGED:
            strategy = self._strategy_from_event(event)
            if strategy is not None:
                self._upsert(event, strategy)
        elif event.kind == CognitiveEventKind.STRATEGY_EXPIRED:
            strategy_id = event.payload.get("strategy_id")
            if strategy_id is not None:
                self._mark_status(str(strategy_id), "expired", event)

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM strategy_view")

    def view(self) -> tuple[StrategyOverride, ...]:
        return tuple(self.list_all())

    def active(
        self,
        *,
        now: Instant | str | None = None,
        counterpart: CounterpartRef | None = None,
        stage: str | None = None,
    ) -> list[StrategyOverride]:
        current = str(now or utc_now_iso())
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM strategy_view
                WHERE status = 'active'
                  AND valid_until > ?
                ORDER BY set_at ASC, id ASC
                """,
                (current,),
            ).fetchall()
        strategies = [self._from_row(row) for row in rows]
        if counterpart is not None:
            strategies = [
                item for item in strategies if strategy_applies_to_counterpart(item, counterpart)
            ]
        if stage is not None:
            strategies = [
                item for item in strategies if not item.target_stages or stage in item.target_stages
            ]
        return strategies

    def list_all(self) -> list[StrategyOverride]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM strategy_view
                ORDER BY set_at ASC, id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def is_active(self, name: str, *, stage: str | None = None) -> bool:
        return any(item.name == name for item in self.active(stage=stage))

    def expire_due(self, now: Instant | str) -> list[StrategyOverride]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM strategy_view
                WHERE status = 'active'
                  AND valid_until <= ?
                ORDER BY valid_until ASC, id ASC
                """,
                (str(now),),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _rebuild_if_empty(self, event_log: EventLog) -> None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM strategy_view LIMIT 1").fetchone()
        if row is not None:
            return
        for event in event_log.iter(kinds=self.handles):
            self.apply(event)

    def _strategy_from_event(self, event: CognitiveEvent) -> StrategyOverride | None:
        raw = event.payload.get("strategy")
        if not isinstance(raw, dict):
            return None
        return StrategyOverride.from_record(raw)

    def _upsert(self, event: CognitiveEvent, strategy: StrategyOverride) -> None:
        with self.store.transaction() as conn:
            if self._would_exceed_active_limit(conn, strategy):
                return
            conn.execute(
                """
                INSERT INTO strategy_view
                    (id, name, payload, target_stages, for_counterpart, set_by, set_at,
                     valid_until, status, last_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    payload = excluded.payload,
                    target_stages = excluded.target_stages,
                    for_counterpart = excluded.for_counterpart,
                    set_by = excluded.set_by,
                    set_at = excluded.set_at,
                    valid_until = excluded.valid_until,
                    status = excluded.status,
                    last_event_id = excluded.last_event_id
                """,
                (
                    str(strategy.id),
                    strategy.name,
                    _dumps(strategy.payload),
                    _dumps(strategy.target_stages),
                    strategy.for_counterpart.id if strategy.for_counterpart else None,
                    strategy.set_by,
                    str(strategy.set_at),
                    str(strategy.valid_until),
                    str(event.id),
                ),
            )

    def _would_exceed_active_limit(self, conn: Any, strategy: StrategyOverride) -> bool:
        existing = conn.execute(
            "SELECT status, valid_until FROM strategy_view WHERE id = ?",
            (str(strategy.id),),
        ).fetchone()
        if (
            existing is not None
            and existing["status"] == "active"
            and str(existing["valid_until"]) > str(strategy.set_at)
        ):
            return False
        if str(strategy.valid_until) <= str(strategy.set_at):
            return False
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM strategy_view
            WHERE status = 'active'
              AND valid_until > ?
              AND id != ?
            """,
            (str(strategy.set_at), str(strategy.id)),
        ).fetchone()
        return int(row["count"]) >= 5

    def _mark_status(self, strategy_id: str, status: str, event: CognitiveEvent) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE strategy_view
                SET status = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (status, str(event.id), strategy_id),
            )

    def _from_row(self, row: Any) -> StrategyOverride:
        counterpart = (
            CounterpartRef("counterpart", row["for_counterpart"])
            if row["for_counterpart"] is not None
            else None
        )
        return StrategyOverride(
            id=StrategyId(row["id"]),
            name=row["name"],
            payload=_loads(row["payload"], {}),
            target_stages=_loads(row["target_stages"], []),
            for_counterpart=counterpart,
            set_by=row["set_by"],
            set_at=Instant(row["set_at"]),
            valid_until=Instant(row["valid_until"]),
        )


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-strategy-{uuid.uuid4().hex}.db"
    return StateStore(path)


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
