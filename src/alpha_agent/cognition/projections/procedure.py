"""SQLite-backed procedure projection."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    EventId,
    NLStatement,
    Procedure,
    ProcedureId,
    ProcedureRef,
    Step,
    TriggerPattern,
)
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.state.store import StateStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedure_view (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger_pattern TEXT NOT NULL,
    steps TEXT NOT NULL DEFAULT '[]',
    expected_outcome TEXT NOT NULL DEFAULT '',
    learned_from_event_ids TEXT NOT NULL DEFAULT '[]',
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_procedure_trigger
    ON procedure_view(trigger_pattern);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_list(value: str | None) -> list[Any]:
    if not value:
        return []
    loaded = json.loads(value)
    return loaded if isinstance(loaded, list) else []


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_:-]+", value.casefold()) if len(token) > 2}


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-procedure-{uuid.uuid4().hex}.db"
    return StateStore(path)


@dataclass(frozen=True)
class ProcedureProjectionView:
    matched: tuple[ProcedureRef, ...] = ()
    status: str = "materialized"


class ProcedureProjection(Projection):
    """Materialize learned procedures and return deterministic trigger matches."""

    name = "procedure"
    handles = frozenset(
        {
            CognitiveEventKind.PROCEDURE_LEARNED,
            CognitiveEventKind.PROCEDURE_STRENGTHENED,
            CognitiveEventKind.PROCEDURE_WEAKENED,
            CognitiveEventKind.PROCEDURE_MATCHED,
        }
    )
    status = "materialized"

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

    def match(self, context: Any, *_args: Any, **_kwargs: Any) -> list[ProcedureRef]:
        source_text = _match_text(context)
        if not source_text:
            return []
        source_tokens = _tokens(source_text)
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM procedure_view
                WHERE status = 'active'
                ORDER BY confidence DESC, success_count DESC, updated_at DESC, id ASC
                """
            ).fetchall()
        matched: list[ProcedureRef] = []
        normalized_source = _normalize(source_text)
        for row in rows:
            trigger = _normalize(row["trigger_pattern"])
            trigger_tokens = _tokens(trigger)
            if trigger and (trigger in normalized_source or trigger_tokens & source_tokens):
                matched.append(ProcedureRef("procedure", row["id"]))
        return matched[:5]

    def list_active(self) -> list[Procedure]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM procedure_view
                WHERE status = 'active'
                ORDER BY confidence DESC, updated_at DESC, id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get(self, procedure_id: ProcedureId | str) -> Procedure | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM procedure_view WHERE id = ?",
                (str(procedure_id),),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind == CognitiveEventKind.PROCEDURE_LEARNED:
            procedure = self._procedure_from_payload(event)
            if procedure is not None:
                self._upsert(event, procedure)
        elif event.kind in {
            CognitiveEventKind.PROCEDURE_STRENGTHENED,
            CognitiveEventKind.PROCEDURE_WEAKENED,
        }:
            self._adjust(event)

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM procedure_view")

    def view(self) -> ProcedureProjectionView:
        return ProcedureProjectionView(
            matched=tuple(ProcedureRef("procedure", str(item.id)) for item in self.list_active())
        )

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _rebuild_if_empty(self, event_log: EventLog) -> None:
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM procedure_view LIMIT 1").fetchone()
        if row is not None:
            return
        for event in event_log.iter(kinds=self.handles):
            self.apply(event)

    def _procedure_from_payload(self, event: CognitiveEvent) -> Procedure | None:
        raw = event.payload.get("procedure")
        if not isinstance(raw, dict):
            return None
        return Procedure.from_record(raw)

    def _upsert(self, event: CognitiveEvent, procedure: Procedure) -> None:
        trigger = _normalize(procedure.trigger)
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO procedure_view
                    (id, name, trigger_pattern, steps, expected_outcome, learned_from_event_ids,
                     success_count, failure_count, confidence, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    trigger_pattern = excluded.trigger_pattern,
                    steps = excluded.steps,
                    expected_outcome = excluded.expected_outcome,
                    learned_from_event_ids = excluded.learned_from_event_ids,
                    success_count = excluded.success_count,
                    failure_count = excluded.failure_count,
                    confidence = excluded.confidence,
                    status = 'active',
                    updated_at = excluded.updated_at
                """,
                (
                    str(procedure.id),
                    str(event.payload.get("name") or f"Procedure for {trigger}")[:120],
                    trigger,
                    _dumps([str(step) for step in procedure.steps]),
                    str(procedure.expected_outcome),
                    _dumps([str(item) for item in procedure.learned_from]),
                    int(procedure.success_count),
                    int(procedure.failure_count),
                    float(procedure.confidence),
                    str(event.timestamp),
                    str(event.timestamp),
                ),
            )

    def _adjust(self, event: CognitiveEvent) -> None:
        procedure_id = event.payload.get("procedure_id") or event.payload.get("id")
        if procedure_id is None:
            return
        delta = float(event.payload.get("delta", 0.1))
        if event.kind == CognitiveEventKind.PROCEDURE_WEAKENED:
            delta = -abs(delta)
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT confidence, success_count, failure_count FROM procedure_view WHERE id = ?",
                (str(procedure_id),),
            ).fetchone()
            if row is None:
                return
            confidence = max(0.0, min(1.0, float(row["confidence"]) + delta))
            success_count = int(row["success_count"]) + (
                1 if event.kind == CognitiveEventKind.PROCEDURE_STRENGTHENED else 0
            )
            failure_count = int(row["failure_count"]) + (
                1 if event.kind == CognitiveEventKind.PROCEDURE_WEAKENED else 0
            )
            conn.execute(
                """
                UPDATE procedure_view
                SET confidence = ?,
                    success_count = ?,
                    failure_count = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (confidence, success_count, failure_count, str(event.timestamp), str(procedure_id)),
            )

    def _from_row(self, row: Any) -> Procedure:
        return Procedure(
            id=ProcedureId(row["id"]),
            trigger=TriggerPattern(row["trigger_pattern"]),
            steps=[Step(str(item)) for item in _loads_list(row["steps"])],
            expected_outcome=NLStatement(row["expected_outcome"]),
            learned_from=[
                EventId(str(item)) for item in _loads_list(row["learned_from_event_ids"])
            ],
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            confidence=float(row["confidence"]),
        )


def _match_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    parts: list[str] = []
    for attr in ("source_text", "claim", "action"):
        value = getattr(context, attr, None)
        if value:
            parts.append(str(value))
    for attr in ("novel_claims", "salient_claims"):
        values = getattr(context, attr, None)
        if values:
            parts.extend(str(item) for item in values)
    if isinstance(context, list):
        parts.extend(str(getattr(item, "claim", item)) for item in context)
    return " ".join(parts)
