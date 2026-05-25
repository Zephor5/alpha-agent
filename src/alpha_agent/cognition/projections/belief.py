"""SQLite-backed belief projection."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    Belief,
    BeliefId,
    CognitiveEvent,
    CognitiveEventKind,
    CognitiveType,
    Reference,
)
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.cognition.stages.types import AttentionFocus
from alpha_agent.state.store import StateStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS belief_view (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    cognitive_type TEXT NOT NULL,
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.5,
    applicability TEXT NOT NULL DEFAULT '{}',
    value_profile TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    last_event_id TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_belief_view_status
    ON belief_view(status);
CREATE INDEX IF NOT EXISTS idx_belief_view_type
    ON belief_view(cognitive_type, status);

CREATE TABLE IF NOT EXISTS belief_entity_index (
    belief_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_entity_lookup
    ON belief_entity_index(entity_id, belief_id);

CREATE TABLE IF NOT EXISTS belief_about_index (
    belief_id TEXT NOT NULL,
    about_kind TEXT NOT NULL,
    about_id TEXT NOT NULL,
    PRIMARY KEY(belief_id, about_kind, about_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_about_lookup
    ON belief_about_index(about_kind, about_id, belief_id);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-belief-{uuid.uuid4().hex}.db"
    return StateStore(path)


@dataclass(frozen=True)
class BeliefRecallParams:
    focus: AttentionFocus
    counterpart: Reference | None = None
    include_global: bool = True
    types: frozenset[CognitiveType] | None = None
    limit: int = 32


@dataclass(frozen=True)
class BeliefProjectionView:
    beliefs: tuple[Belief, ...] = ()
    status: str = "materialized"


class BeliefProjection(Projection):
    """Materialize belief lifecycle events into queryable SQLite tables."""

    name = "belief"
    handles = frozenset(
        {
            CognitiveEventKind.BELIEF_FORMED,
            CognitiveEventKind.BELIEF_STRENGTHENED,
            CognitiveEventKind.BELIEF_WEAKENED,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            CognitiveEventKind.BELIEF_RETRACTED,
            CognitiveEventKind.BELIEF_ARCHIVED,
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

    def recall(
        self,
        params: BeliefRecallParams | AttentionFocus | object,
        **_: Any,
    ) -> list[Belief]:
        recall_params = self._coerce_params(params)
        if recall_params is None:
            return []
        conditions = ["status = ?"]
        sql_params: list[Any] = ["active"]

        if recall_params.types:
            placeholders = ",".join("?" for _ in recall_params.types)
            conditions.append(f"cognitive_type IN ({placeholders})")
            sql_params.extend(sorted(kind.value for kind in recall_params.types))

        about_clause = self._about_clause(recall_params)
        if about_clause is not None:
            clause, clause_params = about_clause
            conditions.append(clause)
            sql_params.extend(clause_params)

        entity_ids = self._focus_entity_ids(recall_params.focus)
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            like_clause = " OR ".join("normalized_content LIKE ?" for _ in entity_ids)
            conditions.append(
                f"""
                (
                    id IN (
                        SELECT belief_id
                        FROM belief_entity_index
                        WHERE entity_id IN ({placeholders})
                    )
                    OR {like_clause}
                )
                """
            )
            sql_params.extend(entity_ids)
            sql_params.extend(f"%{entity_id}%" for entity_id in entity_ids)

        order_by = "held_since ASC, id ASC"
        if recall_params.counterpart is not None:
            order_by = """
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM belief_about_index
                        WHERE belief_about_index.belief_id = belief_view.id
                          AND belief_about_index.about_kind = ?
                          AND belief_about_index.about_id = ?
                    ) THEN 0
                    ELSE 1
                END,
                held_since ASC,
                id ASC
            """
            sql_params.extend([recall_params.counterpart.kind, recall_params.counterpart.id])

        limit = max(1, recall_params.limit)
        query = f"""
            SELECT *
            FROM belief_view
            WHERE {' AND '.join(conditions)}
            ORDER BY {order_by}
            LIMIT ?
        """
        sql_params.append(limit)
        with self.store.connect() as conn:
            rows = conn.execute(query, sql_params).fetchall()
        return [self._from_row(row) for row in rows]

    def recall_about(self, ref: Reference) -> list[Belief]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT belief_view.*
                FROM belief_view
                JOIN belief_about_index ON belief_about_index.belief_id = belief_view.id
                WHERE belief_view.status = 'active'
                  AND belief_about_index.about_kind = ?
                  AND belief_about_index.about_id = ?
                ORDER BY belief_view.held_since ASC, belief_view.id ASC
                """,
                (ref.kind, ref.id),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def get_by_id(self, belief_id: BeliefId | str) -> Belief | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM belief_view WHERE id = ?",
                (str(belief_id),),
            ).fetchone()
        return self._from_row(row) if row is not None else None

    def list_active(self) -> list[Belief]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM belief_view
                WHERE status = 'active'
                ORDER BY held_since ASC, id ASC
                """
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def apply(self, event: CognitiveEvent) -> None:
        if event.kind not in self.handles:
            return
        if event.kind == CognitiveEventKind.BELIEF_FORMED:
            belief = self._belief_from_payload(event)
            if belief is not None:
                self._upsert_belief(event, belief)
        elif event.kind in {
            CognitiveEventKind.BELIEF_STRENGTHENED,
            CognitiveEventKind.BELIEF_WEAKENED,
        }:
            self._adjust_confidence(event)
        elif event.kind == CognitiveEventKind.BELIEF_SUPERSEDED:
            self._supersede(event)
        elif event.kind == CognitiveEventKind.BELIEF_RETRACTED:
            self._mark_status(event, "retracted")
        elif event.kind == CognitiveEventKind.BELIEF_ARCHIVED:
            self._mark_status(event, "archived")

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM belief_about_index")
            conn.execute("DELETE FROM belief_entity_index")
            conn.execute("DELETE FROM belief_view")

    def view(self) -> BeliefProjectionView:
        return BeliefProjectionView(beliefs=tuple(self.list_active()))

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _rebuild_if_empty(self, event_log: EventLog) -> None:
        if self._view_has_rows():
            return
        for event in event_log.iter(kinds=self.handles):
            self.apply(event)

    def _view_has_rows(self) -> bool:
        with self.store.connect() as conn:
            row = conn.execute("SELECT 1 FROM belief_view LIMIT 1").fetchone()
        return row is not None

    def _coerce_params(
        self,
        params: BeliefRecallParams | AttentionFocus | object,
    ) -> BeliefRecallParams | None:
        if isinstance(params, BeliefRecallParams):
            return params
        if isinstance(params, AttentionFocus):
            return BeliefRecallParams(
                focus=params,
                counterpart=next(
                    (ref for ref in params.entities if ref.kind == "counterpart"),
                    None,
                ),
            )
        return None

    def _about_clause(self, params: BeliefRecallParams) -> tuple[str, list[Any]] | None:
        if params.counterpart is None:
            if not params.include_global:
                return None
            return (
                """
                NOT EXISTS (
                    SELECT 1
                    FROM belief_about_index
                    WHERE belief_about_index.belief_id = belief_view.id
                )
                """,
                [],
            )
        counterpart_clause = """
            EXISTS (
                SELECT 1
                FROM belief_about_index
                WHERE belief_about_index.belief_id = belief_view.id
                  AND belief_about_index.about_kind = ?
                  AND belief_about_index.about_id = ?
            )
        """
        if not params.include_global:
            return counterpart_clause, [params.counterpart.kind, params.counterpart.id]
        return (
            f"""
            (
                {counterpart_clause}
                OR NOT EXISTS (
                    SELECT 1
                    FROM belief_about_index
                    WHERE belief_about_index.belief_id = belief_view.id
                )
            )
            """,
            [params.counterpart.kind, params.counterpart.id],
        )

    def _focus_entity_ids(self, focus: AttentionFocus) -> list[str]:
        return sorted({_normalize_text(ref.id) for ref in focus.entities if ref.kind == "entity"})

    def _belief_from_payload(self, event: CognitiveEvent) -> Belief | None:
        raw = event.payload.get("belief")
        if not isinstance(raw, dict):
            return None
        return Belief.from_record(raw)

    def _upsert_belief(self, event: CognitiveEvent, belief: Belief) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO belief_view
                    (id, record, object, content, normalized_content, cognitive_type, structure,
                     sources, confidence, applicability, value_profile, relations,
                     formed_in_situation, holder_role, action_orientation, update_policy, status,
                     held_since, held_until, supersedes, superseded_by, last_event_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    record = excluded.record,
                    object = excluded.object,
                    content = excluded.content,
                    normalized_content = excluded.normalized_content,
                    cognitive_type = excluded.cognitive_type,
                    structure = excluded.structure,
                    sources = excluded.sources,
                    confidence = excluded.confidence,
                    applicability = excluded.applicability,
                    value_profile = excluded.value_profile,
                    relations = excluded.relations,
                    formed_in_situation = excluded.formed_in_situation,
                    holder_role = excluded.holder_role,
                    action_orientation = excluded.action_orientation,
                    update_policy = excluded.update_policy,
                    status = excluded.status,
                    held_since = excluded.held_since,
                    held_until = excluded.held_until,
                    supersedes = excluded.supersedes,
                    superseded_by = excluded.superseded_by,
                    last_event_id = excluded.last_event_id
                """,
                (
                    str(belief.id),
                    _dumps(belief.to_record()),
                    belief.object,
                    str(belief.content),
                    _normalize_text(belief.content),
                    belief.cognitive_type.value,
                    _dumps(str(belief.structure) if belief.structure is not None else None),
                    _dumps([source.to_record() for source in belief.sources]),
                    float(belief.confidence),
                    str(belief.applicability),
                    _dumps(belief.value_profile.to_record()),
                    _dumps([str(relation) for relation in belief.relations]),
                    belief.formed_in.id,
                    str(belief.holder_role),
                    _dumps([str(action) for action in belief.action_orientation]),
                    str(belief.update_policy),
                    str(belief.status),
                    str(belief.held_since),
                    str(belief.held_until) if belief.held_until is not None else None,
                    belief.supersedes.id if belief.supersedes is not None else None,
                    belief.superseded_by.id if belief.superseded_by is not None else None,
                    str(event.id),
                ),
            )
            conn.execute("DELETE FROM belief_about_index WHERE belief_id = ?", (str(belief.id),))
            conn.execute("DELETE FROM belief_entity_index WHERE belief_id = ?", (str(belief.id),))
            conn.executemany(
                """
                INSERT OR IGNORE INTO belief_about_index (belief_id, about_kind, about_id)
                VALUES (?, ?, ?)
                """,
                [(str(belief.id), ref.kind, ref.id) for ref in belief.about],
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO belief_entity_index (belief_id, entity_id)
                VALUES (?, ?)
                """,
                [(str(belief.id), entity_id) for entity_id in self._belief_entity_ids(belief)],
            )

    def _belief_entity_ids(self, belief: Belief) -> list[str]:
        entity_ids = {_normalize_text(belief.object)}
        entity_ids.update(_normalize_text(ref.id) for ref in belief.about if ref.kind == "entity")
        entity_ids.discard("")
        return sorted(entity_ids)

    def _adjust_confidence(self, event: CognitiveEvent) -> None:
        belief_id = self._payload_belief_id(event)
        if belief_id is None:
            belief = self._belief_from_payload(event)
            if belief is not None:
                self._upsert_belief(event, belief)
            return
        with self.store.transaction() as conn:
            row = conn.execute(
                "SELECT confidence FROM belief_view WHERE id = ?",
                (belief_id,),
            ).fetchone()
            if row is None:
                return
            confidence = event.payload.get("confidence")
            if confidence is None:
                delta = float(event.payload.get("delta", 0.1))
                if event.kind == CognitiveEventKind.BELIEF_WEAKENED:
                    delta = -abs(delta)
                confidence = float(row["confidence"]) + delta
            confidence = max(0.0, min(1.0, float(confidence)))
            conn.execute(
                """
                UPDATE belief_view
                SET confidence = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (confidence, str(event.id), belief_id),
            )

    def _supersede(self, event: CognitiveEvent) -> None:
        belief = self._belief_from_payload(event)
        if belief is not None:
            self._upsert_belief(event, belief)
        old_id = event.payload.get("old_belief_id") or event.payload.get("superseded_belief_id")
        new_id = event.payload.get("new_belief_id") or (
            str(belief.id) if belief is not None else None
        )
        if old_id is None or new_id is None:
            return
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE belief_view
                SET status = 'superseded',
                    held_until = ?,
                    superseded_by = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (str(event.timestamp), str(new_id), str(event.id), str(old_id)),
            )
            conn.execute(
                """
                UPDATE belief_view
                SET status = 'active',
                    supersedes = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (str(old_id), str(event.id), str(new_id)),
            )

    def _mark_status(self, event: CognitiveEvent, status: str) -> None:
        belief_id = self._payload_belief_id(event)
        if belief_id is None:
            return
        with self.store.transaction() as conn:
            conn.execute(
                """
                UPDATE belief_view
                SET status = ?,
                    held_until = ?,
                    last_event_id = ?
                WHERE id = ?
                """,
                (status, str(event.timestamp), str(event.id), belief_id),
            )

    def _payload_belief_id(self, event: CognitiveEvent) -> str | None:
        value = event.payload.get("belief_id") or event.payload.get("id")
        return str(value) if value is not None else None

    def _from_row(self, row: Any) -> Belief:
        record = _loads(row["record"], {})
        if not isinstance(record, dict) or not record:
            raise ValueError(f"belief_view row {row['id']!r} is missing its belief record")
        return Belief.from_record(self._record_with_projection_state(row, record))

    def _record_with_projection_state(self, row: Any, record: dict[str, Any]) -> dict[str, Any]:
        materialized = dict(record)
        materialized.update(
            {
                "id": row["id"],
                "object": row["object"],
                "content": row["content"],
                "cognitive_type": row["cognitive_type"],
                "confidence": float(row["confidence"]),
                "status": row["status"],
                "held_since": row["held_since"],
                "held_until": row["held_until"],
                "superseded_by": (
                    {"kind": "belief", "id": row["superseded_by"]}
                    if row["superseded_by"] is not None
                    else None
                ),
                "supersedes": (
                    {"kind": "belief", "id": row["supersedes"]}
                    if row["supersedes"] is not None
                    else None
                ),
            }
        )
        return materialized
