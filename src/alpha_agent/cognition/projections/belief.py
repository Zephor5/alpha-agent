"""SQLite-backed belief projection."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from collections.abc import Sequence
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
from alpha_agent.cognition.search_tokenizer import tokenize_mixed_text
from alpha_agent.cognition.value.profile_derivation import derive_value_profile
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

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_terms_fts
USING fts5(
    belief_id UNINDEXED,
    search_terms,
    object,
    tokenize = "unicode61 remove_diacritics 1 tokenchars '_-#./:+'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_trigram_fts
USING fts5(
    belief_id UNINDEXED,
    content,
    object,
    normalized_content,
    tokenize = "trigram"
);
"""

_CANDIDATE_REASON_ORDER = {
    "entity_exact": 0,
    "object_exact": 1,
    "object_partial": 2,
    "term_fts": 3,
    "trigram_fts": 4,
    "substring": 5,
}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def build_term_fts_query(tokens: Sequence[str]) -> str:
    """Build a safe FTS5 OR query for token/term lookup."""

    return _build_fts_phrase_query(tokens, min_length=1)


def build_trigram_fts_query(probes: Sequence[str]) -> str:
    """Build a safe FTS5 OR query for trigram lookup."""

    return _build_fts_phrase_query(probes, min_length=3)


def _build_fts_phrase_query(values: Sequence[str], *, min_length: int) -> str:
    phrases: list[str] = []
    seen: set[str] = set()
    for value in values:
        probe = _normalize_text(value)
        if len(probe) < min_length or probe in seen:
            continue
        seen.add(probe)
        phrases.append(_fts_phrase(probe))
    return " OR ".join(phrases)


def _fts_phrase(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def _escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _temporary_store() -> StateStore:
    path = f"{tempfile.gettempdir()}/alpha-agent-belief-{uuid.uuid4().hex}.db"
    return StateStore(path)


@dataclass(frozen=True)
class BeliefRecallParams:
    entities: tuple[Reference, ...] = ()
    counterpart: Reference | None = None
    include_global: bool = True
    types: frozenset[CognitiveType] | None = None
    limit: int = 32


@dataclass(frozen=True)
class BeliefSearchParams:
    query: str
    keywords: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    counterpart: Reference | None = None
    include_global: bool = True
    types: frozenset[CognitiveType] | None = None
    limit: int = 64


@dataclass(frozen=True)
class BeliefSearchCandidate:
    belief: Belief
    reasons: tuple[str, ...]
    term_rank: float | None = None
    trigram_rank: float | None = None


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
        params: BeliefRecallParams,
        **_: Any,
    ) -> list[Belief]:
        conditions = ["status = ?"]
        sql_params: list[Any] = ["active"]

        if params.types:
            placeholders = ",".join("?" for _ in params.types)
            conditions.append(f"cognitive_type IN ({placeholders})")
            sql_params.extend(sorted(kind.value for kind in params.types))

        about_clause = self._about_clause(params)
        if about_clause is not None:
            clause, clause_params = about_clause
            conditions.append(clause)
            sql_params.extend(clause_params)

        entity_ids = self._entity_ids(params.entities)
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
        if params.counterpart is not None:
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
            sql_params.extend([params.counterpart.kind, params.counterpart.id])

        limit = max(1, params.limit)
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

    def recall_candidates(self, params: BeliefSearchParams) -> list[BeliefSearchCandidate]:
        """Recall active beliefs with explicit search match signals."""

        limit = max(1, int(params.limit))
        source_limit = max(limit * 4, 32)
        candidates: dict[str, dict[str, Any]] = {}
        with self.store.connect() as conn:
            self._collect_entity_exact_candidates(conn, params, candidates, source_limit)
            self._collect_object_candidates(conn, params, candidates, source_limit)
            self._collect_term_fts_candidates(conn, params, candidates, source_limit)
            self._collect_trigram_fts_candidates(conn, params, candidates, source_limit)
            self._collect_substring_candidates(conn, params, candidates, source_limit)

        merged = [
            BeliefSearchCandidate(
                belief=item["belief"],
                reasons=tuple(
                    sorted(
                        item["reasons"],
                        key=lambda reason: _CANDIDATE_REASON_ORDER.get(reason, 100),
                    )
                ),
                term_rank=item["term_rank"],
                trigram_rank=item["trigram_rank"],
            )
            for item in candidates.values()
        ]
        merged.sort(key=self._candidate_sort_key)
        return merged[:limit]

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
            conn.execute("DELETE FROM belief_search_trigram_fts")
            conn.execute("DELETE FROM belief_search_terms_fts")
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

    def _entity_ids(self, entities: Sequence[Reference]) -> list[str]:
        return sorted({_normalize_text(ref.id) for ref in entities if ref.kind == "entity"})

    def _belief_from_payload(self, event: CognitiveEvent) -> Belief | None:
        raw = event.payload.get("belief")
        if not isinstance(raw, dict):
            return None
        return Belief.from_record(raw)

    def _upsert_belief(self, event: CognitiveEvent, belief: Belief) -> None:
        with self.store.transaction() as conn:
            self._upsert_belief_row(conn, event, belief)

    def _upsert_belief_row(self, conn: Any, event: CognitiveEvent, belief: Belief) -> Belief:
        belief = _belief_with_derived_profile(belief)
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
        entity_ids = self._belief_entity_ids(belief)
        conn.executemany(
            """
            INSERT OR IGNORE INTO belief_entity_index (belief_id, entity_id)
            VALUES (?, ?)
            """,
            [(str(belief.id), entity_id) for entity_id in entity_ids],
        )
        self._replace_belief_fts(conn, belief, entity_ids)
        return belief

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
        old_id = event.payload.get("old_belief_id") or event.payload.get("superseded_belief_id")
        new_id = str(belief.id) if belief is not None else event.payload.get("new_belief_id")
        if old_id is None or new_id is None:
            return
        with self.store.transaction() as conn:
            if belief is not None:
                belief = self._upsert_belief_row(conn, event, belief)
                new_id = str(belief.id)
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
            self._delete_belief_fts(conn, str(old_id))
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
            self._delete_belief_fts(conn, belief_id)

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

    def _collect_entity_exact_candidates(
        self,
        conn: Any,
        params: BeliefSearchParams,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> None:
        entity_ids = _normalized_unique(self._search_probes(params))
        if not entity_ids:
            return
        conditions, sql_params = self._search_filter_clause(params)
        placeholders = ",".join("?" for _ in entity_ids)
        query = f"""
            SELECT DISTINCT belief_view.*
            FROM belief_view
            JOIN belief_entity_index
              ON belief_entity_index.belief_id = belief_view.id
            WHERE {' AND '.join(conditions)}
              AND belief_entity_index.entity_id IN ({placeholders})
            ORDER BY belief_view.held_since ASC, belief_view.id ASC
            LIMIT ?
        """
        rows = conn.execute(query, [*sql_params, *entity_ids, limit]).fetchall()
        for row in rows:
            self._merge_candidate(candidates, row, "entity_exact")

    def _collect_object_candidates(
        self,
        conn: Any,
        params: BeliefSearchParams,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> None:
        probes = self._search_probes(params)
        exact_probes = [probe for probe in probes if probe]
        if exact_probes:
            conditions, sql_params = self._search_filter_clause(params)
            placeholders = ",".join("?" for _ in exact_probes)
            query = f"""
                SELECT belief_view.*
                FROM belief_view
                WHERE {' AND '.join(conditions)}
                  AND lower(belief_view.object) IN ({placeholders})
                ORDER BY belief_view.held_since ASC, belief_view.id ASC
                LIMIT ?
            """
            rows = conn.execute(query, [*sql_params, *exact_probes, limit]).fetchall()
            for row in rows:
                self._merge_candidate(candidates, row, "object_exact")

        partial_probes = [probe for probe in probes if len(probe) >= 3]
        for probe in partial_probes:
            conditions, sql_params = self._search_filter_clause(params)
            like_probe = f"%{_escape_like(probe)}%"
            rows = conn.execute(
                f"""
                SELECT belief_view.*
                FROM belief_view
                WHERE {' AND '.join(conditions)}
                  AND lower(belief_view.object) LIKE ? ESCAPE '!'
                ORDER BY belief_view.held_since ASC, belief_view.id ASC
                LIMIT ?
                """,
                [*sql_params, like_probe, limit],
            ).fetchall()
            for row in rows:
                self._merge_candidate(candidates, row, "object_partial")

    def _collect_term_fts_candidates(
        self,
        conn: Any,
        params: BeliefSearchParams,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> None:
        query_text = build_term_fts_query(self._term_tokens(params))
        if not query_text:
            return
        conditions, sql_params = self._search_filter_clause(params)
        rows = conn.execute(
            f"""
            SELECT belief_view.*, bm25(belief_search_terms_fts) AS term_rank
            FROM belief_search_terms_fts
            JOIN belief_view
              ON belief_view.id = belief_search_terms_fts.belief_id
            WHERE {' AND '.join(conditions)}
              AND belief_search_terms_fts MATCH ?
            ORDER BY term_rank ASC, belief_view.held_since ASC, belief_view.id ASC
            LIMIT ?
            """,
            [*sql_params, query_text, limit],
        ).fetchall()
        for row in rows:
            self._merge_candidate(
                candidates,
                row,
                "term_fts",
                term_rank=float(row["term_rank"]),
            )

    def _collect_trigram_fts_candidates(
        self,
        conn: Any,
        params: BeliefSearchParams,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> None:
        query_text = build_trigram_fts_query(self._trigram_probes(params))
        if not query_text:
            return
        conditions, sql_params = self._search_filter_clause(params)
        rows = conn.execute(
            f"""
            SELECT belief_view.*, bm25(belief_search_trigram_fts) AS trigram_rank
            FROM belief_search_trigram_fts
            JOIN belief_view
              ON belief_view.id = belief_search_trigram_fts.belief_id
            WHERE {' AND '.join(conditions)}
              AND belief_search_trigram_fts MATCH ?
            ORDER BY trigram_rank ASC, belief_view.held_since ASC, belief_view.id ASC
            LIMIT ?
            """,
            [*sql_params, query_text, limit],
        ).fetchall()
        for row in rows:
            self._merge_candidate(
                candidates,
                row,
                "trigram_fts",
                trigram_rank=float(row["trigram_rank"]),
            )

    def _collect_substring_candidates(
        self,
        conn: Any,
        params: BeliefSearchParams,
        candidates: dict[str, dict[str, Any]],
        limit: int,
    ) -> None:
        probes = [probe for probe in self._search_probes(params) if len(probe) >= 3]
        for probe in probes:
            conditions, sql_params = self._search_filter_clause(params)
            like_probe = f"%{_escape_like(probe)}%"
            rows = conn.execute(
                f"""
                SELECT belief_view.*
                FROM belief_view
                WHERE {' AND '.join(conditions)}
                  AND (
                    belief_view.normalized_content LIKE ? ESCAPE '!'
                    OR lower(belief_view.object) LIKE ? ESCAPE '!'
                  )
                ORDER BY belief_view.held_since ASC, belief_view.id ASC
                LIMIT ?
                """,
                [*sql_params, like_probe, like_probe, limit],
            ).fetchall()
            for row in rows:
                self._merge_candidate(candidates, row, "substring")

    def _search_filter_clause(self, params: BeliefSearchParams) -> tuple[list[str], list[Any]]:
        conditions = ["belief_view.status = ?"]
        sql_params: list[Any] = ["active"]
        if params.types:
            placeholders = ",".join("?" for _ in params.types)
            conditions.append(f"belief_view.cognitive_type IN ({placeholders})")
            sql_params.extend(sorted(kind.value for kind in params.types))
        about_clause = self._search_about_clause(params)
        if about_clause is not None:
            clause, clause_params = about_clause
            conditions.append(clause)
            sql_params.extend(clause_params)
        return conditions, sql_params

    def _search_about_clause(
        self,
        params: BeliefSearchParams,
    ) -> tuple[str, list[Any]] | None:
        if params.counterpart is None:
            if params.include_global:
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
            return "0 = 1", []
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

    def _term_tokens(self, params: BeliefSearchParams) -> tuple[str, ...]:
        tokens: list[str] = []
        for text in self._search_texts(params):
            tokens.extend(tokenize_mixed_text(text))
        return tuple(_normalized_unique(tokens))

    def _trigram_probes(self, params: BeliefSearchParams) -> tuple[str, ...]:
        return tuple(_normalized_unique([*self._search_texts(params), *self._term_tokens(params)]))

    def _search_probes(self, params: BeliefSearchParams) -> tuple[str, ...]:
        return tuple(_normalized_unique([*self._search_texts(params), *self._term_tokens(params)]))

    def _search_texts(self, params: BeliefSearchParams) -> tuple[str, ...]:
        return tuple(
            text
            for text in (params.query, *params.keywords, *params.entities)
            if _normalize_text(text)
        )

    def _replace_belief_fts(
        self,
        conn: Any,
        belief: Belief,
        entity_ids: Sequence[str],
    ) -> None:
        belief_id = str(belief.id)
        self._delete_belief_fts(conn, belief_id)
        if str(belief.status) != "active":
            return
        content = str(belief.content)
        normalized_content = _normalize_text(content)
        conn.execute(
            """
            INSERT INTO belief_search_terms_fts (belief_id, search_terms, object)
            VALUES (?, ?, ?)
            """,
            (
                belief_id,
                self._belief_search_terms(belief, entity_ids, normalized_content),
                belief.object,
            ),
        )
        conn.execute(
            """
            INSERT INTO belief_search_trigram_fts
                (belief_id, content, object, normalized_content)
            VALUES (?, ?, ?, ?)
            """,
            (belief_id, content, belief.object, normalized_content),
        )

    def _delete_belief_fts(self, conn: Any, belief_id: str) -> None:
        conn.execute("DELETE FROM belief_search_terms_fts WHERE belief_id = ?", (belief_id,))
        conn.execute("DELETE FROM belief_search_trigram_fts WHERE belief_id = ?", (belief_id,))

    def _belief_search_terms(
        self,
        belief: Belief,
        entity_ids: Sequence[str],
        normalized_content: str,
    ) -> str:
        terms: list[str] = []
        terms.extend(tokenize_mixed_text(belief.content))
        terms.extend(tokenize_mixed_text(belief.object))
        terms.append(normalized_content)
        terms.extend(entity_ids)
        return " ".join(_normalized_unique(terms))

    def _merge_candidate(
        self,
        candidates: dict[str, dict[str, Any]],
        row: Any,
        reason: str,
        *,
        term_rank: float | None = None,
        trigram_rank: float | None = None,
    ) -> None:
        belief_id = str(row["id"])
        item = candidates.get(belief_id)
        if item is None:
            item = {
                "belief": self._from_row(row),
                "reasons": [],
                "term_rank": None,
                "trigram_rank": None,
            }
            candidates[belief_id] = item
        if reason not in item["reasons"]:
            item["reasons"].append(reason)
        if term_rank is not None:
            current = item["term_rank"]
            item["term_rank"] = term_rank if current is None else min(current, term_rank)
        if trigram_rank is not None:
            current = item["trigram_rank"]
            item["trigram_rank"] = trigram_rank if current is None else min(current, trigram_rank)

    def _candidate_sort_key(self, candidate: BeliefSearchCandidate) -> tuple[Any, ...]:
        reason_priority = min(
            (_CANDIDATE_REASON_ORDER.get(reason, 100) for reason in candidate.reasons),
            default=100,
        )
        return (
            reason_priority,
            candidate.term_rank if candidate.term_rank is not None else float("inf"),
            candidate.trigram_rank if candidate.trigram_rank is not None else float("inf"),
            str(candidate.belief.held_since),
            str(candidate.belief.id),
        )


def _belief_with_derived_profile(belief: Belief) -> Belief:
    if belief.value_profile.weights:
        return belief
    profile = derive_value_profile(
        belief.content,
        belief.structure,
        belief.cognitive_type,
        belief.about,
    )
    if not profile.weights:
        return belief
    return Belief.from_record(
        {
            **belief.to_record(),
            "value_profile": profile.to_record(),
        }
    )


def _normalized_unique(values: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized
