"""SQLite-backed belief entity store."""

from __future__ import annotations

import json
import re
import tempfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.models import (
    AtomicBelief,
    BeliefId,
    BeliefLifecycle,
    BeliefRecord,
    BeliefScope,
    CognitiveEventKind,
    MemoryKind,
    Reference,
    SummaryBelief,
    SummaryKind,
    belief_ref,
)
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.cognition.search_tokenizer import tokenize_mixed_text
from alpha_agent.state.store import StateStore

_ATOMIC_TABLE = "atomic"
_SUMMARY_TABLE = "summary"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS atomic_beliefs (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    memory_kind TEXT NOT NULL,
    derivation_stage TEXT NOT NULL,
    scope TEXT NOT NULL,
    authority TEXT NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'active',
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    validity TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_kind_scope_lifecycle
    ON atomic_beliefs(memory_kind, scope, lifecycle);
CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_lifecycle
    ON atomic_beliefs(lifecycle, held_since);
CREATE INDEX IF NOT EXISTS idx_atomic_beliefs_scope
    ON atomic_beliefs(scope, lifecycle);

CREATE TABLE IF NOT EXISTS summary_beliefs (
    id TEXT PRIMARY KEY,
    record TEXT NOT NULL DEFAULT '{}',
    object TEXT NOT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    summary_kind TEXT NOT NULL,
    derivation_stage TEXT NOT NULL,
    scope TEXT NOT NULL,
    authority TEXT NOT NULL,
    lifecycle TEXT NOT NULL DEFAULT 'active',
    structure TEXT NOT NULL DEFAULT '{}',
    sources TEXT NOT NULL DEFAULT '[]',
    validity TEXT NOT NULL DEFAULT '{}',
    relations TEXT NOT NULL DEFAULT '[]',
    update_policy TEXT NOT NULL DEFAULT '{}',
    source_belief_ids TEXT NOT NULL DEFAULT '[]',
    formed_in_situation TEXT,
    holder_role TEXT,
    action_orientation TEXT NOT NULL DEFAULT '[]',
    held_since TEXT NOT NULL,
    held_until TEXT,
    supersedes TEXT,
    superseded_by TEXT,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summary_beliefs_kind_scope_lifecycle
    ON summary_beliefs(summary_kind, scope, lifecycle);
CREATE INDEX IF NOT EXISTS idx_summary_beliefs_lifecycle
    ON summary_beliefs(lifecycle, held_since);
CREATE INDEX IF NOT EXISTS idx_summary_beliefs_scope
    ON summary_beliefs(scope, lifecycle);

CREATE TABLE IF NOT EXISTS belief_entity_index (
    belief_table TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY(belief_table, belief_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_entity_lookup
    ON belief_entity_index(entity_id, belief_table, belief_id);

CREATE TABLE IF NOT EXISTS belief_about_index (
    belief_table TEXT NOT NULL,
    belief_id TEXT NOT NULL,
    about_kind TEXT NOT NULL,
    about_id TEXT NOT NULL,
    PRIMARY KEY(belief_table, belief_id, about_kind, about_id)
);

CREATE INDEX IF NOT EXISTS idx_belief_about_lookup
    ON belief_about_index(about_kind, about_id, belief_table, belief_id);

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_terms_fts
USING fts5(
    belief_table UNINDEXED,
    belief_id UNINDEXED,
    search_terms,
    object,
    about,
    tokenize = "unicode61 remove_diacritics 1 tokenchars '_-#./:+'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS belief_search_trigram_fts
USING fts5(
    belief_table UNINDEXED,
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
    memory_kinds: frozenset[MemoryKind] | None = None
    summary_kinds: frozenset[SummaryKind] | None = None
    scopes: frozenset[BeliefScope] | None = None
    lifecycles: frozenset[BeliefLifecycle] = frozenset({BeliefLifecycle.ACTIVE})
    limit: int = 32


@dataclass(frozen=True)
class BeliefSearchParams:
    query: str
    keywords: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    counterpart: Reference | None = None
    include_global: bool = True
    memory_kinds: frozenset[MemoryKind] | None = None
    summary_kinds: frozenset[SummaryKind] | None = None
    scopes: frozenset[BeliefScope] | None = None
    lifecycles: frozenset[BeliefLifecycle] = frozenset({BeliefLifecycle.ACTIVE})
    limit: int = 64


@dataclass(frozen=True)
class BeliefSearchCandidate:
    belief: BeliefRecord
    reasons: tuple[str, ...]
    term_rank: float | None = None
    trigram_rank: float | None = None


@dataclass(frozen=True)
class BeliefProjectionView:
    beliefs: tuple[AtomicBelief, ...] = ()
    status: str = "entity_store"


@dataclass(frozen=True)
class _TableSpec:
    table_key: str
    table_name: str
    kind_column: str
    kinds: frozenset[MemoryKind] | frozenset[SummaryKind] | None


class BeliefProjection(Projection):
    """Direct store for current belief entity state."""

    name = "belief"
    handles = frozenset[CognitiveEventKind]()
    status = "entity_store"

    def __init__(self, store: StateStore | None = None):
        self.store = store or _temporary_store()
        self.store.initialize()
        self._ensure_schema()

    def upsert_atomic(self, belief: AtomicBelief) -> AtomicBelief:
        with self.store.transaction() as conn:
            self._upsert_belief_row(conn, _ATOMIC_TABLE, belief)
        return belief

    def upsert_summary(self, belief: SummaryBelief) -> SummaryBelief:
        with self.store.transaction() as conn:
            self._upsert_belief_row(conn, _SUMMARY_TABLE, belief)
        return belief

    def reaffirm(
        self,
        belief_id: BeliefId | str,
        *,
        source: Reference,
        observed_at: str,
    ) -> AtomicBelief | None:
        belief = self.get_by_id(belief_id)
        if not isinstance(belief, AtomicBelief):
            return None
        source_key = (source.kind, source.id)
        sources = list(belief.sources)
        if source_key not in {(item.kind, item.id) for item in sources}:
            sources.append(source)
        validity = belief.validity
        record = {
            **belief.to_record(),
            "sources": [item.to_record() for item in sources],
            "validity": {
                **validity.to_record(),
                "observed_at": observed_at or validity.observed_at,
            },
        }
        updated = AtomicBelief.from_record(record)
        return self.upsert_atomic(updated)

    def supersede_many(
        self,
        old_belief_ids: Sequence[BeliefId | str],
        new_belief: AtomicBelief,
        *,
        at: str,
    ) -> AtomicBelief:
        with self.store.transaction() as conn:
            self._upsert_belief_row(conn, _ATOMIC_TABLE, new_belief)
            for old_id in old_belief_ids:
                self._mark_lifecycle_row(
                    conn,
                    _ATOMIC_TABLE,
                    str(old_id),
                    BeliefLifecycle.SUPERSEDED,
                    at=at,
                    superseded_by=str(new_belief.id),
                )
        return new_belief

    def mark_lifecycle(
        self,
        belief_id: BeliefId | str,
        lifecycle: BeliefLifecycle,
        *,
        at: str,
    ) -> None:
        belief = self.get_by_id(belief_id)
        if belief is None:
            return
        table_key = _SUMMARY_TABLE if isinstance(belief, SummaryBelief) else _ATOMIC_TABLE
        with self.store.transaction() as conn:
            self._mark_lifecycle_row(conn, table_key, str(belief_id), lifecycle, at=at)

    def recall(
        self,
        params: BeliefRecallParams,
        **_: Any,
    ) -> list[BeliefRecord]:
        specs = self._table_specs(params.memory_kinds, params.summary_kinds)
        if not specs:
            return []
        rows_with_tables: list[tuple[str, Any]] = []
        with self.store.connect() as conn:
            for spec in specs:
                conditions, sql_params = self._filter_clause(
                    table_name=spec.table_name,
                    table_key=spec.table_key,
                    kind_column=spec.kind_column,
                    kinds=spec.kinds,
                    lifecycles=params.lifecycles,
                    scopes=params.scopes,
                    counterpart=params.counterpart,
                    include_global=params.include_global,
                )
                entity_ids = self._entity_ids(params.entities)
                if entity_ids:
                    placeholders = ",".join("?" for _ in entity_ids)
                    like_clause = " OR ".join(
                        f"{spec.table_name}.normalized_content LIKE ?" for _ in entity_ids
                    )
                    conditions.append(
                        f"""
                        (
                            {spec.table_name}.id IN (
                                SELECT belief_id
                                FROM belief_entity_index
                                WHERE belief_table = ?
                                  AND entity_id IN ({placeholders})
                            )
                            OR {like_clause}
                        )
                        """
                    )
                    sql_params.append(spec.table_key)
                    sql_params.extend(entity_ids)
                    sql_params.extend(f"%{entity_id}%" for entity_id in entity_ids)
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM {spec.table_name}
                    WHERE {' AND '.join(conditions)}
                    ORDER BY held_since ASC, id ASC
                    LIMIT ?
                    """,
                    [*sql_params, max(1, params.limit)],
                ).fetchall()
                rows_with_tables.extend((spec.table_key, row) for row in rows)
        beliefs = [self._from_row(row, table_key) for table_key, row in rows_with_tables]
        beliefs.sort(key=lambda item: self._recall_sort_key(item, params.counterpart))
        return beliefs[: max(1, params.limit)]

    def recall_candidates(self, params: BeliefSearchParams) -> list[BeliefSearchCandidate]:
        """Recall active beliefs with explicit search match signals."""

        specs = self._table_specs(params.memory_kinds, params.summary_kinds)
        if not specs:
            return []
        limit = max(1, int(params.limit))
        source_limit = max(limit * 4, 32)
        candidates: dict[tuple[str, str], dict[str, Any]] = {}
        with self.store.connect() as conn:
            for spec in specs:
                self._collect_entity_exact_candidates(conn, spec, params, candidates, source_limit)
                self._collect_object_candidates(conn, spec, params, candidates, source_limit)
                self._collect_term_fts_candidates(conn, spec, params, candidates, source_limit)
                self._collect_trigram_fts_candidates(conn, spec, params, candidates, source_limit)
                self._collect_substring_candidates(conn, spec, params, candidates, source_limit)

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

    def recall_about(self, ref: Reference) -> list[AtomicBelief]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT atomic_beliefs.*
                FROM atomic_beliefs
                JOIN belief_about_index
                  ON belief_about_index.belief_id = atomic_beliefs.id
                 AND belief_about_index.belief_table = ?
                WHERE atomic_beliefs.lifecycle = ?
                  AND belief_about_index.about_kind = ?
                  AND belief_about_index.about_id = ?
                ORDER BY atomic_beliefs.held_since ASC, atomic_beliefs.id ASC
                """,
                (_ATOMIC_TABLE, BeliefLifecycle.ACTIVE.value, ref.kind, ref.id),
            ).fetchall()
        return [self._from_atomic_row(row) for row in rows]

    def latest_summary(
        self,
        *,
        summary_kind: SummaryKind,
        about: Reference | None = None,
        scope: BeliefScope | None = None,
    ) -> SummaryBelief | None:
        conditions = [
            "summary_beliefs.summary_kind = ?",
            "summary_beliefs.lifecycle = ?",
        ]
        sql_params: list[Any] = [SummaryKind(summary_kind).value, BeliefLifecycle.ACTIVE.value]
        if scope is not None:
            conditions.append("summary_beliefs.scope = ?")
            sql_params.append(BeliefScope(scope).value)
        if about is not None:
            conditions.append(
                """
                EXISTS (
                    SELECT 1
                    FROM belief_about_index
                    WHERE belief_about_index.belief_table = ?
                      AND belief_about_index.belief_id = summary_beliefs.id
                      AND belief_about_index.about_kind = ?
                      AND belief_about_index.about_id = ?
                )
                """
            )
            sql_params.extend([_SUMMARY_TABLE, about.kind, about.id])
        with self.store.connect() as conn:
            row = conn.execute(
                f"""
                SELECT *
                FROM summary_beliefs
                WHERE {' AND '.join(conditions)}
                ORDER BY held_since DESC, id DESC
                LIMIT 1
                """,
                sql_params,
            ).fetchone()
        return self._from_summary_row(row) if row is not None else None

    def get_by_id(self, belief_id: BeliefId | str) -> BeliefRecord | None:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM atomic_beliefs WHERE id = ?",
                (str(belief_id),),
            ).fetchone()
            if row is not None:
                return self._from_atomic_row(row)
            row = conn.execute(
                "SELECT * FROM summary_beliefs WHERE id = ?",
                (str(belief_id),),
            ).fetchone()
        return self._from_summary_row(row) if row is not None else None

    def list_active(self) -> list[AtomicBelief]:
        with self.store.connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM atomic_beliefs
                WHERE lifecycle = ?
                ORDER BY held_since ASC, id ASC
                """,
                (BeliefLifecycle.ACTIVE.value,),
            ).fetchall()
        return [self._from_atomic_row(row) for row in rows]

    def reset(self) -> None:
        self._ensure_schema()
        with self.store.transaction() as conn:
            conn.execute("DELETE FROM belief_search_trigram_fts")
            conn.execute("DELETE FROM belief_search_terms_fts")
            conn.execute("DELETE FROM belief_about_index")
            conn.execute("DELETE FROM belief_entity_index")
            conn.execute("DELETE FROM summary_beliefs")
            conn.execute("DELETE FROM atomic_beliefs")

    def view(self) -> BeliefProjectionView:
        return BeliefProjectionView(beliefs=tuple(self.list_active()))

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def _table_specs(
        self,
        memory_kinds: frozenset[MemoryKind] | None,
        summary_kinds: frozenset[SummaryKind] | None,
    ) -> list[_TableSpec]:
        specs: list[_TableSpec] = []
        if summary_kinds is None:
            specs.append(_TableSpec(_ATOMIC_TABLE, "atomic_beliefs", "memory_kind", memory_kinds))
            return specs
        if memory_kinds is not None:
            specs.append(_TableSpec(_ATOMIC_TABLE, "atomic_beliefs", "memory_kind", memory_kinds))
        specs.append(_TableSpec(_SUMMARY_TABLE, "summary_beliefs", "summary_kind", summary_kinds))
        return specs

    def _filter_clause(
        self,
        *,
        table_name: str,
        table_key: str,
        kind_column: str,
        kinds: frozenset[MemoryKind] | frozenset[SummaryKind] | None,
        lifecycles: frozenset[BeliefLifecycle],
        scopes: frozenset[BeliefScope] | None,
        counterpart: Reference | None,
        include_global: bool,
    ) -> tuple[list[str], list[Any]]:
        lifecycle_values = sorted(BeliefLifecycle(item).value for item in lifecycles)
        conditions = [f"{table_name}.lifecycle IN ({','.join('?' for _ in lifecycle_values)})"]
        sql_params: list[Any] = list(lifecycle_values)

        if kinds:
            kind_values = sorted(item.value for item in kinds)
            placeholders = ",".join("?" for _ in kind_values)
            conditions.append(f"{table_name}.{kind_column} IN ({placeholders})")
            sql_params.extend(kind_values)
        if scopes:
            scope_values = sorted(BeliefScope(item).value for item in scopes)
            conditions.append(f"{table_name}.scope IN ({','.join('?' for _ in scope_values)})")
            sql_params.extend(scope_values)

        if scopes is None:
            about_clause = self._about_clause(
                table_name=table_name,
                table_key=table_key,
                counterpart=counterpart,
                include_global=include_global,
            )
            if about_clause is not None:
                clause, clause_params = about_clause
                conditions.append(clause)
                sql_params.extend(clause_params)
        elif counterpart is not None and BeliefScope.COUNTERPART in scopes:
            conditions.append(
                f"""
                (
                    {table_name}.scope != ?
                    OR EXISTS (
                        SELECT 1
                        FROM belief_about_index
                        WHERE belief_about_index.belief_table = ?
                          AND belief_about_index.belief_id = {table_name}.id
                          AND belief_about_index.about_kind = ?
                          AND belief_about_index.about_id = ?
                    )
                )
                """
            )
            sql_params.extend(
                [
                    BeliefScope.COUNTERPART.value,
                    table_key,
                    counterpart.kind,
                    counterpart.id,
                ]
            )
        return conditions, sql_params

    def _about_clause(
        self,
        *,
        table_name: str,
        table_key: str,
        counterpart: Reference | None,
        include_global: bool,
    ) -> tuple[str, list[Any]] | None:
        if counterpart is None:
            if include_global:
                return f"{table_name}.scope = ?", [BeliefScope.GLOBAL.value]
            return "0 = 1", []
        counterpart_clause = f"""
            (
                {table_name}.scope = ?
                AND EXISTS (
                    SELECT 1
                    FROM belief_about_index
                    WHERE belief_about_index.belief_table = ?
                      AND belief_about_index.belief_id = {table_name}.id
                      AND belief_about_index.about_kind = ?
                      AND belief_about_index.about_id = ?
                )
            )
        """
        counterpart_params = [
            BeliefScope.COUNTERPART.value,
            table_key,
            counterpart.kind,
            counterpart.id,
        ]
        if not include_global:
            return counterpart_clause, counterpart_params
        return (
            f"({counterpart_clause} OR {table_name}.scope = ?)",
            [*counterpart_params, BeliefScope.GLOBAL.value],
        )

    def _entity_ids(self, entities: Sequence[Reference]) -> list[str]:
        return sorted({_normalize_text(ref.id) for ref in entities if ref.kind == "entity"})

    def _recall_sort_key(
        self,
        belief: BeliefRecord,
        counterpart: Reference | None,
    ) -> tuple[int, str, str]:
        return (
            0 if self._matches_counterpart_scope(belief, counterpart) else 1,
            str(belief.held_since),
            str(belief.id),
        )

    def _matches_counterpart_scope(
        self,
        belief: BeliefRecord,
        counterpart: Reference | None,
    ) -> bool:
        if counterpart is None or belief.scope != BeliefScope.COUNTERPART:
            return False
        return any(
            ref.kind == counterpart.kind and ref.id == counterpart.id
            for ref in belief.about
        )

    def _upsert_belief_row(
        self,
        conn: Any,
        table_key: str,
        belief: BeliefRecord,
    ) -> None:
        if isinstance(belief, SummaryBelief):
            self._upsert_summary_row(conn, belief)
        else:
            self._upsert_atomic_row(conn, belief)
        self._replace_indexes(conn, table_key, belief)

    def _upsert_atomic_row(self, conn: Any, belief: AtomicBelief) -> None:
        conn.execute(
            """
            INSERT INTO atomic_beliefs
                (id, record, object, content, normalized_content, memory_kind,
                 derivation_stage, scope, authority, lifecycle, structure, sources,
                 validity, relations, update_policy, formed_in_situation, holder_role,
                 action_orientation, held_since, held_until, supersedes, superseded_by,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                record = excluded.record,
                object = excluded.object,
                content = excluded.content,
                normalized_content = excluded.normalized_content,
                memory_kind = excluded.memory_kind,
                derivation_stage = excluded.derivation_stage,
                scope = excluded.scope,
                authority = excluded.authority,
                lifecycle = excluded.lifecycle,
                structure = excluded.structure,
                sources = excluded.sources,
                validity = excluded.validity,
                relations = excluded.relations,
                update_policy = excluded.update_policy,
                formed_in_situation = excluded.formed_in_situation,
                holder_role = excluded.holder_role,
                action_orientation = excluded.action_orientation,
                held_since = excluded.held_since,
                held_until = excluded.held_until,
                supersedes = excluded.supersedes,
                superseded_by = excluded.superseded_by,
                updated_at = excluded.updated_at
            """,
            self._row_values(belief, kind_value=belief.memory_kind.value, extra_values=()),
        )

    def _upsert_summary_row(self, conn: Any, belief: SummaryBelief) -> None:
        values = self._row_values(
            belief,
            kind_value=belief.summary_kind.value,
            extra_values=(_dumps([str(item) for item in belief.source_belief_ids]),),
        )
        conn.execute(
            """
            INSERT INTO summary_beliefs
                (id, record, object, content, normalized_content, summary_kind,
                 derivation_stage, scope, authority, lifecycle, structure, sources,
                 validity, relations, update_policy, source_belief_ids,
                 formed_in_situation, holder_role, action_orientation, held_since,
                 held_until, supersedes, superseded_by, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                record = excluded.record,
                object = excluded.object,
                content = excluded.content,
                normalized_content = excluded.normalized_content,
                summary_kind = excluded.summary_kind,
                derivation_stage = excluded.derivation_stage,
                scope = excluded.scope,
                authority = excluded.authority,
                lifecycle = excluded.lifecycle,
                structure = excluded.structure,
                sources = excluded.sources,
                validity = excluded.validity,
                relations = excluded.relations,
                update_policy = excluded.update_policy,
                source_belief_ids = excluded.source_belief_ids,
                formed_in_situation = excluded.formed_in_situation,
                holder_role = excluded.holder_role,
                action_orientation = excluded.action_orientation,
                held_since = excluded.held_since,
                held_until = excluded.held_until,
                supersedes = excluded.supersedes,
                superseded_by = excluded.superseded_by,
                updated_at = excluded.updated_at
            """,
            values,
        )

    def _row_values(
        self,
        belief: BeliefRecord,
        *,
        kind_value: str,
        extra_values: tuple[Any, ...],
    ) -> tuple[Any, ...]:
        held_since = str(belief.held_since)
        updated_at = held_since or str(belief.validity.observed_at or "")
        common = (
            str(belief.id),
            _dumps(belief.to_record()),
            belief.object,
            str(belief.content),
            _normalize_text(belief.content),
            kind_value,
            belief.derivation_stage.value,
            belief.scope.value,
            belief.authority.value,
            belief.lifecycle.value,
            _dumps(belief.structure),
            _dumps([source.to_record() for source in belief.sources]),
            _dumps(belief.validity.to_record()),
            _dumps([relation.to_record() for relation in belief.relations]),
            _dumps(belief.update_policy),
            *extra_values,
            belief.formed_in.id,
            str(belief.holder_role),
            _dumps([str(action) for action in belief.action_orientation]),
            held_since,
            str(belief.held_until) if belief.held_until is not None else None,
            belief.supersedes.id if belief.supersedes is not None else None,
            belief.superseded_by.id if belief.superseded_by is not None else None,
            updated_at,
        )
        return common

    def _replace_indexes(self, conn: Any, table_key: str, belief: BeliefRecord) -> None:
        belief_id = str(belief.id)
        conn.execute(
            "DELETE FROM belief_about_index WHERE belief_table = ? AND belief_id = ?",
            (table_key, belief_id),
        )
        conn.execute(
            "DELETE FROM belief_entity_index WHERE belief_table = ? AND belief_id = ?",
            (table_key, belief_id),
        )
        conn.executemany(
            """
            INSERT OR IGNORE INTO belief_about_index
                (belief_table, belief_id, about_kind, about_id)
            VALUES (?, ?, ?, ?)
            """,
            [(table_key, belief_id, ref.kind, ref.id) for ref in belief.about],
        )
        entity_ids = self._belief_entity_ids(belief)
        conn.executemany(
            """
            INSERT OR IGNORE INTO belief_entity_index (belief_table, belief_id, entity_id)
            VALUES (?, ?, ?)
            """,
            [(table_key, belief_id, entity_id) for entity_id in entity_ids],
        )
        self._replace_belief_fts(conn, table_key, belief, entity_ids)

    def _belief_entity_ids(self, belief: BeliefRecord) -> list[str]:
        entity_ids = {_normalize_text(belief.object)}
        entity_ids.update(_normalize_text(ref.id) for ref in belief.about)
        entity_ids.discard("")
        return sorted(entity_ids)

    def _mark_lifecycle_row(
        self,
        conn: Any,
        table_key: str,
        belief_id: str,
        lifecycle: BeliefLifecycle,
        *,
        at: str,
        superseded_by: str | None = None,
    ) -> None:
        table_name = "summary_beliefs" if table_key == _SUMMARY_TABLE else "atomic_beliefs"
        record_row = conn.execute(
            f"SELECT record FROM {table_name} WHERE id = ?",
            (belief_id,),
        ).fetchone()
        if record_row is not None:
            record = _loads(record_row["record"], {})
            if isinstance(record, dict):
                record["lifecycle"] = BeliefLifecycle(lifecycle).value
                record["held_until"] = at
                if superseded_by is not None:
                    record["superseded_by"] = belief_ref(BeliefId(superseded_by)).to_record()
                conn.execute(
                    f"""
                    UPDATE {table_name}
                    SET record = ?,
                        lifecycle = ?,
                        held_until = ?,
                        superseded_by = COALESCE(?, superseded_by),
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        _dumps(record),
                        BeliefLifecycle(lifecycle).value,
                        at,
                        superseded_by,
                        at,
                        belief_id,
                    ),
                )
        if lifecycle != BeliefLifecycle.ACTIVE:
            self._delete_belief_fts(conn, table_key, belief_id)

    def _from_row(self, row: Any, table_key: str) -> BeliefRecord:
        record = _loads(row["record"], {})
        if not isinstance(record, dict) or not record:
            raise ValueError(f"{table_key} belief row {row['id']!r} is missing its record")
        record.update(
            {
                "id": row["id"],
                "object": row["object"],
                "content": row["content"],
                "derivation_stage": row["derivation_stage"],
                "scope": row["scope"],
                "authority": row["authority"],
                "lifecycle": row["lifecycle"],
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
        if table_key == _SUMMARY_TABLE:
            record["summary_kind"] = row["summary_kind"]
            return SummaryBelief.from_record(record)
        record["memory_kind"] = row["memory_kind"]
        return AtomicBelief.from_record(record)

    def _from_atomic_row(self, row: Any) -> AtomicBelief:
        belief = self._from_row(row, _ATOMIC_TABLE)
        if not isinstance(belief, AtomicBelief):
            raise TypeError("atomic belief row materialized as summary belief")
        return belief

    def _from_summary_row(self, row: Any) -> SummaryBelief:
        belief = self._from_row(row, _SUMMARY_TABLE)
        if not isinstance(belief, SummaryBelief):
            raise TypeError("summary belief row materialized as atomic belief")
        return belief

    def _collect_entity_exact_candidates(
        self,
        conn: Any,
        spec: _TableSpec,
        params: BeliefSearchParams,
        candidates: dict[tuple[str, str], dict[str, Any]],
        limit: int,
    ) -> None:
        entity_ids = _normalized_unique(self._search_probes(params))
        if not entity_ids:
            return
        conditions, sql_params = self._search_filter_clause(spec, params)
        placeholders = ",".join("?" for _ in entity_ids)
        query = f"""
            SELECT DISTINCT {spec.table_name}.*
            FROM {spec.table_name}
            JOIN belief_entity_index
              ON belief_entity_index.belief_id = {spec.table_name}.id
             AND belief_entity_index.belief_table = ?
            WHERE {' AND '.join(conditions)}
              AND belief_entity_index.entity_id IN ({placeholders})
            ORDER BY {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
            LIMIT ?
        """
        rows = conn.execute(
            query,
            [spec.table_key, *sql_params, *entity_ids, limit],
        ).fetchall()
        for row in rows:
            self._merge_candidate(candidates, spec.table_key, row, "entity_exact")

    def _collect_object_candidates(
        self,
        conn: Any,
        spec: _TableSpec,
        params: BeliefSearchParams,
        candidates: dict[tuple[str, str], dict[str, Any]],
        limit: int,
    ) -> None:
        probes = self._search_probes(params)
        exact_probes = [probe for probe in probes if probe]
        if exact_probes:
            conditions, sql_params = self._search_filter_clause(spec, params)
            placeholders = ",".join("?" for _ in exact_probes)
            rows = conn.execute(
                f"""
                SELECT {spec.table_name}.*
                FROM {spec.table_name}
                WHERE {' AND '.join(conditions)}
                  AND lower({spec.table_name}.object) IN ({placeholders})
                ORDER BY {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
                LIMIT ?
                """,
                [*sql_params, *exact_probes, limit],
            ).fetchall()
            for row in rows:
                self._merge_candidate(candidates, spec.table_key, row, "object_exact")

        partial_probes = [probe for probe in probes if len(probe) >= 3]
        for probe in partial_probes:
            conditions, sql_params = self._search_filter_clause(spec, params)
            like_probe = f"%{_escape_like(probe)}%"
            rows = conn.execute(
                f"""
                SELECT {spec.table_name}.*
                FROM {spec.table_name}
                WHERE {' AND '.join(conditions)}
                  AND lower({spec.table_name}.object) LIKE ? ESCAPE '!'
                ORDER BY {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
                LIMIT ?
                """,
                [*sql_params, like_probe, limit],
            ).fetchall()
            for row in rows:
                self._merge_candidate(candidates, spec.table_key, row, "object_partial")

    def _collect_term_fts_candidates(
        self,
        conn: Any,
        spec: _TableSpec,
        params: BeliefSearchParams,
        candidates: dict[tuple[str, str], dict[str, Any]],
        limit: int,
    ) -> None:
        query_text = build_term_fts_query(self._term_tokens(params))
        if not query_text:
            return
        conditions, sql_params = self._search_filter_clause(spec, params)
        rows = conn.execute(
            f"""
            SELECT {spec.table_name}.*, bm25(belief_search_terms_fts) AS term_rank
            FROM belief_search_terms_fts
            JOIN {spec.table_name}
              ON {spec.table_name}.id = belief_search_terms_fts.belief_id
             AND belief_search_terms_fts.belief_table = ?
            WHERE {' AND '.join(conditions)}
              AND belief_search_terms_fts MATCH ?
            ORDER BY term_rank ASC, {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
            LIMIT ?
            """,
            [spec.table_key, *sql_params, query_text, limit],
        ).fetchall()
        for row in rows:
            self._merge_candidate(
                candidates,
                spec.table_key,
                row,
                "term_fts",
                term_rank=float(row["term_rank"]),
            )

    def _collect_trigram_fts_candidates(
        self,
        conn: Any,
        spec: _TableSpec,
        params: BeliefSearchParams,
        candidates: dict[tuple[str, str], dict[str, Any]],
        limit: int,
    ) -> None:
        query_text = build_trigram_fts_query(self._trigram_probes(params))
        if not query_text:
            return
        conditions, sql_params = self._search_filter_clause(spec, params)
        rows = conn.execute(
            f"""
            SELECT {spec.table_name}.*, bm25(belief_search_trigram_fts) AS trigram_rank
            FROM belief_search_trigram_fts
            JOIN {spec.table_name}
              ON {spec.table_name}.id = belief_search_trigram_fts.belief_id
             AND belief_search_trigram_fts.belief_table = ?
            WHERE {' AND '.join(conditions)}
              AND belief_search_trigram_fts MATCH ?
            ORDER BY trigram_rank ASC, {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
            LIMIT ?
            """,
            [spec.table_key, *sql_params, query_text, limit],
        ).fetchall()
        for row in rows:
            self._merge_candidate(
                candidates,
                spec.table_key,
                row,
                "trigram_fts",
                trigram_rank=float(row["trigram_rank"]),
            )

    def _collect_substring_candidates(
        self,
        conn: Any,
        spec: _TableSpec,
        params: BeliefSearchParams,
        candidates: dict[tuple[str, str], dict[str, Any]],
        limit: int,
    ) -> None:
        probes = [probe for probe in self._search_probes(params) if len(probe) >= 3]
        for probe in probes:
            conditions, sql_params = self._search_filter_clause(spec, params)
            like_probe = f"%{_escape_like(probe)}%"
            rows = conn.execute(
                f"""
                SELECT {spec.table_name}.*
                FROM {spec.table_name}
                WHERE {' AND '.join(conditions)}
                  AND (
                    {spec.table_name}.normalized_content LIKE ? ESCAPE '!'
                    OR lower({spec.table_name}.object) LIKE ? ESCAPE '!'
                  )
                ORDER BY {spec.table_name}.held_since ASC, {spec.table_name}.id ASC
                LIMIT ?
                """,
                [*sql_params, like_probe, like_probe, limit],
            ).fetchall()
            for row in rows:
                self._merge_candidate(candidates, spec.table_key, row, "substring")

    def _search_filter_clause(
        self,
        spec: _TableSpec,
        params: BeliefSearchParams,
    ) -> tuple[list[str], list[Any]]:
        return self._filter_clause(
            table_name=spec.table_name,
            table_key=spec.table_key,
            kind_column=spec.kind_column,
            kinds=spec.kinds,
            lifecycles=params.lifecycles,
            scopes=params.scopes,
            counterpart=params.counterpart,
            include_global=params.include_global,
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
        table_key: str,
        belief: BeliefRecord,
        entity_ids: Sequence[str],
    ) -> None:
        belief_id = str(belief.id)
        self._delete_belief_fts(conn, table_key, belief_id)
        if belief.lifecycle != BeliefLifecycle.ACTIVE:
            return
        content = str(belief.content)
        normalized_content = _normalize_text(content)
        conn.execute(
            """
            INSERT INTO belief_search_terms_fts
                (belief_table, belief_id, search_terms, object, about)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                table_key,
                belief_id,
                self._belief_search_terms(belief, entity_ids, normalized_content),
                belief.object,
                self._about_search_terms(belief.about),
            ),
        )
        conn.execute(
            """
            INSERT INTO belief_search_trigram_fts
                (belief_table, belief_id, content, object, normalized_content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (table_key, belief_id, content, belief.object, normalized_content),
        )

    def _delete_belief_fts(self, conn: Any, table_key: str, belief_id: str) -> None:
        conn.execute(
            "DELETE FROM belief_search_terms_fts WHERE belief_table = ? AND belief_id = ?",
            (table_key, belief_id),
        )
        conn.execute(
            "DELETE FROM belief_search_trigram_fts WHERE belief_table = ? AND belief_id = ?",
            (table_key, belief_id),
        )

    def _belief_search_terms(
        self,
        belief: BeliefRecord,
        entity_ids: Sequence[str],
        normalized_content: str,
    ) -> str:
        terms: list[str] = []
        terms.extend(tokenize_mixed_text(belief.content))
        terms.extend(tokenize_mixed_text(belief.object))
        terms.extend(tokenize_mixed_text(self._about_search_terms(belief.about)))
        terms.append(normalized_content)
        terms.extend(entity_ids)
        return " ".join(_normalized_unique(terms))

    def _about_search_terms(self, about: Sequence[Reference]) -> str:
        return " ".join(f"{ref.kind}:{ref.id} {ref.id}" for ref in about)

    def _merge_candidate(
        self,
        candidates: dict[tuple[str, str], dict[str, Any]],
        table_key: str,
        row: Any,
        reason: str,
        *,
        term_rank: float | None = None,
        trigram_rank: float | None = None,
    ) -> None:
        belief_id = str(row["id"])
        key = (table_key, belief_id)
        item = candidates.get(key)
        if item is None:
            item = {
                "belief": self._from_row(row, table_key),
                "reasons": [],
                "term_rank": None,
                "trigram_rank": None,
            }
            candidates[key] = item
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


def _normalized_unique(values: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _normalize_text(value)
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized
