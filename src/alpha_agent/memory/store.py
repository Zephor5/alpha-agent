"""SQLite persistence for Alpha Agent memory."""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

from alpha_agent.graph.models import EntityNode, RelationEdge, RelationEdgeAudit
from alpha_agent.memory.models import (
    ConversationMessage,
    ConversationRole,
    EpisodicMemory,
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    ProceduralMemory,
    RuntimeTrace,
    SemanticMemory,
    SemanticMemoryDrillDown,
    SessionContextState,
)
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now, utc_now_iso

T = TypeVar("T")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _loads_list(value: str | None) -> list[str]:
    if not value:
        return []
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return []
    return [str(item) for item in loaded]


def _loads_dict_list(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


def _metadata_string_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _scope_from_row(row: sqlite3.Row) -> MemoryScope:
    record = _loads_dict(row["scope_metadata"])
    record["kind"] = row["scope_kind"]
    record["scope_key"] = row["scope_key"]
    return MemoryScope.from_record(record)


def _scope_params(scope: MemoryScope) -> tuple[str, str, str]:
    return scope.kind, scope.scope_key, _dumps(scope.to_record())


def _scope_filter(
    scopes: list[MemoryScope] | None,
    *,
    column: str = "scope_key",
) -> tuple[str, list[str]]:
    if not scopes:
        return "", []
    keys = [scope.scope_key for scope in scopes]
    placeholders = ",".join("?" for _ in keys)
    return f" AND {column} IN ({placeholders})", keys


def _normalize_content(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", value.casefold())).strip()


def _normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.casefold().split()).strip()
    return normalized or None


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _dedupe_memories(memories: list[SemanticMemory]) -> list[SemanticMemory]:
    result: list[SemanticMemory] = []
    seen: set[str] = set()
    for memory in memories:
        if memory.id in seen:
            continue
        seen.add(memory.id)
        result.append(memory)
    return result


def _dedupe_strings(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


class MemoryStore:
    """Low-level SQLite operations for transcript, context, traces, and memory layers."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()
        self.fts_available = False

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        """Create database tables and optional FTS5 indexes."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            self.fts_available = self._try_initialize_fts(conn)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run operations in a SQLite transaction."""

        with self.connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @contextmanager
    def immediate_transaction(self) -> Iterator[sqlite3.Connection]:
        """Run operations after acquiring the SQLite write lock."""

        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def _try_initialize_fts(self, conn: sqlite3.Connection) -> bool:
        try:
            conn.executescript(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts
                USING fts5(memory_id UNINDEXED, content, summary);
                CREATE VIRTUAL TABLE IF NOT EXISTS semantic_fts
                USING fts5(memory_id UNINDEXED, subject, predicate, object, content);
                CREATE VIRTUAL TABLE IF NOT EXISTS procedural_fts
                USING fts5(memory_id UNINDEXED, name, description, trigger, procedure_markdown);
                """
            )
        except sqlite3.OperationalError:
            return False
        return True

    def _has_fts_table(self, conn: sqlite3.Connection, name: str) -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        return row is not None

    def has_fts_index(self, name: str) -> bool:
        """Return whether an FTS index table is available."""

        with self.connect() as conn:
            return self._has_fts_table(conn, name)

    def _with_conn(
        self,
        conn: sqlite3.Connection | None,
        fn: Callable[[sqlite3.Connection], T],
    ) -> T:
        if conn is not None:
            return fn(conn)
        with self.connect() as local:
            self.fts_available = self._has_fts_table(local, "episodic_fts")
            return fn(local)

    def append_conversation_message(
        self,
        *,
        session_id: str,
        role: ConversationRole,
        raw_content: str,
        model_content: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_result_id: str | None = None,
        provider_metadata: dict[str, Any] | None = None,
        source_metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ConversationMessage:
        """Append a transcript message with the next monotonic session ordinal."""

        def op(db: sqlite3.Connection) -> ConversationMessage:
            message = ConversationMessage(
                id=new_id("msg"),
                session_id=session_id,
                ordinal=self._next_conversation_ordinal(db, session_id),
                role=role,
                raw_content=raw_content,
                model_content=model_content,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls or [],
                tool_result_id=tool_result_id,
                provider_metadata=provider_metadata or {},
                source_metadata=source_metadata or {},
                created_at=created_at or utc_now_iso(),
                metadata=metadata or {},
            )
            return self._insert_conversation_message(db, message)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def insert_conversation_message(
        self,
        message: ConversationMessage,
        conn: sqlite3.Connection | None = None,
    ) -> ConversationMessage:
        """Insert a transcript message only if it is the next ordinal for its session."""

        def op(db: sqlite3.Connection) -> ConversationMessage:
            expected_ordinal = self._next_conversation_ordinal(db, message.session_id)
            if message.ordinal != expected_ordinal:
                raise ValueError(
                    "conversation message ordinal for session "
                    f"{message.session_id!r} must be {expected_ordinal}, got {message.ordinal}"
                )
            return self._insert_conversation_message(db, message)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def list_conversation_messages(
        self,
        session_id: str,
        *,
        after_ordinal: int | None = None,
        before_ordinal: int | None = None,
        limit: int | None = None,
    ) -> list[ConversationMessage]:
        """List transcript messages in ascending ordinal order."""

        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if after_ordinal is not None:
            conditions.append("ordinal > ?")
            params.append(after_ordinal)
        if before_ordinal is not None:
            conditions.append("ordinal < ?")
            params.append(before_ordinal)
        query = f"""
            SELECT * FROM conversation_messages
            WHERE {' AND '.join(conditions)}
            ORDER BY ordinal ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._conversation_message_from_row(row) for row in rows]

    def list_conversation_messages_by_ids(
        self,
        message_ids: list[str],
    ) -> list[ConversationMessage]:
        """Return transcript messages for the given ids, preserving requested order."""

        if not message_ids:
            return []
        placeholders = ",".join("?" for _ in message_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM conversation_messages
                WHERE id IN ({placeholders})
                """,
                message_ids,
            ).fetchall()
        by_id = {str(row["id"]): self._conversation_message_from_row(row) for row in rows}
        return [by_id[message_id] for message_id in message_ids if message_id in by_id]

    def latest_conversation_ordinal(self, session_id: str) -> int:
        """Return the latest message ordinal for a session, or zero if it has none."""

        with self.connect() as conn:
            return self._latest_conversation_ordinal(conn, session_id)

    def upsert_session_context_state(
        self,
        state: SessionContextState,
        conn: sqlite3.Connection | None = None,
    ) -> SessionContextState:
        """Create or replace the active compressed context state for a session."""

        if state.compressed_until_ordinal < 0:
            raise ValueError("compressed_until_ordinal must be greater than or equal to 0")

        def op(db: sqlite3.Connection) -> SessionContextState:
            existing = db.execute(
                """
                SELECT compressed_until_ordinal
                FROM session_context_states
                WHERE session_id = ?
                """,
                (state.session_id,),
            ).fetchone()
            if (
                existing is not None
                and int(existing["compressed_until_ordinal"]) > state.compressed_until_ordinal
            ):
                raise ValueError(
                    "compressed_until_ordinal cannot move backward for session "
                    f"{state.session_id!r}"
                )
            db.execute(
                """
                INSERT INTO session_context_states
                    (session_id, compressed_until_ordinal, summary,
                     summary_source_message_ids, compression_version,
                     created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    compressed_until_ordinal = excluded.compressed_until_ordinal,
                    summary = excluded.summary,
                    summary_source_message_ids = excluded.summary_source_message_ids,
                    compression_version = excluded.compression_version,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    state.session_id,
                    state.compressed_until_ordinal,
                    state.summary,
                    _dumps(state.summary_source_message_ids),
                    state.compression_version,
                    state.created_at,
                    state.updated_at,
                    _dumps(state.metadata),
                ),
            )
            row = db.execute(
                "SELECT * FROM session_context_states WHERE session_id = ?",
                (state.session_id,),
            ).fetchone()
            return self._session_context_state_from_row(row)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def get_session_context_state(self, session_id: str) -> SessionContextState | None:
        """Return the active compressed context state for a session."""

        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM session_context_states WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return self._session_context_state_from_row(row) if row else None

    def append_runtime_trace(
        self,
        *,
        session_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        timestamp: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> RuntimeTrace:
        """Append a narrow runtime diagnostic trace."""

        trace = RuntimeTrace(
            id=new_id("trace"),
            session_id=session_id,
            event_type=event_type,
            content=content,
            timestamp=timestamp or utc_now_iso(),
            metadata=metadata or {},
        )
        return self.insert_runtime_trace(trace, conn)

    def insert_runtime_trace(
        self,
        trace: RuntimeTrace,
        conn: sqlite3.Connection | None = None,
    ) -> RuntimeTrace:
        """Insert a runtime diagnostic trace."""

        def op(db: sqlite3.Connection) -> RuntimeTrace:
            db.execute(
                """
                INSERT INTO runtime_traces
                    (id, event_type, session_id, content, metadata, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    trace.id,
                    trace.event_type,
                    trace.session_id,
                    trace.content,
                    _dumps(trace.metadata),
                    trace.timestamp,
                ),
            )
            return trace

        return self._with_conn(conn, op)

    def list_runtime_traces(
        self,
        session_id: str | None = None,
        *,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[RuntimeTrace]:
        """List recent runtime diagnostic traces."""

        conditions: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)
        query = "SELECT * FROM runtime_traces"
        if conditions:
            query += f" WHERE {' AND '.join(conditions)}"
        query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._runtime_trace_from_row(row) for row in rows]

    def insert_episodic_memory(
        self,
        memory: EpisodicMemory,
        conn: sqlite3.Connection | None = None,
    ) -> EpisodicMemory:
        """Insert an episodic memory and optional FTS row."""

        def op(db: sqlite3.Connection) -> EpisodicMemory:
            db.execute(
                """
                INSERT INTO episodic_memories
                    (id, content, summary, source_event_ids, people, places, topics,
                     salience, confidence, created_at, last_accessed_at, access_count,
                     scope_kind, scope_key, scope_metadata, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.id,
                    memory.content,
                    memory.summary,
                    _dumps(memory.source_event_ids),
                    _dumps(memory.people),
                    _dumps(memory.places),
                    _dumps(memory.topics),
                    memory.salience,
                    memory.confidence,
                    memory.created_at,
                    memory.last_accessed_at,
                    memory.access_count,
                    *_scope_params(memory.scope),
                    _dumps(memory.metadata),
                ),
            )
            if self._has_fts_table(db, "episodic_fts"):
                db.execute(
                    "INSERT INTO episodic_fts (memory_id, content, summary) VALUES (?, ?, ?)",
                    (memory.id, memory.content, memory.summary),
                )
            return memory

        return self._with_conn(conn, op)

    def list_episodic_memories(
        self,
        limit: int = 50,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[EpisodicMemory]:
        """List recent episodic memories."""

        scope_sql, scope_params = _scope_filter(scopes)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM episodic_memories
                WHERE 1 = 1{scope_sql}
                ORDER BY created_at DESC LIMIT ?
                """,
                (*scope_params, limit),
            ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def search_episodic(
        self,
        query: str,
        limit: int = 20,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[EpisodicMemory]:
        """Search episodic memories using FTS5 when available, otherwise LIKE."""

        scope_sql, scope_params = _scope_filter(scopes)
        with self.connect() as conn:
            if self._has_fts_table(conn, "episodic_fts") and query.strip():
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM episodic_fts f
                    JOIN episodic_memories m ON m.id = f.memory_id
                    WHERE episodic_fts MATCH ?
                    {scope_sql}
                    ORDER BY bm25(episodic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), *scope_params, limit),
                ).fetchall()
            else:
                scope_sql, scope_params = _scope_filter(scopes)
                like = f"%{query}%"
                rows = conn.execute(
                    f"""
                    SELECT * FROM episodic_memories
                    WHERE (content LIKE ? OR summary LIKE ?)
                    {scope_sql}
                    ORDER BY salience DESC, created_at DESC LIMIT ?
                    """,
                    (like, like, *scope_params, limit),
                ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def upsert_semantic_memory(
        self,
        memory: SemanticMemory,
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory:
        """Insert or update one atomic semantic memory by id."""

        def op(db: sqlite3.Connection) -> SemanticMemory:
            db.execute(
                """
                INSERT INTO semantic_memories
                    (id, content, normalized_content, memory_type, subject, predicate,
                     object, entities, confidence, salience, stability, source_memory_ids,
                     status, valid_from, valid_until, supersedes_id, superseded_by_id,
                     deleted_at, scope_kind, scope_key, scope_metadata, created_at,
                     updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    normalized_content = excluded.normalized_content,
                    memory_type = excluded.memory_type,
                    subject = excluded.subject,
                    predicate = excluded.predicate,
                    object = excluded.object,
                    entities = excluded.entities,
                    confidence = excluded.confidence,
                    salience = excluded.salience,
                    stability = excluded.stability,
                    source_memory_ids = excluded.source_memory_ids,
                    status = excluded.status,
                    valid_from = excluded.valid_from,
                    valid_until = excluded.valid_until,
                    supersedes_id = excluded.supersedes_id,
                    superseded_by_id = excluded.superseded_by_id,
                    deleted_at = excluded.deleted_at,
                    scope_kind = excluded.scope_kind,
                    scope_key = excluded.scope_key,
                    scope_metadata = excluded.scope_metadata,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    memory.content,
                    _normalize_content(memory.content),
                    memory.memory_type,
                    _normalize_optional(memory.subject),
                    _normalize_optional(memory.predicate),
                    _normalize_optional(memory.object),
                    _dumps(memory.entities),
                    memory.confidence,
                    memory.salience,
                    memory.stability,
                    _dumps(memory.source_memory_ids),
                    memory.status,
                    memory.valid_from,
                    memory.valid_until,
                    memory.supersedes_id,
                    memory.superseded_by_id,
                    memory.deleted_at,
                    *_scope_params(memory.scope),
                    memory.created_at,
                    memory.updated_at,
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                "SELECT * FROM semantic_memories WHERE id = ?",
                (memory.id,),
            ).fetchone()
            saved = self._semantic_from_row(row)
            if self._has_fts_table(db, "semantic_fts"):
                db.execute("DELETE FROM semantic_fts WHERE memory_id = ?", (saved.id,))
                db.execute(
                    """
                    INSERT INTO semantic_fts
                        (memory_id, subject, predicate, object, content)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        saved.id,
                        saved.subject or "",
                        saved.predicate or "",
                        saved.object or "",
                        saved.content,
                    ),
                )
            return saved

        return self._with_conn(conn, op)

    def get_semantic_memory(
        self,
        memory_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory | None:
        """Return one semantic memory by id."""

        def op(db: sqlite3.Connection) -> SemanticMemory | None:
            row = db.execute(
                "SELECT * FROM semantic_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
            return self._semantic_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def find_semantic_by_structure(
        self,
        *,
        subject: str | None,
        predicate: str | None,
        object_value: str | None,
        scope: MemoryScope,
        statuses: list[str] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[SemanticMemory]:
        """Find semantic memories with the same weak structure in one scope."""

        def op(db: sqlite3.Connection) -> list[SemanticMemory]:
            status_sql = ""
            params: list[Any] = [
                scope.scope_key,
                _normalize_optional(subject),
                _normalize_optional(predicate),
                _normalize_optional(object_value),
            ]
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                status_sql = f" AND status IN ({placeholders})"
                params.extend(statuses)
            rows = db.execute(
                f"""
                SELECT * FROM semantic_memories
                WHERE scope_key = ?
                  AND subject IS ?
                  AND predicate IS ?
                  AND object IS ?
                  {status_sql}
                ORDER BY updated_at DESC, id DESC
                """,
                params,
            ).fetchall()
            return [self._semantic_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def find_semantic_by_subject_predicate(
        self,
        *,
        subject: str,
        predicate: str,
        scope: MemoryScope,
        statuses: list[str] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[SemanticMemory]:
        """Find semantic memories sharing subject and predicate in one scope."""

        def op(db: sqlite3.Connection) -> list[SemanticMemory]:
            status_sql = ""
            params: list[Any] = [
                scope.scope_key,
                _normalize_optional(subject),
                _normalize_optional(predicate),
            ]
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                status_sql = f" AND status IN ({placeholders})"
                params.extend(statuses)
            rows = db.execute(
                f"""
                SELECT * FROM semantic_memories
                WHERE scope_key = ?
                  AND subject IS ?
                  AND predicate IS ?
                  {status_sql}
                ORDER BY updated_at DESC, id DESC
                """,
                params,
            ).fetchall()
            return [self._semantic_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def find_semantic_by_normalized_content(
        self,
        *,
        content: str,
        scope: MemoryScope,
        statuses: list[str] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[SemanticMemory]:
        """Find semantic memories with equivalent normalized content in one scope."""

        def op(db: sqlite3.Connection) -> list[SemanticMemory]:
            status_sql = ""
            params: list[Any] = [scope.scope_key, _normalize_content(content)]
            if statuses:
                placeholders = ",".join("?" for _ in statuses)
                status_sql = f" AND status IN ({placeholders})"
                params.extend(statuses)
            rows = db.execute(
                f"""
                SELECT * FROM semantic_memories
                WHERE scope_key = ? AND normalized_content = ?{status_sql}
                ORDER BY updated_at DESC, id DESC
                """,
                params,
            ).fetchall()
            return [self._semantic_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def forget_semantic_memory(
        self,
        memory_id: str,
        *,
        reason: str = "",
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory:
        """Mark a semantic memory deleted without removing audit evidence."""

        def op(db: sqlite3.Connection) -> SemanticMemory:
            existing = self.get_semantic_memory(memory_id, conn=db)
            if existing is None:
                raise KeyError(f"semantic memory not found: {memory_id}")
            now = utc_now_iso()
            metadata = dict(existing.metadata)
            if reason:
                metadata["forget_reason"] = reason
            deleted = SemanticMemory(
                id=existing.id,
                content=existing.content,
                memory_type=existing.memory_type,
                subject=existing.subject,
                predicate=existing.predicate,
                object=existing.object,
                entities=list(existing.entities),
                confidence=existing.confidence,
                salience=existing.salience,
                stability=existing.stability,
                source_memory_ids=list(existing.source_memory_ids),
                created_at=existing.created_at,
                updated_at=now,
                metadata=metadata,
                status="deleted",
                valid_from=existing.valid_from,
                valid_until=existing.valid_until,
                supersedes_id=existing.supersedes_id,
                superseded_by_id=existing.superseded_by_id,
                deleted_at=now,
                scope=existing.scope,
            )
            return self.upsert_semantic_memory(deleted, conn=db)

        return self._with_conn(conn, op)

    def list_semantic_memories(
        self,
        limit: int = 50,
        *,
        scopes: list[MemoryScope] | None = None,
        statuses: list[str] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[SemanticMemory]:
        """List semantic memories ordered by salience and recency."""

        scope_sql, scope_params = _scope_filter(scopes)
        status_sql = ""
        status_params: list[str] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            status_sql = f" AND status IN ({placeholders})"
            status_params = list(statuses)

        def op(db: sqlite3.Connection) -> list[SemanticMemory]:
            rows = db.execute(
                f"""
                SELECT * FROM semantic_memories
                WHERE 1 = 1{scope_sql}{status_sql}
                ORDER BY salience DESC, updated_at DESC LIMIT ?
                """,
                (*scope_params, *status_params, limit),
            ).fetchall()
            return [self._semantic_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def search_semantic(
        self,
        query: str,
        limit: int = 20,
        *,
        scopes: list[MemoryScope] | None = None,
        statuses: list[str] | None = None,
    ) -> list[SemanticMemory]:
        """Search semantic memories using FTS5 when available, otherwise LIKE."""

        scope_sql, scope_params = _scope_filter(scopes, column="m.scope_key")
        status_sql = ""
        status_params: list[str] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            status_sql = f" AND m.status IN ({placeholders})"
            status_params = list(statuses)
        with self.connect() as conn:
            if self._has_fts_table(conn, "semantic_fts") and query.strip():
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM semantic_fts f
                    JOIN semantic_memories m ON m.id = f.memory_id
                    WHERE semantic_fts MATCH ?
                    {scope_sql}{status_sql}
                    ORDER BY bm25(semantic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), *scope_params, *status_params, limit),
                ).fetchall()
            else:
                scope_sql, scope_params = _scope_filter(scopes)
                status_sql = ""
                status_params = []
                if statuses:
                    placeholders = ",".join("?" for _ in statuses)
                    status_sql = f" AND status IN ({placeholders})"
                    status_params = list(statuses)
                like = f"%{query}%"
                rows = conn.execute(
                    f"""
                    SELECT * FROM semantic_memories
                    WHERE (subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR content LIKE ?)
                    {scope_sql}{status_sql}
                    ORDER BY salience DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, *scope_params, *status_params, limit),
                ).fetchall()
        return [self._semantic_from_row(row) for row in rows]

    def upsert_procedural_memory(
        self,
        memory: ProceduralMemory,
        conn: sqlite3.Connection | None = None,
    ) -> ProceduralMemory:
        """Insert or update a procedural memory keyed by name and scope."""

        def op(db: sqlite3.Connection) -> ProceduralMemory:
            db.execute(
                """
                INSERT INTO procedural_memories
                    (id, name, description, trigger, procedure_markdown, success_count,
                     failure_count, confidence, created_at, updated_at,
                     scope_kind, scope_key, scope_metadata, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name, scope_key) DO UPDATE SET
                    description = excluded.description,
                    trigger = excluded.trigger,
                    procedure_markdown = excluded.procedure_markdown,
                    confidence = max(procedural_memories.confidence, excluded.confidence),
                    updated_at = excluded.updated_at,
                    scope_metadata = excluded.scope_metadata,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    memory.name,
                    memory.description,
                    memory.trigger,
                    memory.procedure_markdown,
                    memory.success_count,
                    memory.failure_count,
                    memory.confidence,
                    memory.created_at,
                    memory.updated_at,
                    *_scope_params(memory.scope),
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                "SELECT * FROM procedural_memories WHERE name = ? AND scope_key = ?",
                (memory.name, memory.scope.scope_key),
            ).fetchone()
            saved = self._procedural_from_row(row)
            if self._has_fts_table(db, "procedural_fts"):
                db.execute("DELETE FROM procedural_fts WHERE memory_id = ?", (saved.id,))
                db.execute(
                    """
                    INSERT INTO procedural_fts
                        (memory_id, name, description, trigger, procedure_markdown)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        saved.id,
                        saved.name,
                        saved.description,
                        saved.trigger,
                        saved.procedure_markdown,
                    ),
                )
            return saved

        return self._with_conn(conn, op)

    def list_procedural_memories(
        self,
        limit: int = 50,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[ProceduralMemory]:
        """List procedural memories."""

        scope_sql, scope_params = _scope_filter(scopes)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM procedural_memories
                WHERE 1 = 1{scope_sql}
                ORDER BY confidence DESC, updated_at DESC LIMIT ?
                """,
                (*scope_params, limit),
            ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def search_procedural(
        self,
        query: str,
        limit: int = 20,
        *,
        scopes: list[MemoryScope] | None = None,
    ) -> list[ProceduralMemory]:
        """Search procedural memories using FTS5 when available, otherwise LIKE."""

        scope_sql, scope_params = _scope_filter(scopes, column="m.scope_key")
        with self.connect() as conn:
            if self._has_fts_table(conn, "procedural_fts") and query.strip():
                rows = conn.execute(
                    f"""
                    SELECT m.* FROM procedural_fts f
                    JOIN procedural_memories m ON m.id = f.memory_id
                    WHERE procedural_fts MATCH ?
                    {scope_sql}
                    ORDER BY bm25(procedural_fts), m.confidence DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), *scope_params, limit),
                ).fetchall()
            else:
                scope_sql, scope_params = _scope_filter(scopes)
                like = f"%{query}%"
                rows = conn.execute(
                    f"""
                    SELECT * FROM procedural_memories
                    WHERE (name LIKE ?
                       OR description LIKE ?
                       OR trigger LIKE ?
                       OR procedure_markdown LIKE ?)
                    {scope_sql}
                    ORDER BY confidence DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, *scope_params, limit),
                ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def log_memory_access(
        self,
        memory_id: str,
        memory_type: str,
        query: str,
        score: float,
        *,
        scope: MemoryScope | None = None,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record that a memory was retrieved and update access counters when applicable."""

        def op(db: sqlite3.Connection) -> None:
            db.execute(
                """
                INSERT INTO memory_access_log
                    (id, memory_id, memory_type, query, accessed_at, score, scope_key, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_id("access"),
                    memory_id,
                    memory_type,
                    query,
                    utc_now_iso(),
                    score,
                    (scope or MemoryScope.default()).scope_key,
                    _dumps(metadata or {}),
                ),
            )
            if memory_type == "episodic":
                db.execute(
                    """
                    UPDATE episodic_memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    (utc_now_iso(), memory_id),
                )

        self._with_conn(conn, op)

    def count_memory_access(
        self,
        memory_id: str,
        memory_type: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> int:
        """Count access-log rows for one memory."""

        def op(db: sqlite3.Connection) -> int:
            row = db.execute(
                """
                SELECT count(*) AS count
                FROM memory_access_log
                WHERE memory_id = ? AND memory_type = ?
                """,
                (memory_id, memory_type),
            ).fetchone()
            return int(row["count"]) if row else 0

        return self._with_conn(conn, op)

    def list_stale_memory_candidates(
        self,
        *,
        older_than_days: int = 14,
        scopes: list[MemoryScope] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """Return reviewable candidates that have been idle past the cutoff."""

        cutoff = (utc_now() - timedelta(days=max(0, older_than_days))).isoformat()
        candidate_statuses = statuses or ["pending", "edited"]
        placeholders = ",".join("?" for _ in candidate_statuses)
        scope_sql, scope_params = _scope_filter(scopes)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_candidates
                WHERE status IN ({placeholders})
                  AND updated_at < ?
                  {scope_sql}
                ORDER BY updated_at ASC, id ASC
                LIMIT ?
                """,
                (*candidate_statuses, cutoff, *scope_params, limit),
            ).fetchall()
        return [self._memory_candidate_from_row(row) for row in rows]

    def list_inactive_semantic_memories(
        self,
        *,
        scopes: list[MemoryScope] | None = None,
        statuses: list[str] | None = None,
        limit: int = 50,
    ) -> list[SemanticMemory]:
        """Return inactive semantic memories for maintenance and audit."""

        inactive_statuses = statuses or ["superseded", "deleted", "conflict_review"]
        return self.list_semantic_memories(
            limit=limit,
            scopes=scopes,
            statuses=inactive_statuses,
        )

    def cleanup_inactive_semantic_search_index(
        self,
        *,
        scopes: list[MemoryScope] | None = None,
        statuses: list[str] | None = None,
        limit: int | None = None,
    ) -> list[SemanticMemory]:
        """Remove scoped inactive semantic memories from the optional FTS index."""

        inactive_statuses = statuses or ["superseded", "deleted", "conflict_review"]
        scope_sql, scope_params = _scope_filter(scopes, column="m.scope_key")

        def op(db: sqlite3.Connection) -> list[SemanticMemory]:
            if not self._has_fts_table(db, "semantic_fts"):
                return []
            placeholders = ",".join("?" for _ in inactive_statuses)
            limit_sql = " LIMIT ?" if limit is not None else ""
            params: list[Any] = [*inactive_statuses, *scope_params]
            if limit is not None:
                params.append(limit)
            rows = db.execute(
                f"""
                SELECT m.* FROM semantic_memories m
                JOIN semantic_fts f ON f.memory_id = m.id
                WHERE m.status IN ({placeholders})
                {scope_sql}
                ORDER BY m.updated_at DESC, m.id DESC
                {limit_sql}
                """,
                params,
            ).fetchall()
            memories = [self._semantic_from_row(row) for row in rows]
            for memory in memories:
                db.execute("DELETE FROM semantic_fts WHERE memory_id = ?", (memory.id,))
            return memories

        with self.immediate_transaction() as conn:
            return op(conn)

    def insert_memory_candidate(
        self,
        candidate: MemoryCandidate,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryCandidate:
        """Insert a stored memory candidate."""

        def op(db: sqlite3.Connection) -> MemoryCandidate:
            db.execute(
                """
                INSERT INTO memory_candidates
                    (id, candidate_type, proposed_layer, content, weak_structure,
                     salience, confidence, scope_kind, scope_key, scope_metadata,
                     source_message_ids, status, reviewer_metadata,
                     created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.candidate_type,
                    candidate.proposed_layer,
                    candidate.content,
                    _dumps(candidate.weak_structure),
                    candidate.salience,
                    candidate.confidence,
                    *_scope_params(candidate.scope),
                    _dumps(candidate.source_message_ids),
                    candidate.status,
                    _dumps(candidate.reviewer_metadata),
                    candidate.created_at,
                    candidate.updated_at,
                    _dumps(candidate.metadata),
                ),
            )
            return candidate

        return self._with_conn(conn, op)

    def list_memory_candidates(
        self,
        *,
        status: str | None = None,
        statuses: list[str] | None = None,
        scopes: list[MemoryScope] | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """List stored memory candidates for review or audit."""

        conditions = ["1 = 1"]
        params: list[Any] = []
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            conditions.append(f"status IN ({placeholders})")
            params.extend(statuses)
        scope_sql, scope_params = _scope_filter(scopes)
        query = f"""
            SELECT * FROM memory_candidates
            WHERE {' AND '.join(conditions)}{scope_sql}
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
        """
        params.extend(scope_params)
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._memory_candidate_from_row(row) for row in rows]

    def get_memory_candidate(
        self,
        candidate_id: str,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryCandidate | None:
        """Return one stored memory candidate by id."""

        def op(db: sqlite3.Connection) -> MemoryCandidate | None:
            row = db.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            return self._memory_candidate_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def update_memory_candidate_status(
        self,
        candidate_id: str,
        status: str,
        *,
        reviewer_metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryCandidate:
        """Update candidate status and return the stored row."""

        def op(db: sqlite3.Connection) -> MemoryCandidate:
            db.execute(
                """
                UPDATE memory_candidates
                SET status = ?,
                    reviewer_metadata = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    _dumps(reviewer_metadata or {}),
                    utc_now_iso(),
                    candidate_id,
                ),
            )
            row = db.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"memory candidate not found: {candidate_id}")
            return self._memory_candidate_from_row(row)

        return self._with_conn(conn, op)

    def update_memory_candidate_review(
        self,
        candidate_id: str,
        *,
        content: str,
        weak_structure: dict[str, Any],
        status: str,
        reviewer_metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryCandidate:
        """Update editable review fields without changing candidate source evidence."""

        def op(db: sqlite3.Connection) -> MemoryCandidate:
            db.execute(
                """
                UPDATE memory_candidates
                SET content = ?,
                    weak_structure = ?,
                    status = ?,
                    reviewer_metadata = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    _dumps(weak_structure),
                    status,
                    _dumps(reviewer_metadata or {}),
                    utc_now_iso(),
                    candidate_id,
                ),
            )
            row = db.execute(
                "SELECT * FROM memory_candidates WHERE id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"memory candidate not found: {candidate_id}")
            return self._memory_candidate_from_row(row)

        return self._with_conn(conn, op)

    def insert_memory_decision(
        self,
        decision: MemoryDecision,
        conn: sqlite3.Connection | None = None,
    ) -> MemoryDecision:
        """Insert an auditable memory decision."""

        def op(db: sqlite3.Connection) -> MemoryDecision:
            db.execute(
                """
                INSERT INTO memory_decisions
                    (id, candidate_id, action, memory_type, memory_id, reviewer,
                     rationale, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.id,
                    decision.candidate_id,
                    decision.action,
                    decision.memory_type,
                    decision.memory_id,
                    decision.reviewer,
                    decision.rationale,
                    decision.created_at,
                    _dumps(decision.metadata),
                ),
            )
            return decision

        return self._with_conn(conn, op)

    def list_memory_decisions(self, candidate_id: str) -> list[MemoryDecision]:
        """Return the ordered decision history for a stored candidate."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memory_decisions
                WHERE candidate_id = ?
                ORDER BY created_at ASC, rowid ASC
                """,
                (candidate_id,),
            ).fetchall()
        return [self._memory_decision_from_row(row) for row in rows]

    def drill_down_semantic_memory(self, memory_id: str) -> SemanticMemoryDrillDown:
        """Return projection evidence as active atomic memories and source messages."""

        root = self.get_semantic_memory(memory_id)
        if root is None:
            raise KeyError(f"semantic memory not found: {memory_id}")
        atomic_memories: list[SemanticMemory] = []
        message_ids: list[str] = []
        seen_memory_ids: set[str] = set()

        def visit(memory: SemanticMemory) -> None:
            if memory.id in seen_memory_ids:
                return
            seen_memory_ids.add(memory.id)
            for source_id in memory.source_memory_ids:
                source_memory = self.get_semantic_memory(source_id)
                if source_memory is None:
                    continue
                if source_memory.memory_type in {"scene", "persona"}:
                    visit(source_memory)
                    continue
                if source_memory.status == "active":
                    atomic_memories.append(source_memory)
                    for message_id in source_memory.source_memory_ids:
                        _append_unique(message_ids, message_id)
                    for message_id in _metadata_string_list(
                        source_memory.metadata,
                        "source_message_ids",
                    ):
                        _append_unique(message_ids, message_id)

        if root.memory_type in {"scene", "persona"}:
            visit(root)
        elif root.status == "active":
            atomic_memories.append(root)
            for source_id in root.source_memory_ids:
                _append_unique(message_ids, source_id)
        return SemanticMemoryDrillDown(
            memory=root,
            atomic_memories=_dedupe_memories(atomic_memories),
            source_messages=self.list_conversation_messages_by_ids(message_ids),
        )

    def list_entity_nodes(self, limit: int = 100) -> list[EntityNode]:
        """List entity nodes ordered by salience."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM entity_nodes ORDER BY salience DESC, name ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._entity_node_from_row(row) for row in rows]

    def list_relation_edges(self, limit: int = 100) -> list[RelationEdge]:
        """List graph relation edges with evidence memory ids."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM relation_edges ORDER BY confidence DESC, id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._relation_edge_from_row(row) for row in rows]

    def list_reviewed_semantic_memory_ids(
        self,
        *,
        statuses: list[str] | None = None,
    ) -> set[str]:
        """Return memory ids promoted from approved or auto-approved candidates."""

        candidate_statuses = statuses or ["approved", "auto_approved"]
        placeholders = ",".join("?" for _ in candidate_statuses)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT d.memory_id
                FROM memory_decisions d
                JOIN memory_candidates c ON c.id = d.candidate_id
                WHERE d.memory_id IS NOT NULL
                  AND c.status IN ({placeholders})
                  AND d.action IN ('store', 'update', 'merge', 'supersede')
                """,
                candidate_statuses,
            ).fetchall()
        return {str(row["memory_id"]) for row in rows if row["memory_id"]}

    def audit_relation_edges(
        self,
        *,
        source_name: str | None = None,
        target_name: str | None = None,
        relation_type: str | None = None,
        limit: int = 20,
    ) -> list[RelationEdgeAudit]:
        """Search relation edges and resolve active reviewed evidence for audit."""

        conditions: list[str] = []
        params: list[Any] = []
        if source_name:
            conditions.append("lower(source.name) = lower(?)")
            params.append(source_name)
        if target_name:
            conditions.append("lower(target.name) = lower(?)")
            params.append(target_name)
        if relation_type:
            conditions.append("edge.relation_type = ?")
            params.append(relation_type)
        where = " AND ".join(conditions) if conditions else "1 = 1"
        params.append(limit)
        reviewed_ids = self.list_reviewed_semantic_memory_ids()
        audits: list[RelationEdgeAudit] = []
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT
                    edge.id AS edge_id,
                    edge.source_node_id,
                    edge.target_node_id,
                    edge.relation_type,
                    edge.evidence_memory_ids,
                    edge.confidence,
                    edge.metadata AS edge_metadata,
                    source.id AS source_id,
                    source.name AS source_name,
                    source.kind AS source_kind,
                    source.aliases AS source_aliases,
                    source.salience AS source_salience,
                    source.metadata AS source_metadata,
                    target.id AS target_id,
                    target.name AS target_name,
                    target.kind AS target_kind,
                    target.aliases AS target_aliases,
                    target.salience AS target_salience,
                    target.metadata AS target_metadata
                FROM relation_edges edge
                JOIN entity_nodes source ON source.id = edge.source_node_id
                JOIN entity_nodes target ON target.id = edge.target_node_id
                WHERE {where}
                ORDER BY edge.confidence DESC, edge.id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
        for row in rows:
            edge = RelationEdge(
                id=row["edge_id"],
                source_node_id=row["source_node_id"],
                target_node_id=row["target_node_id"],
                relation_type=row["relation_type"],
                evidence_memory_ids=_loads_list(row["evidence_memory_ids"]),
                confidence=float(row["confidence"]),
                metadata=_loads_dict(row["edge_metadata"]),
            )
            evidence = [
                memory
                for memory_id in edge.evidence_memory_ids
                if (memory := self.get_semantic_memory(memory_id)) is not None
                and memory.status == "active"
                and memory.id in reviewed_ids
            ]
            source_messages = self.list_conversation_messages_by_ids(
                _dedupe_strings(
                    [message_id for memory in evidence for message_id in memory.source_memory_ids]
                )
            )
            if not evidence:
                continue
            audits.append(
                RelationEdgeAudit(
                    edge=edge,
                    source_node=EntityNode(
                        id=row["source_id"],
                        name=row["source_name"],
                        kind=row["source_kind"],
                        aliases=_loads_list(row["source_aliases"]),
                        salience=float(row["source_salience"]),
                        metadata=_loads_dict(row["source_metadata"]),
                    ),
                    target_node=EntityNode(
                        id=row["target_id"],
                        name=row["target_name"],
                        kind=row["target_kind"],
                        aliases=_loads_list(row["target_aliases"]),
                        salience=float(row["target_salience"]),
                        metadata=_loads_dict(row["target_metadata"]),
                    ),
                    evidence_memories=evidence,
                    source_messages=source_messages,
                )
            )
        return audits

    def audit_relation_edges_for_memory(
        self,
        memory_id: str,
        *,
        limit: int = 20,
    ) -> list[RelationEdgeAudit]:
        """Return relation audit edges that cite a semantic memory as evidence."""

        audits = self.audit_relation_edges(limit=1000)
        return [
            audit
            for audit in audits
            if any(memory.id == memory_id for memory in audit.evidence_memories)
        ][:limit]

    def stats(self) -> dict[str, int]:
        """Return counts by memory table."""

        tables = {
            "conversation_messages": "conversation_messages",
            "session_context_states": "session_context_states",
            "runtime_traces": "runtime_traces",
            "episodic": "episodic_memories",
            "semantic": "semantic_memories",
            "procedural": "procedural_memories",
            "memory_candidates": "memory_candidates",
            "memory_decisions": "memory_decisions",
            "entity_nodes": "entity_nodes",
            "relation_edges": "relation_edges",
        }
        with self.connect() as conn:
            return {
                key: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for key, table in tables.items()
            }

    def operational_metrics(self) -> dict[str, int | float]:
        """Return product and operations metrics for the memory lifecycle."""

        with self.connect() as conn:
            candidate_volume = int(
                conn.execute("SELECT count(*) FROM memory_candidates").fetchone()[0]
            )
            approved_candidates = int(
                conn.execute(
                    """
                    SELECT count(*) FROM memory_candidates
                    WHERE status IN ('approved', 'auto_approved')
                    """
                ).fetchone()[0]
            )
            rejected_candidates = int(
                conn.execute(
                    "SELECT count(*) FROM memory_candidates WHERE status = 'rejected'"
                ).fetchone()[0]
            )
            conflict_count = int(
                conn.execute(
                    """
                    SELECT count(*) FROM memory_decisions
                    WHERE action = 'conflict-review'
                    """
                ).fetchone()[0]
            )
            retrieval_count = int(
                conn.execute("SELECT count(*) FROM memory_access_log").fetchone()[0]
            )
            retrieval_hit_count = int(
                conn.execute(
                    "SELECT count(*) FROM memory_access_log WHERE score > 0"
                ).fetchone()[0]
            )
            forgotten_deleted_count = int(
                conn.execute(
                    "SELECT count(*) FROM semantic_memories WHERE status = 'deleted'"
                ).fetchone()[0]
            )
        decisioned = approved_candidates + rejected_candidates
        return {
            "candidate_volume": candidate_volume,
            "approved_candidates": approved_candidates,
            "rejected_candidates": rejected_candidates,
            "approval_rate": (
                approved_candidates / decisioned if decisioned > 0 else 0.0
            ),
            "conflict_count": conflict_count,
            "conflict_rate": (
                conflict_count / candidate_volume if candidate_volume > 0 else 0.0
            ),
            "retrieval_count": retrieval_count,
            "retrieval_hit_count": retrieval_hit_count,
            "retrieval_hit_rate": (
                retrieval_hit_count / retrieval_count if retrieval_count > 0 else 0.0
            ),
            "forgotten_deleted_count": forgotten_deleted_count,
        }

    def _fts_query(self, query: str) -> str:
        terms = [term.replace('"', "") for term in query.split() if term.strip()]
        return " OR ".join(f'"{term}"' for term in terms) or '""'

    def _latest_conversation_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            "SELECT max(ordinal) FROM conversation_messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row[0] or 0)

    def _next_conversation_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        return self._latest_conversation_ordinal(conn, session_id) + 1

    def _insert_conversation_message(
        self,
        db: sqlite3.Connection,
        message: ConversationMessage,
    ) -> ConversationMessage:
        if message.ordinal < 1:
            raise ValueError("conversation message ordinal must be greater than or equal to 1")
        if message.role not in {"user", "assistant", "tool"}:
            raise ValueError(f"unsupported conversation message role: {message.role}")
        db.execute(
            """
            INSERT INTO conversation_messages
                (id, session_id, ordinal, role, raw_content, model_content,
                 tool_call_id, tool_calls, tool_result_id, provider_metadata,
                 source_metadata, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.ordinal,
                message.role,
                message.raw_content,
                message.model_content,
                message.tool_call_id,
                _dumps(message.tool_calls),
                message.tool_result_id,
                _dumps(message.provider_metadata),
                _dumps(message.source_metadata),
                message.created_at,
                _dumps(message.metadata),
            ),
        )
        return message

    def _conversation_message_from_row(self, row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            id=row["id"],
            session_id=row["session_id"],
            ordinal=int(row["ordinal"]),
            role=row["role"],
            raw_content=row["raw_content"],
            model_content=row["model_content"],
            tool_call_id=row["tool_call_id"],
            tool_calls=_loads_dict_list(row["tool_calls"]),
            tool_result_id=row["tool_result_id"],
            provider_metadata=_loads_dict(row["provider_metadata"]),
            source_metadata=_loads_dict(row["source_metadata"]),
            created_at=row["created_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _session_context_state_from_row(self, row: sqlite3.Row) -> SessionContextState:
        return SessionContextState(
            session_id=row["session_id"],
            compressed_until_ordinal=int(row["compressed_until_ordinal"]),
            summary=row["summary"],
            summary_source_message_ids=_loads_list(row["summary_source_message_ids"]),
            compression_version=row["compression_version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _runtime_trace_from_row(self, row: sqlite3.Row) -> RuntimeTrace:
        return RuntimeTrace(
            id=row["id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            content=row["content"],
            timestamp=row["timestamp"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _episodic_from_row(self, row: sqlite3.Row) -> EpisodicMemory:
        return EpisodicMemory(
            id=row["id"],
            content=row["content"],
            summary=row["summary"],
            source_event_ids=_loads_list(row["source_event_ids"]),
            people=_loads_list(row["people"]),
            places=_loads_list(row["places"]),
            topics=_loads_list(row["topics"]),
            salience=float(row["salience"]),
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            last_accessed_at=row["last_accessed_at"],
            access_count=int(row["access_count"]),
            metadata=_loads_dict(row["metadata"]),
            scope=_scope_from_row(row),
        )

    def _semantic_from_row(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            content=row["content"],
            memory_type=row["memory_type"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            entities=_loads_list(row["entities"]),
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            stability=float(row["stability"]),
            source_memory_ids=_loads_list(row["source_memory_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
            status=row["status"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            supersedes_id=row["supersedes_id"],
            superseded_by_id=row["superseded_by_id"],
            deleted_at=row["deleted_at"],
            scope=_scope_from_row(row),
        )

    def _procedural_from_row(self, row: sqlite3.Row) -> ProceduralMemory:
        return ProceduralMemory(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            trigger=row["trigger"],
            procedure_markdown=row["procedure_markdown"],
            success_count=int(row["success_count"]),
            failure_count=int(row["failure_count"]),
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
            scope=_scope_from_row(row),
        )

    def _memory_candidate_from_row(self, row: sqlite3.Row) -> MemoryCandidate:
        return MemoryCandidate(
            id=row["id"],
            candidate_type=row["candidate_type"],
            proposed_layer=row["proposed_layer"],
            content=row["content"],
            weak_structure=_loads_dict(row["weak_structure"]),
            salience=float(row["salience"]),
            confidence=float(row["confidence"]),
            scope=_scope_from_row(row),
            source_message_ids=_loads_list(row["source_message_ids"]),
            status=row["status"],
            reviewer_metadata=_loads_dict(row["reviewer_metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _memory_decision_from_row(self, row: sqlite3.Row) -> MemoryDecision:
        return MemoryDecision(
            id=row["id"],
            candidate_id=row["candidate_id"],
            action=row["action"],
            memory_type=row["memory_type"],
            memory_id=row["memory_id"],
            reviewer=row["reviewer"],
            rationale=row["rationale"],
            created_at=row["created_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _entity_node_from_row(self, row: sqlite3.Row) -> EntityNode:
        return EntityNode(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            aliases=_loads_list(row["aliases"]),
            salience=float(row["salience"]),
            metadata=_loads_dict(row["metadata"]),
        )

    def _relation_edge_from_row(self, row: sqlite3.Row) -> RelationEdge:
        return RelationEdge(
            id=row["id"],
            source_node_id=row["source_node_id"],
            target_node_id=row["target_node_id"],
            relation_type=row["relation_type"],
            evidence_memory_ids=_loads_list(row["evidence_memory_ids"]),
            confidence=float(row["confidence"]),
            metadata=_loads_dict(row["metadata"]),
        )
