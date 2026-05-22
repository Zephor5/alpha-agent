"""SQLite persistence for Alpha Agent memory."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from alpha_agent.memory.models import (
    ConversationMessage,
    ConversationRole,
    EpisodicMemory,
    ProceduralMemory,
    RuntimeTrace,
    SemanticMemory,
    SessionContextState,
)
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

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
                     salience, confidence, created_at, last_accessed_at, access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def list_episodic_memories(self, limit: int = 50) -> list[EpisodicMemory]:
        """List recent episodic memories."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM episodic_memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def search_episodic(self, query: str, limit: int = 20) -> list[EpisodicMemory]:
        """Search episodic memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "episodic_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM episodic_fts f
                    JOIN episodic_memories m ON m.id = f.memory_id
                    WHERE episodic_fts MATCH ?
                    ORDER BY bm25(episodic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM episodic_memories
                    WHERE content LIKE ? OR summary LIKE ?
                    ORDER BY salience DESC, created_at DESC LIMIT ?
                    """,
                    (like, like, limit),
                ).fetchall()
        return [self._episodic_from_row(row) for row in rows]

    def upsert_semantic_memory(
        self,
        memory: SemanticMemory,
        conn: sqlite3.Connection | None = None,
    ) -> SemanticMemory:
        """Insert or update a semantic fact keyed by subject, predicate, and object."""

        def op(db: sqlite3.Connection) -> SemanticMemory:
            db.execute(
                """
                INSERT INTO semantic_memories
                    (id, subject, predicate, object, content, confidence, salience,
                     source_memory_ids, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject, predicate, object) DO UPDATE SET
                    content = excluded.content,
                    confidence = max(semantic_memories.confidence, excluded.confidence),
                    salience = max(semantic_memories.salience, excluded.salience),
                    source_memory_ids = excluded.source_memory_ids,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    memory.subject.lower().strip(),
                    memory.predicate.lower().strip(),
                    memory.object.lower().strip(),
                    memory.content,
                    memory.confidence,
                    memory.salience,
                    _dumps(memory.source_memory_ids),
                    memory.created_at,
                    memory.updated_at,
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                """
                SELECT * FROM semantic_memories
                WHERE subject = ? AND predicate = ? AND object = ?
                """,
                (
                    memory.subject.lower().strip(),
                    memory.predicate.lower().strip(),
                    memory.object.lower().strip(),
                ),
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
                    (saved.id, saved.subject, saved.predicate, saved.object, saved.content),
                )
            return saved

        return self._with_conn(conn, op)

    def list_semantic_memories(self, limit: int = 50) -> list[SemanticMemory]:
        """List semantic memories ordered by salience and recency."""

        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM semantic_memories ORDER BY salience DESC, updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._semantic_from_row(row) for row in rows]

    def search_semantic(self, query: str, limit: int = 20) -> list[SemanticMemory]:
        """Search semantic memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "semantic_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM semantic_fts f
                    JOIN semantic_memories m ON m.id = f.memory_id
                    WHERE semantic_fts MATCH ?
                    ORDER BY bm25(semantic_fts), m.salience DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM semantic_memories
                    WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? OR content LIKE ?
                    ORDER BY salience DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
        return [self._semantic_from_row(row) for row in rows]

    def upsert_procedural_memory(
        self,
        memory: ProceduralMemory,
        conn: sqlite3.Connection | None = None,
    ) -> ProceduralMemory:
        """Insert or update a procedural memory keyed by name."""

        def op(db: sqlite3.Connection) -> ProceduralMemory:
            db.execute(
                """
                INSERT INTO procedural_memories
                    (id, name, description, trigger, procedure_markdown, success_count,
                     failure_count, confidence, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    description = excluded.description,
                    trigger = excluded.trigger,
                    procedure_markdown = excluded.procedure_markdown,
                    confidence = max(procedural_memories.confidence, excluded.confidence),
                    updated_at = excluded.updated_at,
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
                    _dumps(memory.metadata),
                ),
            )
            row = db.execute(
                "SELECT * FROM procedural_memories WHERE name = ?",
                (memory.name,),
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

    def list_procedural_memories(self, limit: int = 50) -> list[ProceduralMemory]:
        """List procedural memories."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM procedural_memories
                ORDER BY confidence DESC, updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def search_procedural(self, query: str, limit: int = 20) -> list[ProceduralMemory]:
        """Search procedural memories using FTS5 when available, otherwise LIKE."""

        with self.connect() as conn:
            if self._has_fts_table(conn, "procedural_fts") and query.strip():
                rows = conn.execute(
                    """
                    SELECT m.* FROM procedural_fts f
                    JOIN procedural_memories m ON m.id = f.memory_id
                    WHERE procedural_fts MATCH ?
                    ORDER BY bm25(procedural_fts), m.confidence DESC
                    LIMIT ?
                    """,
                    (self._fts_query(query), limit),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT * FROM procedural_memories
                    WHERE name LIKE ?
                       OR description LIKE ?
                       OR trigger LIKE ?
                       OR procedure_markdown LIKE ?
                    ORDER BY confidence DESC, updated_at DESC LIMIT ?
                    """,
                    (like, like, like, like, limit),
                ).fetchall()
        return [self._procedural_from_row(row) for row in rows]

    def log_memory_access(
        self,
        memory_id: str,
        memory_type: str,
        query: str,
        score: float,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        """Record that a memory was retrieved and update access counters when applicable."""

        def op(db: sqlite3.Connection) -> None:
            db.execute(
                """
                INSERT INTO memory_access_log
                    (id, memory_id, memory_type, query, accessed_at, score, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (new_id("access"), memory_id, memory_type, query, utc_now_iso(), score, "{}"),
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

    def stats(self) -> dict[str, int]:
        """Return counts by memory table."""

        tables = {
            "conversation_messages": "conversation_messages",
            "session_context_states": "session_context_states",
            "runtime_traces": "runtime_traces",
            "episodic": "episodic_memories",
            "semantic": "semantic_memories",
            "procedural": "procedural_memories",
            "entity_nodes": "entity_nodes",
            "relation_edges": "relation_edges",
        }
        with self.connect() as conn:
            return {
                key: int(conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
                for key, table in tables.items()
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
        )

    def _semantic_from_row(self, row: sqlite3.Row) -> SemanticMemory:
        return SemanticMemory(
            id=row["id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            content=row["content"],
            confidence=float(row["confidence"]),
            salience=float(row["salience"]),
            source_memory_ids=_loads_list(row["source_memory_ids"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_loads_dict(row["metadata"]),
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
        )
