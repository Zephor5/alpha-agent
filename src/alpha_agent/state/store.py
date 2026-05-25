"""SQLite persistence for Alpha Agent session state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from alpha_agent.state.models import ConversationMessage, ConversationRole, RuntimeTrace
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


def _loads_dict_list(value: str | None) -> list[dict[str, Any]]:
    if not value:
        return []
    loaded = json.loads(value)
    if not isinstance(loaded, list):
        return []
    return [item for item in loaded if isinstance(item, dict)]


class StateStore:
    """Low-level SQLite operations for transcript, traces, and gateway state."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with row dictionaries enabled."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def initialize(self) -> None:
        """Create database tables."""

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with self.connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))

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

    def _with_conn(
        self,
        conn: sqlite3.Connection | None,
        fn: Callable[[sqlite3.Connection], T],
    ) -> T:
        if conn is not None:
            return fn(conn)
        with self.connect() as local:
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
        """Append an operational trace."""

        trace = RuntimeTrace(
            id=new_id("trace"),
            session_id=session_id,
            event_type=event_type,
            content=content,
            timestamp=timestamp or utc_now_iso(),
            metadata=metadata or {},
        )

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
        limit: int | None = None,
    ) -> list[RuntimeTrace]:
        """List runtime traces in chronological order."""

        conditions: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            conditions.append("session_id = ?")
            params.append(session_id)
        if event_type is not None:
            conditions.append("event_type = ?")
            params.append(event_type)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT * FROM runtime_traces
            {where}
            ORDER BY timestamp ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._runtime_trace_from_row(row) for row in rows]

    def _next_conversation_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        return self._latest_conversation_ordinal(conn, session_id) + 1

    def _latest_conversation_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(ordinal), 0) AS ordinal
            FROM conversation_messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return int(row["ordinal"]) if row is not None else 0

    def _insert_conversation_message(
        self,
        conn: sqlite3.Connection,
        message: ConversationMessage,
    ) -> ConversationMessage:
        conn.execute(
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

    def _runtime_trace_from_row(self, row: sqlite3.Row) -> RuntimeTrace:
        return RuntimeTrace(
            id=row["id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            content=row["content"],
            timestamp=row["timestamp"],
            metadata=_loads_dict(row["metadata"]),
        )
