"""SQLite persistence for Alpha Agent session state."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from alpha_agent.state.models import (
    ImportBatchRecord,
    ImportedConversationRecord,
    ImportedMessageRecord,
    ImportStatusSummary,
    LLMRole,
    RuntimeTrace,
    SessionCounterpart,
    SessionMessage,
    SessionMessageKind,
    SessionRecord,
    SessionSummarySnapshot,
)
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.system_reminder import SYSTEM_REMINDER_CLOSE, SYSTEM_REMINDER_OPEN
from alpha_agent.utils.time import local_timezone_identifier, utc_now_iso, validate_timezone

T = TypeVar("T")

REMINDER_TYPE_SESSION_TIME = "session_time"
REMINDER_TYPE_SELF_MEMORY_SUMMARY = "self_memory_summary"
REMINDER_TYPE_COUNTERPART_PROFILE = "counterpart_profile"
STABLE_CONTEXT_REMINDER_TYPES = frozenset(
    {
        REMINDER_TYPE_SELF_MEMORY_SUMMARY,
        REMINDER_TYPE_COUNTERPART_PROFILE,
    }
)
KNOWN_REMINDER_TYPES = frozenset(
    {
        REMINDER_TYPE_SESSION_TIME,
        *STABLE_CONTEXT_REMINDER_TYPES,
    }
)


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


def _system_reminder(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith(SYSTEM_REMINDER_OPEN) and stripped.endswith(SYSTEM_REMINDER_CLOSE):
        return stripped
    return f"{SYSTEM_REMINDER_OPEN}\n{stripped}\n{SYSTEM_REMINDER_CLOSE}"


def _validate_reminder_type(reminder_type: str) -> None:
    if reminder_type not in KNOWN_REMINDER_TYPES:
        allowed = ", ".join(sorted(KNOWN_REMINDER_TYPES))
        raise ValueError(f"unsupported reminder_type {reminder_type!r}; allowed: {allowed}")


def _normalize_session_timezone(timezone: str | None) -> str:
    if timezone is None:
        return local_timezone_identifier()
    return validate_timezone(timezone)


def _normalize_session_message_timestamps(message: SessionMessage) -> SessionMessage:
    return replace(
        message,
        created_at=_normalize_timestamp(message.created_at, "created_at"),
        updated_at=(
            _normalize_timestamp(message.updated_at, "updated_at")
            if message.updated_at is not None
            else None
        ),
    )


def _normalize_timestamp(value: str | datetime, field_name: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            raise ValueError(f"{field_name} must be a non-empty datetime")
        if raw_value.endswith("Z"):
            raw_value = f"{raw_value[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a parseable datetime") from exc
    else:
        raise ValueError(f"{field_name} must be a datetime or ISO datetime string")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).isoformat()


class StateStore:
    """Low-level SQLite operations for source messages, traces, and gateway state."""

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

    def get_session_record(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> SessionRecord | None:
        """Return durable session metadata, if it exists."""

        def op(db: sqlite3.Connection) -> SessionRecord | None:
            row = db.execute(
                """
                SELECT *
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return self._session_record_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def create_session_record(
        self,
        session_id: str,
        *,
        timezone: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionRecord:
        """Create durable session metadata."""

        normalized_timezone = _normalize_session_timezone(timezone)
        created_timestamp = _normalize_timestamp(
            created_at if created_at is not None else utc_now_iso(),
            "created_at",
        )
        updated_timestamp = (
            _normalize_timestamp(updated_at, "updated_at")
            if updated_at is not None
            else created_timestamp
        )

        def op(db: sqlite3.Connection) -> SessionRecord:
            db.execute(
                """
                INSERT INTO sessions
                    (session_id, timezone, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, normalized_timezone, created_timestamp, updated_timestamp),
            )
            record = self.get_session_record(session_id, conn=db)
            if record is None:
                raise RuntimeError(f"failed to create session record for {session_id!r}")
            return record

        return self._with_conn(conn, op)

    def ensure_session_record(
        self,
        session_id: str,
        *,
        timezone: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionRecord:
        """Create durable session metadata if missing, preserving existing timezone."""

        normalized_timezone = _normalize_session_timezone(timezone)
        created_timestamp = _normalize_timestamp(
            created_at if created_at is not None else utc_now_iso(),
            "created_at",
        )
        updated_timestamp = (
            _normalize_timestamp(updated_at, "updated_at")
            if updated_at is not None
            else created_timestamp
        )

        def op(db: sqlite3.Connection) -> SessionRecord:
            db.execute(
                """
                INSERT OR IGNORE INTO sessions
                    (session_id, timezone, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, normalized_timezone, created_timestamp, updated_timestamp),
            )
            record = self.get_session_record(session_id, conn=db)
            if record is None:
                raise RuntimeError(f"failed to ensure session record for {session_id!r}")
            return record

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def update_session_record_history_times(
        self,
        session_id: str,
        *,
        created_at: str | None = None,
        updated_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionRecord:
        """Update persisted session history timestamps without changing timezone."""

        normalized_created_at = (
            _normalize_timestamp(created_at, "created_at") if created_at is not None else None
        )
        normalized_updated_at = (
            _normalize_timestamp(updated_at, "updated_at") if updated_at is not None else None
        )

        def op(db: sqlite3.Connection) -> SessionRecord:
            record = self.get_session_record(session_id, conn=db)
            if record is None:
                raise KeyError(f"session record {session_id!r} not found")
            db.execute(
                """
                UPDATE sessions
                SET created_at = ?,
                    updated_at = ?
                WHERE session_id = ?
                """,
                (
                    normalized_created_at or record.created_at,
                    normalized_updated_at or record.updated_at,
                    session_id,
                ),
            )
            updated = self.get_session_record(session_id, conn=db)
            if updated is None:
                raise RuntimeError(f"failed to update session record for {session_id!r}")
            return updated

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def append_session_message(
        self,
        *,
        session_id: str,
        kind: SessionMessageKind,
        llm_role: LLMRole | None,
        raw_content: str,
        model_content: str | None = None,
        reasoning_content: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_result_id: str | None = None,
        provider_metadata: dict[str, Any] | None = None,
        source_metadata: dict[str, Any] | None = None,
        compression_point_ordinal: int | None = None,
        compression_version: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Append a source message with the next monotonic session ordinal."""

        self._validate_session_message_shape(kind=kind, llm_role=llm_role)

        def op(db: sqlite3.Connection) -> SessionMessage:
            message = SessionMessage(
                id=new_id("msg"),
                session_id=session_id,
                ordinal=self._next_session_ordinal(db, session_id),
                kind=kind,
                llm_role=llm_role,
                raw_content=raw_content,
                model_content=model_content,
                reasoning_content=reasoning_content,
                tool_call_id=tool_call_id,
                tool_calls=tool_calls or [],
                tool_result_id=tool_result_id,
                provider_metadata=provider_metadata or {},
                source_metadata=source_metadata or {},
                compression_point_ordinal=compression_point_ordinal,
                compression_version=compression_version,
                created_at=created_at if created_at is not None else utc_now_iso(),
                updated_at=updated_at,
                metadata=metadata or {},
            )
            normalized_message = _normalize_session_message_timestamps(message)
            self.ensure_session_record(normalized_message.session_id, conn=db)
            return self._insert_session_message(db, normalized_message)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def append_compressed_message(
        self,
        *,
        session_id: str,
        raw_content: str,
        compression_point_ordinal: int,
        compression_version: str,
        provider_metadata: dict[str, Any] | None = None,
        source_metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Append a synthetic handover message for LLM replay continuity."""

        if compression_point_ordinal < 1:
            raise ValueError("compression_point_ordinal must be greater than 0")
        return self.append_session_message(
            session_id=session_id,
            kind="compressed_message",
            llm_role="user",
            raw_content=_system_reminder(raw_content),
            compression_point_ordinal=compression_point_ordinal,
            compression_version=compression_version,
            provider_metadata=provider_metadata,
            source_metadata=source_metadata,
            created_at=created_at,
            metadata=metadata,
            conn=conn,
        )

    def append_session_time_reminder(
        self,
        *,
        session_id: str,
        raw_content: str,
        reminder_kind: str,
        local_datetime: str,
        local_date: str,
        created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Append a local-time system reminder to the session source stream."""

        return self.append_session_reminder(
            session_id=session_id,
            raw_content=raw_content,
            reminder_type=REMINDER_TYPE_SESSION_TIME,
            created_at=created_at,
            metadata={
                "time_reminder_kind": reminder_kind,
                "local_date": local_date,
                "local_datetime": local_datetime,
            },
            conn=conn,
        )

    def append_session_reminder(
        self,
        *,
        session_id: str,
        raw_content: str,
        reminder_type: str,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Append a typed system reminder to the durable session source stream."""

        _validate_reminder_type(reminder_type)
        reminder_metadata = {
            **dict(metadata or {}),
            "reminder_type": reminder_type,
        }
        return self.append_session_message(
            session_id=session_id,
            kind="system_reminder",
            llm_role="user",
            raw_content=_system_reminder(raw_content),
            created_at=created_at,
            metadata=reminder_metadata,
            conn=conn,
        )

    def insert_session_message(
        self,
        message: SessionMessage,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Insert a source message only if it is the next ordinal for its session."""

        self._validate_session_message_shape(kind=message.kind, llm_role=message.llm_role)

        def op(db: sqlite3.Connection) -> SessionMessage:
            normalized_message = _normalize_session_message_timestamps(message)
            expected_ordinal = self._next_session_ordinal(db, message.session_id)
            if normalized_message.ordinal != expected_ordinal:
                raise ValueError(
                    "session message ordinal for session "
                    f"{message.session_id!r} must be {expected_ordinal}, got "
                    f"{normalized_message.ordinal}"
                )
            self.ensure_session_record(normalized_message.session_id, conn=db)
            return self._insert_session_message(db, normalized_message)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def list_session_messages(
        self,
        session_id: str,
        *,
        after_ordinal: int | None = None,
        before_ordinal: int | None = None,
        limit: int | None = None,
    ) -> list[SessionMessage]:
        """List source messages in ascending ordinal order."""

        conditions = ["session_id = ?"]
        params: list[Any] = [session_id]
        if after_ordinal is not None:
            conditions.append("ordinal > ?")
            params.append(after_ordinal)
        if before_ordinal is not None:
            conditions.append("ordinal < ?")
            params.append(before_ordinal)
        query = f"""
            SELECT * FROM session_messages
            WHERE {' AND '.join(conditions)}
            ORDER BY ordinal ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._session_message_from_row(row) for row in rows]

    def list_session_messages_by_ids(
        self,
        message_ids: list[str],
    ) -> list[SessionMessage]:
        """Return source messages for the given ids, preserving requested order."""

        if not message_ids:
            return []
        placeholders = ",".join("?" for _ in message_ids)
        with self.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM session_messages
                WHERE id IN ({placeholders})
                """,
                message_ids,
            ).fetchall()
        by_id = {str(row["id"]): self._session_message_from_row(row) for row in rows}
        return [by_id[message_id] for message_id in message_ids if message_id in by_id]

    def list_session_ids(self) -> list[str]:
        """List known session ids from session records, source messages, or traces."""

        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id
                FROM sessions
                UNION
                SELECT session_id
                FROM session_messages
                UNION
                SELECT session_id
                FROM runtime_traces
                ORDER BY session_id ASC
                """
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def list_session_records(
        self,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> list[SessionRecord]:
        """List durable session records by creation time with a stable id tie-break."""

        def op(db: sqlite3.Connection) -> list[SessionRecord]:
            rows = db.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY created_at ASC, session_id ASC
                """
            ).fetchall()
            return [self._session_record_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def latest_session_ordinal(self, session_id: str) -> int:
        """Return the latest source ordinal for a session, or zero if it has none."""

        with self.connect() as conn:
            return self._latest_session_ordinal(conn, session_id)

    def find_latest_session_time_reminder(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage | None:
        """Return the newest local-time reminder in a session, if any."""

        return self.find_latest_session_reminder(
            session_id,
            reminder_type=REMINDER_TYPE_SESSION_TIME,
            conn=conn,
        )

    def find_latest_session_reminder(
        self,
        session_id: str,
        *,
        reminder_type: str,
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage | None:
        """Return the newest reminder of one concrete semantic type."""

        _validate_reminder_type(reminder_type)

        def op(db: sqlite3.Connection) -> SessionMessage | None:
            row = db.execute(
                """
                SELECT * FROM session_messages
                WHERE session_id = ?
                  AND kind = 'system_reminder'
                  AND json_extract(metadata, '$.reminder_type') = ?
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                (session_id, reminder_type),
            ).fetchone()
            return self._session_message_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def list_session_reminders(
        self,
        session_id: str,
        *,
        reminder_type: str,
        conn: sqlite3.Connection | None = None,
    ) -> list[SessionMessage]:
        """Return reminders of one concrete semantic type in source order."""

        _validate_reminder_type(reminder_type)

        def op(db: sqlite3.Connection) -> list[SessionMessage]:
            rows = db.execute(
                """
                SELECT * FROM session_messages
                WHERE session_id = ?
                  AND kind = 'system_reminder'
                  AND json_extract(metadata, '$.reminder_type') = ?
                ORDER BY ordinal ASC
                """,
                (session_id, reminder_type),
            ).fetchall()
            return [self._session_message_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def find_latest_compressed_message(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> SessionMessage | None:
        """Return the newest compressed source message by session ordinal."""

        conditions = ["session_id = ?", "kind = 'compressed_message'"]
        params: list[Any] = [session_id]
        if before_ordinal is not None:
            conditions.append("ordinal < ?")
            params.append(before_ordinal)
        with self.connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM session_messages
                WHERE {' AND '.join(conditions)}
                ORDER BY ordinal DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        return self._session_message_from_row(row) if row is not None else None

    def get_session_counterpart(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> SessionCounterpart | None:
        """Return the counterpart bound to a session, if any."""

        def op(db: sqlite3.Connection) -> SessionCounterpart | None:
            row = db.execute(
                """
                SELECT *
                FROM session_counterparts
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return self._session_counterpart_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def create_session_counterpart(
        self,
        *,
        session_id: str,
        counterpart_id: str,
        source_metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionCounterpart:
        """Bind a session to its first counterpart, or return the existing binding."""

        timestamp = created_at or utc_now_iso()

        def op(db: sqlite3.Connection) -> SessionCounterpart:
            db.execute(
                """
                INSERT OR IGNORE INTO session_counterparts
                    (session_id, counterpart_id, source_metadata, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, counterpart_id, _dumps(source_metadata or {}), timestamp),
            )
            binding = self.get_session_counterpart(session_id, conn=db)
            if binding is None:
                raise RuntimeError(f"failed to bind counterpart for session {session_id!r}")
            return binding

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def create_import_batch(
        self,
        *,
        batch_id: str,
        source_provider: str,
        input_name: str | None,
        payload_digest: str,
        status: str,
        conversations_seen: int,
        messages_seen: int,
        conversations_created: int,
        conversations_reused: int,
        messages_inserted: int,
        messages_deduped: int,
        error_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ImportBatchRecord:
        """Record one external conversation import attempt summary."""

        timestamp = _normalize_timestamp(created_at or utc_now_iso(), "created_at")
        updated_timestamp = _normalize_timestamp(updated_at or timestamp, "updated_at")

        def op(db: sqlite3.Connection) -> ImportBatchRecord:
            db.execute(
                """
                INSERT INTO import_batches
                    (
                        id,
                        source_provider,
                        input_name,
                        payload_digest,
                        status,
                        conversations_seen,
                        messages_seen,
                        conversations_created,
                        conversations_reused,
                        messages_inserted,
                        messages_deduped,
                        error_summary,
                        metadata,
                        created_at,
                        updated_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    source_provider,
                    input_name,
                    payload_digest,
                    status,
                    conversations_seen,
                    messages_seen,
                    conversations_created,
                    conversations_reused,
                    messages_inserted,
                    messages_deduped,
                    error_summary,
                    _dumps(metadata or {}),
                    timestamp,
                    updated_timestamp,
                ),
            )
            batch = self.get_import_batch(batch_id, conn=db)
            if batch is None:
                raise RuntimeError(f"failed to create import batch {batch_id!r}")
            return batch

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def get_import_batch(
        self,
        batch_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> ImportBatchRecord | None:
        """Return one import batch summary, if it exists."""

        def op(db: sqlite3.Connection) -> ImportBatchRecord | None:
            row = db.execute(
                """
                SELECT *
                FROM import_batches
                WHERE id = ?
                """,
                (batch_id,),
            ).fetchone()
            return self._import_batch_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def list_import_batches(
        self,
        *,
        source_provider: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[ImportBatchRecord]:
        """List durable import batch summaries in creation order."""

        def op(db: sqlite3.Connection) -> list[ImportBatchRecord]:
            params: list[Any] = []
            where = ""
            if source_provider is not None:
                where = "WHERE source_provider = ?"
                params.append(source_provider)
            rows = db.execute(
                f"""
                SELECT *
                FROM import_batches
                {where}
                ORDER BY created_at ASC, id ASC
                """,
                params,
            ).fetchall()
            return [self._import_batch_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def get_imported_conversation(
        self,
        source_provider: str,
        external_conversation_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> ImportedConversationRecord | None:
        """Return an imported conversation mapping by external identity."""

        def op(db: sqlite3.Connection) -> ImportedConversationRecord | None:
            row = db.execute(
                """
                SELECT *
                FROM imported_conversations
                WHERE source_provider = ?
                  AND external_conversation_id = ?
                """,
                (source_provider, external_conversation_id),
            ).fetchone()
            return self._imported_conversation_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def get_imported_conversation_by_session(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> ImportedConversationRecord | None:
        """Return an imported conversation mapping by hidden session id."""

        def op(db: sqlite3.Connection) -> ImportedConversationRecord | None:
            row = db.execute(
                """
                SELECT *
                FROM imported_conversations
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return self._imported_conversation_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def is_import_session(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> bool:
        """Return whether a session is a hidden imported conversation session."""

        return self.get_imported_conversation_by_session(session_id, conn=conn) is not None

    def create_or_reuse_imported_conversation(
        self,
        *,
        source_provider: str,
        external_conversation_id: str,
        session_id: str,
        title: str | None,
        external_created_at: str | None,
        external_updated_at: str | None,
        first_import_batch_id: str,
        latest_import_batch_id: str,
        metadata: dict[str, Any] | None = None,
        imported_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> tuple[ImportedConversationRecord, bool]:
        """Create an imported conversation mapping, or mark an existing one reused."""

        timestamp = _normalize_timestamp(imported_at or utc_now_iso(), "imported_at")
        normalized_external_created_at = (
            _normalize_timestamp(external_created_at, "external_created_at")
            if external_created_at is not None
            else None
        )
        normalized_external_updated_at = (
            _normalize_timestamp(external_updated_at, "external_updated_at")
            if external_updated_at is not None
            else None
        )

        def op(db: sqlite3.Connection) -> tuple[ImportedConversationRecord, bool]:
            existing = self.get_imported_conversation(
                source_provider,
                external_conversation_id,
                conn=db,
            )
            if existing is not None:
                db.execute(
                    """
                    UPDATE imported_conversations
                    SET latest_import_batch_id = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (latest_import_batch_id, timestamp, existing.id),
                )
                updated = self.get_imported_conversation(
                    source_provider,
                    external_conversation_id,
                    conn=db,
                )
                if updated is None:
                    raise RuntimeError(
                        "failed to reload imported conversation "
                        f"{source_provider!r}/{external_conversation_id!r}"
                    )
                return updated, False

            conversation_id = new_id("import_conv")
            db.execute(
                """
                INSERT INTO imported_conversations
                    (
                        id,
                        source_provider,
                        external_conversation_id,
                        session_id,
                        title,
                        external_created_at,
                        external_updated_at,
                        first_import_batch_id,
                        latest_import_batch_id,
                        created_at,
                        updated_at,
                        metadata
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    source_provider,
                    external_conversation_id,
                    session_id,
                    title,
                    normalized_external_created_at,
                    normalized_external_updated_at,
                    first_import_batch_id,
                    latest_import_batch_id,
                    timestamp,
                    timestamp,
                    _dumps(metadata or {}),
                ),
            )
            created = self.get_imported_conversation(
                source_provider,
                external_conversation_id,
                conn=db,
            )
            if created is None:
                raise RuntimeError(
                    "failed to create imported conversation "
                    f"{source_provider!r}/{external_conversation_id!r}"
                )
            return created, True

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def list_imported_external_message_ids(
        self,
        source_provider: str,
        external_conversation_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> set[str]:
        """Return known external message ids for one imported conversation."""

        def op(db: sqlite3.Connection) -> set[str]:
            rows = db.execute(
                """
                SELECT external_message_id
                FROM imported_messages
                WHERE source_provider = ?
                  AND external_conversation_id = ?
                """,
                (source_provider, external_conversation_id),
            ).fetchall()
            return {str(row["external_message_id"]) for row in rows}

        return self._with_conn(conn, op)

    def latest_imported_message_timestamp(
        self,
        source_provider: str,
        external_conversation_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> str | None:
        """Return the latest imported external message timestamp for a conversation."""

        def op(db: sqlite3.Connection) -> str | None:
            row = db.execute(
                """
                SELECT external_created_at
                FROM imported_messages
                WHERE source_provider = ?
                  AND external_conversation_id = ?
                ORDER BY external_created_at DESC, id DESC
                LIMIT 1
                """,
                (source_provider, external_conversation_id),
            ).fetchone()
            return str(row["external_created_at"]) if row is not None else None

        return self._with_conn(conn, op)

    def create_imported_message(
        self,
        *,
        source_provider: str,
        external_conversation_id: str,
        external_message_id: str,
        imported_conversation_id: str,
        session_message_id: str,
        import_batch_id: str,
        role: str,
        external_created_at: str,
        metadata: dict[str, Any] | None = None,
        imported_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> ImportedMessageRecord:
        """Create the external-message to session-message mapping."""

        timestamp = _normalize_timestamp(imported_at or utc_now_iso(), "imported_at")
        normalized_external_created_at = _normalize_timestamp(
            external_created_at,
            "external_created_at",
        )

        def op(db: sqlite3.Connection) -> ImportedMessageRecord:
            message_id = new_id("import_msg")
            db.execute(
                """
                INSERT INTO imported_messages
                    (
                        id,
                        source_provider,
                        external_conversation_id,
                        external_message_id,
                        imported_conversation_id,
                        session_message_id,
                        import_batch_id,
                        role,
                        external_created_at,
                        imported_at,
                        metadata
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    source_provider,
                    external_conversation_id,
                    external_message_id,
                    imported_conversation_id,
                    session_message_id,
                    import_batch_id,
                    role,
                    normalized_external_created_at,
                    timestamp,
                    _dumps(metadata or {}),
                ),
            )
            row = db.execute(
                """
                SELECT *
                FROM imported_messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"failed to create imported message {message_id!r}")
            return self._imported_message_from_row(row)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def list_imported_messages(
        self,
        *,
        source_provider: str | None = None,
        external_conversation_id: str | None = None,
        import_batch_id: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> list[ImportedMessageRecord]:
        """List imported message mappings."""

        def op(db: sqlite3.Connection) -> list[ImportedMessageRecord]:
            conditions: list[str] = []
            params: list[Any] = []
            if source_provider is not None:
                conditions.append("source_provider = ?")
                params.append(source_provider)
            if external_conversation_id is not None:
                conditions.append("external_conversation_id = ?")
                params.append(external_conversation_id)
            if import_batch_id is not None:
                conditions.append("import_batch_id = ?")
                params.append(import_batch_id)
            where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            rows = db.execute(
                f"""
                SELECT *
                FROM imported_messages
                {where}
                ORDER BY source_provider ASC,
                         external_conversation_id ASC,
                         external_created_at ASC,
                         id ASC
                """,
                params,
            ).fetchall()
            return [self._imported_message_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def get_import_status_summary(
        self,
        batch_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> ImportStatusSummary | None:
        """Return aggregate import counts plus extraction progress for one batch."""

        def op(db: sqlite3.Connection) -> ImportStatusSummary | None:
            batch = self.get_import_batch(batch_id, conn=db)
            if batch is None:
                return None
            progress_rows = db.execute(
                """
                SELECT COALESCE(progress.status, 'pending') AS status,
                       COUNT(*) AS count
                FROM imported_messages AS imported
                JOIN imported_conversations AS conversation
                  ON conversation.id = imported.imported_conversation_id
                LEFT JOIN background_source_progress AS progress
                  ON progress.source_type = 'session_message'
                 AND progress.source_id = imported.session_message_id
                 AND progress.stage = 'extraction'
                 AND progress.target_unit = 'session:' || conversation.session_id
                WHERE imported.import_batch_id = ?
                GROUP BY COALESCE(progress.status, 'pending')
                """,
                (batch_id,),
            ).fetchall()
            counts = {str(row["status"]): int(row["count"]) for row in progress_rows}
            return ImportStatusSummary(
                batch_id=batch.id,
                source_provider=batch.source_provider,
                status=batch.status,
                conversations_seen=batch.conversations_seen,
                messages_seen=batch.messages_seen,
                conversations_created=batch.conversations_created,
                conversations_reused=batch.conversations_reused,
                messages_inserted=batch.messages_inserted,
                messages_deduped=batch.messages_deduped,
                extraction_pending=counts.get("pending", 0),
                extraction_claimed=counts.get("claimed", 0),
                extraction_processed=counts.get("processed", 0),
                extraction_failed=counts.get("failed", 0),
                extraction_skipped=counts.get("skipped", 0),
                error_summary=batch.error_summary,
                metadata=batch.metadata,
                created_at=batch.created_at,
                updated_at=batch.updated_at,
            )

        return self._with_conn(conn, op)

    def get_session_summary_snapshot(
        self,
        session_id: str,
        summary_kind: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> SessionSummarySnapshot | None:
        """Return one stable summary snapshot for a session, if any."""

        def op(db: sqlite3.Connection) -> SessionSummarySnapshot | None:
            row = db.execute(
                """
                SELECT *
                FROM session_summary_snapshots
                WHERE session_id = ?
                  AND summary_kind = ?
                """,
                (session_id, summary_kind),
            ).fetchone()
            return self._session_summary_snapshot_from_row(row) if row is not None else None

        return self._with_conn(conn, op)

    def list_session_summary_snapshots(
        self,
        session_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> list[SessionSummarySnapshot]:
        """Return stable summary snapshots selected for one session."""

        def op(db: sqlite3.Connection) -> list[SessionSummarySnapshot]:
            rows = db.execute(
                """
                SELECT *
                FROM session_summary_snapshots
                WHERE session_id = ?
                ORDER BY summary_kind
                """,
                (session_id,),
            ).fetchall()
            return [self._session_summary_snapshot_from_row(row) for row in rows]

        return self._with_conn(conn, op)

    def create_session_summary_snapshot(
        self,
        *,
        session_id: str,
        summary_kind: str,
        target_kind: str,
        target_id: str,
        source_belief_id: str,
        content: str,
        created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SessionSummarySnapshot:
        """Create or return one stable summary snapshot for a session."""

        timestamp = created_at or utc_now_iso()

        def op(db: sqlite3.Connection) -> SessionSummarySnapshot:
            db.execute(
                """
                INSERT OR IGNORE INTO session_summary_snapshots
                    (
                        session_id,
                        summary_kind,
                        target_kind,
                        target_id,
                        source_belief_id,
                        content,
                        created_at
                    )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    summary_kind,
                    target_kind,
                    target_id,
                    source_belief_id,
                    content,
                    timestamp,
                ),
            )
            snapshot = self.get_session_summary_snapshot(
                session_id,
                summary_kind,
                conn=db,
            )
            if snapshot is None:
                raise RuntimeError(
                    f"failed to create {summary_kind!r} summary snapshot for {session_id!r}"
                )
            return snapshot

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

    def update_session_message_replay_payload(
        self,
        message_id: str,
        *,
        raw_content: str,
        model_content: str | None,
        tool_calls: list[dict[str, Any]],
        metadata: dict[str, Any],
        conn: sqlite3.Connection | None = None,
    ) -> SessionMessage:
        """Update only replay payload fields and general metadata for a source message."""

        def op(db: sqlite3.Connection) -> SessionMessage:
            db.execute(
                """
                UPDATE session_messages
                SET raw_content = ?,
                    model_content = ?,
                    tool_calls = ?,
                    metadata = ?
                WHERE id = ?
                """,
                (
                    raw_content,
                    model_content,
                    _dumps(tool_calls),
                    _dumps(metadata),
                    message_id,
                ),
            )
            row = db.execute(
                """
                SELECT * FROM session_messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"session message {message_id!r} not found")
            return self._session_message_from_row(row)

        if conn is not None:
            return op(conn)
        with self.immediate_transaction() as local:
            return op(local)

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

    def _next_session_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        return self._latest_session_ordinal(conn, session_id) + 1

    def _latest_session_ordinal(self, conn: sqlite3.Connection, session_id: str) -> int:
        row = conn.execute(
            """
            SELECT COALESCE(MAX(ordinal), 0) AS ordinal
            FROM session_messages
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
        return int(row["ordinal"]) if row is not None else 0

    def _insert_session_message(
        self,
        conn: sqlite3.Connection,
        message: SessionMessage,
    ) -> SessionMessage:
        conn.execute(
            """
            INSERT INTO session_messages
                (id, session_id, ordinal, kind, llm_role, raw_content, model_content,
                 reasoning_content, tool_call_id, tool_calls, tool_result_id, provider_metadata,
                 source_metadata, compression_point_ordinal, compression_version,
                 metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message.id,
                message.session_id,
                message.ordinal,
                message.kind,
                message.llm_role,
                message.raw_content,
                message.model_content,
                message.reasoning_content,
                message.tool_call_id,
                _dumps(message.tool_calls),
                message.tool_result_id,
                _dumps(message.provider_metadata),
                _dumps(message.source_metadata),
                message.compression_point_ordinal,
                message.compression_version,
                _dumps(message.metadata),
                message.created_at,
                message.updated_at,
            ),
        )
        return message

    def _session_record_from_row(self, row: sqlite3.Row) -> SessionRecord:
        return SessionRecord(
            session_id=row["session_id"],
            timezone=row["timezone"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _session_message_from_row(self, row: sqlite3.Row) -> SessionMessage:
        return SessionMessage(
            id=row["id"],
            session_id=row["session_id"],
            ordinal=int(row["ordinal"]),
            kind=row["kind"],
            llm_role=row["llm_role"],
            raw_content=row["raw_content"],
            model_content=row["model_content"],
            reasoning_content=row["reasoning_content"],
            tool_call_id=row["tool_call_id"],
            tool_calls=_loads_dict_list(row["tool_calls"]),
            tool_result_id=row["tool_result_id"],
            provider_metadata=_loads_dict(row["provider_metadata"]),
            source_metadata=_loads_dict(row["source_metadata"]),
            compression_point_ordinal=(
                int(row["compression_point_ordinal"])
                if row["compression_point_ordinal"] is not None
                else None
            ),
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

    def _import_batch_from_row(self, row: sqlite3.Row) -> ImportBatchRecord:
        return ImportBatchRecord(
            id=row["id"],
            source_provider=row["source_provider"],
            input_name=row["input_name"],
            payload_digest=row["payload_digest"],
            status=row["status"],
            conversations_seen=int(row["conversations_seen"]),
            messages_seen=int(row["messages_seen"]),
            conversations_created=int(row["conversations_created"]),
            conversations_reused=int(row["conversations_reused"]),
            messages_inserted=int(row["messages_inserted"]),
            messages_deduped=int(row["messages_deduped"]),
            error_summary=row["error_summary"],
            metadata=_loads_dict(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _imported_conversation_from_row(
        self,
        row: sqlite3.Row,
    ) -> ImportedConversationRecord:
        return ImportedConversationRecord(
            id=row["id"],
            source_provider=row["source_provider"],
            external_conversation_id=row["external_conversation_id"],
            session_id=row["session_id"],
            title=row["title"],
            external_created_at=row["external_created_at"],
            external_updated_at=row["external_updated_at"],
            first_import_batch_id=row["first_import_batch_id"],
            latest_import_batch_id=row["latest_import_batch_id"],
            metadata=_loads_dict(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _imported_message_from_row(self, row: sqlite3.Row) -> ImportedMessageRecord:
        return ImportedMessageRecord(
            id=row["id"],
            source_provider=row["source_provider"],
            external_conversation_id=row["external_conversation_id"],
            external_message_id=row["external_message_id"],
            imported_conversation_id=row["imported_conversation_id"],
            session_message_id=row["session_message_id"],
            import_batch_id=row["import_batch_id"],
            role=row["role"],
            external_created_at=row["external_created_at"],
            imported_at=row["imported_at"],
            metadata=_loads_dict(row["metadata"]),
        )

    def _session_summary_snapshot_from_row(
        self,
        row: sqlite3.Row,
    ) -> SessionSummarySnapshot:
        return SessionSummarySnapshot(
            session_id=row["session_id"],
            summary_kind=row["summary_kind"],
            target_kind=row["target_kind"],
            target_id=row["target_id"],
            source_belief_id=row["source_belief_id"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def _session_counterpart_from_row(
        self,
        row: sqlite3.Row,
    ) -> SessionCounterpart:
        return SessionCounterpart(
            session_id=row["session_id"],
            counterpart_id=row["counterpart_id"],
            source_metadata=_loads_dict(row["source_metadata"]),
            created_at=row["created_at"],
        )

    def _validate_session_message_shape(
        self,
        *,
        kind: SessionMessageKind,
        llm_role: LLMRole | None,
    ) -> None:
        if kind == "compressed_message":
            if llm_role != "user":
                raise ValueError("compressed_message must use llm_role='user'")
            return
        expected_roles: dict[SessionMessageKind, LLMRole] = {
            "system_reminder": "user",
            "system_message": "system",
            "user_message": "user",
            "assistant_message": "assistant",
            "tool_message": "tool",
            "compressed_message": "user",
        }
        if llm_role != expected_roles[kind]:
            raise ValueError(f"{kind} must use llm_role={expected_roles[kind]!r}")
