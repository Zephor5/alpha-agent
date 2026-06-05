"""Sidecar processing ledger for background cognition stages."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeVar

from alpha_agent.state.store import StateStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

T = TypeVar("T")


class BackgroundStage(StrEnum):
    """Background cognition processing stages."""

    INTAKE = "intake"
    EXTRACTION = "extraction"
    CONSOLIDATION = "consolidation"
    CONFLICT_REVIEW = "conflict_review"
    SUMMARY = "summary"


class BackgroundProgressStatus(StrEnum):
    """Per-source and source-window processing status."""

    PENDING = "pending"
    CLAIMED = "claimed"
    PROCESSED = "processed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BackgroundStageRunStatus(StrEnum):
    """One worker attempt status."""

    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    YIELDED = "yielded"
    SKIPPED = "skipped"


_SOURCE_TYPES = frozenset(
    {
        "session_message",
        "runtime_trace",
        "tool_trace",
        "atomic_belief",
        "summary_belief",
        "conflict",
    }
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS background_source_progress (
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    claimed_by TEXT,
    claimed_at TEXT,
    processed_at TEXT,
    checkpoint_id TEXT,
    idempotency_key TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(source_type, source_id, stage, target_unit)
);

CREATE INDEX IF NOT EXISTS idx_background_source_progress_status
    ON background_source_progress(stage, target_unit, status, updated_at);
CREATE INDEX IF NOT EXISTS idx_background_source_progress_idempotency
    ON background_source_progress(idempotency_key);

CREATE TABLE IF NOT EXISTS background_source_window (
    window_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    source_refs TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    closed_at TEXT,
    status TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    claimed_by TEXT,
    claimed_at TEXT,
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_source_window_status
    ON background_source_window(stage, target_unit, status, created_at);

CREATE TABLE IF NOT EXISTS background_stage_run (
    run_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    target_unit TEXT NOT NULL,
    window_id TEXT,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    input_refs TEXT NOT NULL DEFAULT '[]',
    output_refs TEXT NOT NULL DEFAULT '[]',
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_background_stage_run_window
    ON background_stage_run(window_id, started_at);
CREATE INDEX IF NOT EXISTS idx_background_stage_run_status
    ON background_stage_run(stage, target_unit, status, started_at);
"""


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default


@dataclass(frozen=True, order=True)
class BackgroundSourceRef:
    """Exact source unit consumed by a background stage."""

    source_type: str
    source_id: str

    def __post_init__(self) -> None:
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError(f"unsupported background source_type: {self.source_type}")
        if not self.source_id.strip():
            raise ValueError("background source_id must be non-empty")

    def to_record(self) -> dict[str, str]:
        return {"source_type": self.source_type, "source_id": self.source_id}

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> BackgroundSourceRef:
        return cls(source_type=str(record["source_type"]), source_id=str(record["source_id"]))


@dataclass(frozen=True)
class BackgroundSourceProgress:
    """Stage-specific processing status for one source."""

    source_ref: BackgroundSourceRef
    stage: BackgroundStage
    target_unit: str
    status: BackgroundProgressStatus
    attempts: int
    idempotency_key: str
    last_error: str | None = None
    claimed_by: str | None = None
    claimed_at: str | None = None
    processed_at: str | None = None
    checkpoint_id: str | None = None
    updated_at: str = ""


@dataclass(frozen=True)
class BackgroundSourceWindow:
    """Exact source set selected for one background LLM call."""

    window_id: str
    stage: BackgroundStage
    target_unit: str
    source_refs: tuple[BackgroundSourceRef, ...]
    status: BackgroundProgressStatus
    idempotency_key: str
    created_at: str
    closed_at: str | None = None
    claimed_by: str | None = None
    claimed_at: str | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class BackgroundStageRun:
    """One worker attempt over an optional source window."""

    run_id: str
    worker_id: str
    stage: BackgroundStage
    target_unit: str
    status: BackgroundStageRunStatus
    started_at: str
    window_id: str | None = None
    finished_at: str | None = None
    input_refs: tuple[BackgroundSourceRef, ...] = field(default_factory=tuple)
    output_refs: tuple[BackgroundSourceRef, ...] = field(default_factory=tuple)
    error: str | None = None


class ProcessingLedger:
    """SQLite-backed sidecar ledger for background cognition processing."""

    def __init__(self, store: StateStore):
        self.store = store
        self.store.initialize()
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_SCHEMA)

    def mark_source_pending(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        idempotency_key: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        def op(db: sqlite3.Connection) -> BackgroundSourceProgress:
            self._upsert_source_progress(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
                status=BackgroundProgressStatus.PENDING,
                idempotency_key=idempotency_key,
            )
            return self._get_source_progress_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
            )

        return self._write(conn, op)

    def claim_source(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        claimed_by: str,
        idempotency_key: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        def op(db: sqlite3.Connection) -> BackgroundSourceProgress:
            now = utc_now_iso()
            self._ensure_source_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
                idempotency_key=idempotency_key,
            )
            db.execute(
                """
                UPDATE background_source_progress
                SET status = ?,
                    attempts = attempts + 1,
                    claimed_by = ?,
                    claimed_at = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE source_type = ? AND source_id = ? AND stage = ? AND target_unit = ?
                """,
                (
                    BackgroundProgressStatus.CLAIMED.value,
                    claimed_by,
                    now,
                    now,
                    source_ref.source_type,
                    source_ref.source_id,
                    BackgroundStage(stage).value,
                    target_unit,
                ),
            )
            return self._get_source_progress_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
            )

        return self._write(conn, op)

    def mark_source_processed(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        checkpoint_id: str | None = None,
        idempotency_key: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        def op(db: sqlite3.Connection) -> BackgroundSourceProgress:
            now = utc_now_iso()
            self._ensure_source_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
                idempotency_key=idempotency_key,
            )
            db.execute(
                """
                UPDATE background_source_progress
                SET status = ?,
                    last_error = NULL,
                    processed_at = ?,
                    checkpoint_id = ?,
                    updated_at = ?
                WHERE source_type = ? AND source_id = ? AND stage = ? AND target_unit = ?
                """,
                (
                    BackgroundProgressStatus.PROCESSED.value,
                    now,
                    checkpoint_id,
                    now,
                    source_ref.source_type,
                    source_ref.source_id,
                    BackgroundStage(stage).value,
                    target_unit,
                ),
            )
            return self._get_source_progress_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
            )

        return self._write(conn, op)

    def mark_source_failed(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        error: str,
        idempotency_key: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        return self._mark_source_terminal(
            source_ref,
            stage=stage,
            target_unit=target_unit,
            status=BackgroundProgressStatus.FAILED,
            reason=error,
            idempotency_key=idempotency_key,
            conn=conn,
        )

    def mark_source_skipped(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        reason: str,
        idempotency_key: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        return self._mark_source_terminal(
            source_ref,
            stage=stage,
            target_unit=target_unit,
            status=BackgroundProgressStatus.SKIPPED,
            reason=reason,
            idempotency_key=idempotency_key,
            conn=conn,
        )

    def get_source_progress(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceProgress:
        def op(db: sqlite3.Connection) -> BackgroundSourceProgress:
            return self._get_source_progress_row(
                db,
                source_ref,
                stage=BackgroundStage(stage),
                target_unit=target_unit,
            )

        return self._read(conn, op)

    def list_source_progress(
        self,
        *,
        stage: BackgroundStage | str | None = None,
        status: BackgroundProgressStatus | str | None = None,
    ) -> list[BackgroundSourceProgress]:
        conditions: list[str] = []
        params: list[Any] = []
        if stage is not None:
            conditions.append("stage = ?")
            params.append(BackgroundStage(stage).value)
        if status is not None:
            conditions.append("status = ?")
            params.append(BackgroundProgressStatus(status).value)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        with self.store.connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM background_source_progress
                {where}
                ORDER BY updated_at ASC, source_type ASC, source_id ASC
                """,
                params,
            ).fetchall()
        return [self._progress_from_row(row) for row in rows]

    def create_source_window(
        self,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        source_refs: Sequence[BackgroundSourceRef],
        idempotency_key: str,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        if not source_refs:
            raise ValueError("source window requires at least one source ref")
        stage_value = BackgroundStage(stage)
        window_id = _stable_window_id(
            stage=stage_value,
            target_unit=target_unit,
            key=idempotency_key,
        )

        def op(db: sqlite3.Connection) -> BackgroundSourceWindow:
            now = utc_now_iso()
            db.execute(
                """
                INSERT OR IGNORE INTO background_source_window
                    (window_id, stage, target_unit, source_refs, created_at, status,
                     idempotency_key)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    window_id,
                    stage_value.value,
                    target_unit,
                    _dumps([item.to_record() for item in source_refs]),
                    now,
                    BackgroundProgressStatus.PENDING.value,
                    idempotency_key,
                ),
            )
            return self._get_source_window_row(db, window_id)

        return self._write(conn, op)

    def claim_source_window(
        self,
        window_id: str,
        *,
        claimed_by: str,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        def op(db: sqlite3.Connection) -> BackgroundSourceWindow:
            now = utc_now_iso()
            db.execute(
                """
                UPDATE background_source_window
                SET status = ?,
                    claimed_by = ?,
                    claimed_at = ?,
                    last_error = NULL
                WHERE window_id = ?
                """,
                (BackgroundProgressStatus.CLAIMED.value, claimed_by, now, window_id),
            )
            return self._get_source_window_row(db, window_id)

        return self._write(conn, op)

    def mark_source_window_processed(
        self,
        window_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        return self._mark_window_terminal(
            window_id,
            status=BackgroundProgressStatus.PROCESSED,
            reason=None,
            conn=conn,
        )

    def mark_source_window_failed(
        self,
        window_id: str,
        *,
        error: str,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        return self._mark_window_terminal(
            window_id,
            status=BackgroundProgressStatus.FAILED,
            reason=error,
            conn=conn,
        )

    def mark_source_window_skipped(
        self,
        window_id: str,
        *,
        reason: str,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        return self._mark_window_terminal(
            window_id,
            status=BackgroundProgressStatus.SKIPPED,
            reason=reason,
            conn=conn,
        )

    def get_source_window(
        self,
        window_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow:
        return self._read(conn, lambda db: self._get_source_window_row(db, window_id))

    def start_stage_run(
        self,
        *,
        worker_id: str,
        stage: BackgroundStage | str,
        target_unit: str,
        window_id: str | None,
        input_refs: Sequence[BackgroundSourceRef],
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundStageRun:
        stage_value = BackgroundStage(stage)
        run_id = new_id("bgrun")

        def op(db: sqlite3.Connection) -> BackgroundStageRun:
            db.execute(
                """
                INSERT INTO background_stage_run
                    (run_id, worker_id, stage, target_unit, window_id, status,
                     started_at, input_refs)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    worker_id,
                    stage_value.value,
                    target_unit,
                    window_id,
                    BackgroundStageRunStatus.STARTED.value,
                    utc_now_iso(),
                    _dumps([item.to_record() for item in input_refs]),
                ),
            )
            return self._get_stage_run_row(db, run_id)

        return self._write(conn, op)

    def finish_stage_run(
        self,
        run_id: str,
        *,
        status: BackgroundStageRunStatus | str,
        output_refs: Sequence[BackgroundSourceRef] = (),
        error: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundStageRun:
        status_value = BackgroundStageRunStatus(status)

        def op(db: sqlite3.Connection) -> BackgroundStageRun:
            db.execute(
                """
                UPDATE background_stage_run
                SET status = ?,
                    finished_at = ?,
                    output_refs = ?,
                    error = ?
                WHERE run_id = ?
                """,
                (
                    status_value.value,
                    utc_now_iso(),
                    _dumps([item.to_record() for item in output_refs]),
                    error,
                    run_id,
                ),
            )
            return self._get_stage_run_row(db, run_id)

        return self._write(conn, op)

    def get_stage_run(
        self,
        run_id: str,
        *,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundStageRun:
        return self._read(conn, lambda db: self._get_stage_run_row(db, run_id))

    def _mark_source_terminal(
        self,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage | str,
        target_unit: str,
        status: BackgroundProgressStatus,
        reason: str,
        idempotency_key: str | None,
        conn: sqlite3.Connection | None,
    ) -> BackgroundSourceProgress:
        def op(db: sqlite3.Connection) -> BackgroundSourceProgress:
            now = utc_now_iso()
            stage_value = BackgroundStage(stage)
            self._ensure_source_row(
                db,
                source_ref,
                stage=stage_value,
                target_unit=target_unit,
                idempotency_key=idempotency_key,
            )
            db.execute(
                """
                UPDATE background_source_progress
                SET status = ?,
                    last_error = ?,
                    updated_at = ?
                WHERE source_type = ? AND source_id = ? AND stage = ? AND target_unit = ?
                """,
                (
                    status.value,
                    reason,
                    now,
                    source_ref.source_type,
                    source_ref.source_id,
                    stage_value.value,
                    target_unit,
                ),
            )
            return self._get_source_progress_row(
                db,
                source_ref,
                stage=stage_value,
                target_unit=target_unit,
            )

        return self._write(conn, op)

    def _mark_window_terminal(
        self,
        window_id: str,
        *,
        status: BackgroundProgressStatus,
        reason: str | None,
        conn: sqlite3.Connection | None,
    ) -> BackgroundSourceWindow:
        def op(db: sqlite3.Connection) -> BackgroundSourceWindow:
            db.execute(
                """
                UPDATE background_source_window
                SET status = ?,
                    closed_at = ?,
                    last_error = ?
                WHERE window_id = ?
                """,
                (status.value, utc_now_iso(), reason, window_id),
            )
            return self._get_source_window_row(db, window_id)

        return self._write(conn, op)

    def _ensure_source_row(
        self,
        conn: sqlite3.Connection,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage,
        target_unit: str,
        idempotency_key: str | None,
    ) -> None:
        existing = conn.execute(
            """
            SELECT 1
            FROM background_source_progress
            WHERE source_type = ? AND source_id = ? AND stage = ? AND target_unit = ?
            """,
            (source_ref.source_type, source_ref.source_id, stage.value, target_unit),
        ).fetchone()
        if existing is None:
            self._upsert_source_progress(
                conn,
                source_ref,
                stage=stage,
                target_unit=target_unit,
                status=BackgroundProgressStatus.PENDING,
                idempotency_key=idempotency_key,
            )

    def _upsert_source_progress(
        self,
        conn: sqlite3.Connection,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage,
        target_unit: str,
        status: BackgroundProgressStatus,
        idempotency_key: str | None,
    ) -> None:
        now = utc_now_iso()
        key = idempotency_key or _default_source_idempotency_key(
            source_ref,
            stage=stage,
            target_unit=target_unit,
        )
        conn.execute(
            """
            INSERT INTO background_source_progress
                (source_type, source_id, stage, target_unit, status, idempotency_key,
                 updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_type, source_id, stage, target_unit) DO UPDATE SET
                status = excluded.status,
                idempotency_key = excluded.idempotency_key,
                last_error = NULL,
                updated_at = excluded.updated_at
            """,
            (
                source_ref.source_type,
                source_ref.source_id,
                stage.value,
                target_unit,
                status.value,
                key,
                now,
            ),
        )

    def _get_source_progress_row(
        self,
        conn: sqlite3.Connection,
        source_ref: BackgroundSourceRef,
        *,
        stage: BackgroundStage,
        target_unit: str,
    ) -> BackgroundSourceProgress:
        row = conn.execute(
            """
            SELECT *
            FROM background_source_progress
            WHERE source_type = ? AND source_id = ? AND stage = ? AND target_unit = ?
            """,
            (source_ref.source_type, source_ref.source_id, stage.value, target_unit),
        ).fetchone()
        if row is None:
            raise KeyError(
                "background source progress not found for "
                f"{source_ref.source_type}:{source_ref.source_id}:{stage.value}:{target_unit}"
            )
        return self._progress_from_row(row)

    def _get_source_window_row(
        self,
        conn: sqlite3.Connection,
        window_id: str,
    ) -> BackgroundSourceWindow:
        row = conn.execute(
            "SELECT * FROM background_source_window WHERE window_id = ?",
            (window_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"background source window {window_id!r} not found")
        return self._window_from_row(row)

    def _get_stage_run_row(
        self,
        conn: sqlite3.Connection,
        run_id: str,
    ) -> BackgroundStageRun:
        row = conn.execute(
            "SELECT * FROM background_stage_run WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"background stage run {run_id!r} not found")
        return self._stage_run_from_row(row)

    def _progress_from_row(self, row: sqlite3.Row) -> BackgroundSourceProgress:
        return BackgroundSourceProgress(
            source_ref=BackgroundSourceRef(row["source_type"], row["source_id"]),
            stage=BackgroundStage(row["stage"]),
            target_unit=row["target_unit"],
            status=BackgroundProgressStatus(row["status"]),
            attempts=int(row["attempts"]),
            idempotency_key=row["idempotency_key"],
            last_error=row["last_error"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            processed_at=row["processed_at"],
            checkpoint_id=row["checkpoint_id"],
            updated_at=row["updated_at"],
        )

    def _window_from_row(self, row: sqlite3.Row) -> BackgroundSourceWindow:
        return BackgroundSourceWindow(
            window_id=row["window_id"],
            stage=BackgroundStage(row["stage"]),
            target_unit=row["target_unit"],
            source_refs=_source_refs_from_json(row["source_refs"]),
            status=BackgroundProgressStatus(row["status"]),
            idempotency_key=row["idempotency_key"],
            created_at=row["created_at"],
            closed_at=row["closed_at"],
            claimed_by=row["claimed_by"],
            claimed_at=row["claimed_at"],
            last_error=row["last_error"],
        )

    def _stage_run_from_row(self, row: sqlite3.Row) -> BackgroundStageRun:
        return BackgroundStageRun(
            run_id=row["run_id"],
            worker_id=row["worker_id"],
            stage=BackgroundStage(row["stage"]),
            target_unit=row["target_unit"],
            window_id=row["window_id"],
            status=BackgroundStageRunStatus(row["status"]),
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            input_refs=_source_refs_from_json(row["input_refs"]),
            output_refs=_source_refs_from_json(row["output_refs"]),
            error=row["error"],
        )

    def _read(
        self,
        conn: sqlite3.Connection | None,
        op: Callable[[sqlite3.Connection], T],
    ) -> T:
        if conn is not None:
            return op(conn)
        with self.store.connect() as local:
            return op(local)

    def _write(
        self,
        conn: sqlite3.Connection | None,
        op: Callable[[sqlite3.Connection], T],
    ) -> T:
        if conn is not None:
            return op(conn)
        with self.store.immediate_transaction() as local:
            return op(local)


def _source_refs_from_json(value: str | None) -> tuple[BackgroundSourceRef, ...]:
    loaded = _loads(value, [])
    if not isinstance(loaded, list):
        return ()
    refs: list[BackgroundSourceRef] = []
    for item in loaded:
        if isinstance(item, dict):
            refs.append(BackgroundSourceRef.from_record(item))
    return tuple(refs)


def _stable_window_id(*, stage: BackgroundStage, target_unit: str, key: str) -> str:
    digest = hashlib.sha256(f"{stage.value}\n{target_unit}\n{key}".encode()).hexdigest()
    return f"bgwindow_{digest[:24]}"


def _default_source_idempotency_key(
    source_ref: BackgroundSourceRef,
    *,
    stage: BackgroundStage,
    target_unit: str,
) -> str:
    return f"{stage.value}:{target_unit}:{source_ref.source_type}:{source_ref.source_id}"
