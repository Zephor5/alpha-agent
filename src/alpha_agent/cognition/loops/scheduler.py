"""Shared in-process scheduler primitives for cognition loops."""

from __future__ import annotations

import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Literal, Protocol

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import Instant
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore

WorkerStatus = Literal["ok", "yielded", "skipped_no_backlog", "error"]


@dataclass(frozen=True)
class WorkerCheckpoint:
    worker_name: str
    last_run_at: Instant | None = None
    last_status: WorkerStatus = "ok"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerReport:
    worker: str
    inspected: int
    emitted: int
    notes: list[str]
    yielded_to_higher_priority: bool
    new_checkpoint: WorkerCheckpoint


class ScheduledWorker(Protocol):
    @property
    def name(self) -> str: ...

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport: ...


class YieldingCoordinator(Protocol):
    """Cooperative yield and budget surface passed to scheduled workers.

    Workers must check this surface inside loops and before starting blocking
    operations such as LLM calls; the scheduler cannot hard-kill arbitrary
    Python code once an uncooperative operation has started.
    """

    def yield_to_higher_priority(self) -> bool: ...

    def budget_exhausted(self) -> bool: ...

    def remaining_seconds(self) -> float: ...


class AcquiringCoordinator(YieldingCoordinator, Protocol):
    def acquire(self, req: LoopAcquireRequest) -> AbstractContextManager[None]: ...


class CheckpointStore:
    """SQLite persistence for worker progress."""

    def __init__(self, store: StateStore):
        self.store = store
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cognition_worker_checkpoint (
                    worker_name TEXT PRIMARY KEY,
                    last_run_at TEXT,
                    last_status TEXT NOT NULL DEFAULT 'ok',
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

    def load(self, worker_name: str) -> WorkerCheckpoint:
        with self.store.connect() as conn:
            row = conn.execute(
                "SELECT * FROM cognition_worker_checkpoint WHERE worker_name = ?",
                (worker_name,),
            ).fetchone()
        if row is None:
            return WorkerCheckpoint(worker_name=worker_name)
        metadata = json.loads(row["metadata"] or "{}")
        return WorkerCheckpoint(
            worker_name=worker_name,
            last_run_at=Instant(row["last_run_at"]) if row["last_run_at"] else None,
            last_status=row["last_status"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def save(self, checkpoint: WorkerCheckpoint) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO cognition_worker_checkpoint
                    (worker_name, last_run_at, last_status, metadata)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(worker_name) DO UPDATE SET
                    last_run_at = excluded.last_run_at,
                    last_status = excluded.last_status,
                    metadata = excluded.metadata
                """,
                (
                    checkpoint.worker_name,
                    str(checkpoint.last_run_at) if checkpoint.last_run_at is not None else None,
                    checkpoint.last_status,
                    json.dumps(checkpoint.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
