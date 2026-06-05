"""Shared in-process scheduler primitives for cognition loops."""

from __future__ import annotations

import json
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEventKind, EventId, Instant, LoopPriority
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore

WorkerStatus = Literal["ok", "yielded", "skipped_no_backlog", "error"]


@dataclass(frozen=True)
class ScheduleTrigger:
    min_interval: timedelta
    max_interval: timedelta | None
    watches: frozenset[CognitiveEventKind]
    min_new_events: int = 1


@dataclass(frozen=True)
class WorkerCheckpoint:
    worker_name: str
    last_run_at: Instant | None = None
    last_processed_event_id: EventId | None = None
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

    @property
    def trigger(self) -> ScheduleTrigger: ...

    @property
    def handles_event_kinds(self) -> frozenset[CognitiveEventKind]: ...

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
                    last_processed_event_id TEXT,
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
            last_processed_event_id=EventId(row["last_processed_event_id"])
            if row["last_processed_event_id"]
            else None,
            last_status=row["last_status"],
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    def save(self, checkpoint: WorkerCheckpoint) -> None:
        with self.store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO cognition_worker_checkpoint
                    (worker_name, last_run_at, last_processed_event_id, last_status, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(worker_name) DO UPDATE SET
                    last_run_at = excluded.last_run_at,
                    last_processed_event_id = excluded.last_processed_event_id,
                    last_status = excluded.last_status,
                    metadata = excluded.metadata
                """,
                (
                    checkpoint.worker_name,
                    str(checkpoint.last_run_at) if checkpoint.last_run_at is not None else None,
                    str(checkpoint.last_processed_event_id)
                    if checkpoint.last_processed_event_id is not None
                    else None,
                    checkpoint.last_status,
                    json.dumps(checkpoint.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, WorkerCheckpoint] = {}

    def load(self, worker_name: str) -> WorkerCheckpoint:
        return self._items.get(worker_name, WorkerCheckpoint(worker_name=worker_name))

    def save(self, checkpoint: WorkerCheckpoint) -> None:
        self._items[checkpoint.worker_name] = checkpoint


class Scheduler:
    """Shared scheduler with time and backlog gates."""

    def __init__(self, log: EventLog, checkpoints: CheckpointStore | InMemoryCheckpointStore):
        self.log = log
        self.checkpoints = checkpoints
        self._workers: dict[str, ScheduledWorker] = {}
        self._triggers: dict[str, ScheduleTrigger] = {}

    def register(self, worker: ScheduledWorker, trigger: ScheduleTrigger | None = None) -> None:
        self._workers[worker.name] = worker
        self._triggers[worker.name] = trigger or worker.trigger

    def workers(self) -> tuple[ScheduledWorker, ...]:
        return tuple(self._workers.values())

    def should_run(self, worker: ScheduledWorker, now: Instant) -> bool:
        trigger = self._triggers[worker.name]
        checkpoint = self.checkpoints.load(worker.name)
        if checkpoint.last_status == "yielded":
            return True
        if checkpoint.last_run_at is not None:
            elapsed = _parse_instant(now) - _parse_instant(checkpoint.last_run_at)
            if elapsed < trigger.min_interval:
                return False
            if trigger.max_interval is not None and elapsed >= trigger.max_interval:
                return True
        elif trigger.min_interval == trigger.max_interval and not trigger.watches:
            return True

        if not trigger.watches:
            return trigger.min_interval == trigger.max_interval
        return (
            _count_events_after(
                self.log.iter(kinds=trigger.watches),
                checkpoint.last_processed_event_id,
            )
            >= trigger.min_new_events
        )

    def save_report(self, report: WorkerReport) -> None:
        self.checkpoints.save(report.new_checkpoint)

    def tick(
        self,
        now: Instant,
        *,
        coordinator: AcquiringCoordinator,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        config: object,
        loop_name: str = "consolidation",
        priority: LoopPriority = LoopPriority.CONSOLIDATION,
        max_chunk_duration: timedelta = timedelta(seconds=30),
    ) -> list[WorkerReport]:
        return self.run_registered(
            now,
            coordinator=coordinator,
            projections=projections,
            emitter=emitter,
            config=config,
            force=False,
            loop_name=loop_name,
            priority=priority,
            max_chunk_duration=max_chunk_duration,
        )

    def run_registered(
        self,
        now: Instant,
        *,
        coordinator: AcquiringCoordinator,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        config: object,
        force: bool,
        loop_name: str = "consolidation",
        priority: LoopPriority = LoopPriority.CONSOLIDATION,
        max_chunk_duration: timedelta = timedelta(seconds=30),
    ) -> list[WorkerReport]:
        reports: list[WorkerReport] = []
        for worker in self.workers():
            if not force and not self.should_run(worker, now):
                continue
            req = LoopAcquireRequest(
                loop_name=loop_name,
                priority=getattr(worker, "priority", priority),
                max_chunk_duration=max_chunk_duration,
            )
            with coordinator.acquire(req):
                checkpoint = self.checkpoints.load(worker.name)
                report = worker.run(
                    self.log,
                    projections,
                    emitter,
                    coordinator,
                    config,
                    checkpoint,
                )
                report = self._stamp_report(report, worker, now)
                self.save_report(report)
                reports.append(report)
                if report.yielded_to_higher_priority:
                    break
                if coordinator.yield_to_higher_priority():
                    break
        return reports

    def _stamp_report(
        self,
        report: WorkerReport,
        worker: ScheduledWorker,
        now: Instant,
    ) -> WorkerReport:
        checkpoint = report.new_checkpoint
        last_processed = checkpoint.last_processed_event_id
        if not report.yielded_to_higher_priority:
            last_processed = _latest_event_id(self.log.iter(kinds=worker.trigger.watches))
        return WorkerReport(
            worker=report.worker,
            inspected=report.inspected,
            emitted=report.emitted,
            notes=report.notes,
            yielded_to_higher_priority=report.yielded_to_higher_priority,
            new_checkpoint=WorkerCheckpoint(
                worker_name=checkpoint.worker_name,
                last_run_at=now,
                last_processed_event_id=last_processed,
                last_status=checkpoint.last_status,
                metadata=checkpoint.metadata,
            ),
        )


def _count_events_after(events: Iterable[Any], event_id: EventId | None) -> int:
    count = 0
    seen_checkpoint = event_id is None
    for event in events:
        current_id = event.id
        if seen_checkpoint:
            count += 1
        elif str(current_id) == str(event_id):
            seen_checkpoint = True
    return count


def _latest_event_id(events: Iterable[Any]) -> EventId | None:
    latest = None
    for event in events:
        latest = event.id
    return latest



def _parse_instant(value: Instant) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
