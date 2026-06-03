"""Synchronous v1 consolidation loop."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.cognition.coordinator import LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    AcquiringCoordinator,
    InMemoryCheckpointStore,
    ScheduledWorker,
    Scheduler,
    WorkerReport,
)
from alpha_agent.cognition.loops.workers import default_workers
from alpha_agent.cognition.models import Instant
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class ConsolidationConfig:
    enabled: bool = True
    interval_seconds: int = 300
    context_foreground_max: int = 8
    context_absorb_batch: int = 4
    context_summary_chars: int = 480
    counterpart_digest_min_beliefs: int = 5
    counterpart_digest_min_new_beliefs: int = 3
    value_lens_learning_threshold: int = 5
    value_lens_sensitivity_delta: float = 0.1
    dry_run: bool = False


class ConsolidationLoop:
    """Run registered consolidation workers once inside the cooperative loop lock."""

    def __init__(
        self,
        *,
        scheduler: Scheduler | None = None,
        coordinator: AcquiringCoordinator | None = None,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter | None = None,
        config: ConsolidationConfig | None = None,
        workers: list[ScheduledWorker] | None = None,
    ):
        self.log = log
        self.projections = projections
        self.coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
        self.config = config or ConsolidationConfig()
        self.emitter = emitter or EventEmitter(log)
        self.scheduler = scheduler or Scheduler(log, InMemoryCheckpointStore())
        self._workers = list(workers or default_workers())

    def register_all_workers(self) -> None:
        for worker in self._workers:
            self.scheduler.register(worker, worker.trigger)

    def run_once(self) -> list[WorkerReport]:
        if not self.config.enabled:
            return []
        if not self.scheduler.workers():
            self.register_all_workers()

        return self.scheduler.run_registered(
            Instant(utc_now_iso()),
            coordinator=self.coordinator,
            projections=self.projections,
            emitter=self.emitter,
            config=self.config,
            force=True,
        )
