"""Expire active strategy overrides whose validity window has closed."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import emit_projected, report
from alpha_agent.cognition.models import CognitiveEventKind, Instant
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import StrategyProjection
from alpha_agent.utils.time import utc_now_iso


class ExpireStrategiesWorker:
    name: ClassVar[str] = "expire_strategies"
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(hours=1),
        max_interval=timedelta(hours=1),
        watches=frozenset(),
        min_new_events=0,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset()

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        del log
        projection = projections.get_typed(StrategyProjection)
        now = Instant(str(getattr(config, "now", "") or utc_now_iso()))
        due = projection.expire_due(now)
        emitted = 0
        for strategy in due:
            event = emit_projected(
                emitter,
                projections,
                CognitiveEventKind.STRATEGY_EXPIRED,
                config=config,
                payload={
                    "strategy_id": str(strategy.id),
                    "reason": "valid_until reached",
                },
                rationale="Expired strategy override whose validity window closed.",
            )
            if event is not None or getattr(config, "dry_run", False):
                emitted += 1
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(due),
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_strategy_id": str(strategy.id)},
                )
        return report(self.name, checkpoint, inspected=len(due), emitted=emitted, metadata={})
