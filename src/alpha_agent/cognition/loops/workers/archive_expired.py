"""Archive active beliefs whose validity window has expired."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import after_cursor_wrap, report
from alpha_agent.cognition.models import BeliefLifecycle, CognitiveEventKind, Instant
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class ArchiveExpiredWorker:
    name: ClassVar[str] = "archive_expired"
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(hours=6),
        max_interval=timedelta(hours=6),
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
        del log, emitter
        now = datetime.now(UTC)
        projection = projections.get_typed(BeliefProjection)
        active = sorted(projection.list_active(), key=lambda item: str(item.id))
        pending = after_cursor_wrap(
            active,
            str(checkpoint.metadata.get("last_belief_id", "")),
            lambda item: item.id,
        )
        emitted = 0
        for belief in pending:
            valid_until = _valid_until(belief.validity.valid_until)
            if valid_until is not None and valid_until < now:
                if not bool(getattr(config, "dry_run", False)):
                    projection.mark_lifecycle(
                        belief.id,
                        BeliefLifecycle.ARCHIVED,
                        at=datetime.now(UTC).isoformat(),
                    )
                emitted += 1
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(active),
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_belief_id": str(belief.id)},
                )
        return report(self.name, checkpoint, inspected=len(active), emitted=emitted, metadata={})


def _valid_until(raw: Instant | str | None) -> datetime | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed
