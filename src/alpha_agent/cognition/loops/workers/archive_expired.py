"""Archive active beliefs whose validity window has expired."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import report
from alpha_agent.cognition.models import BeliefLifecycle, Instant
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.state_service import CognitionStateStore


class ArchiveExpiredWorker:
    name: ClassVar[str] = "archive_expired"

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        del log, emitter, coordinator, config
        now = datetime.now(UTC)
        projection = projections.get_typed(BeliefProjection)
        state_service = CognitionStateStore(projection.store)
        active = sorted(projection.list_active(), key=lambda item: str(item.id))
        emitted = 0
        for belief in active:
            valid_until = _valid_until(belief.validity.valid_until)
            if valid_until is not None and valid_until < now:
                state_service.mark_belief_lifecycle(
                    belief.id,
                    BeliefLifecycle.ARCHIVED,
                    at=datetime.now(UTC).isoformat(),
                    audit={
                        "kind": "archive_expired_lifecycle_mark",
                        "payload": {"operation": "archive_expired"},
                    },
                )
                emitted += 1
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
