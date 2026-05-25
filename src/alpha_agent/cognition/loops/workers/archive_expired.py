"""Archive active beliefs whose validity window has expired."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import ScheduleTrigger, WorkerCheckpoint, WorkerReport
from alpha_agent.cognition.loops.workers._common import after_cursor_wrap, emit_projected, report
from alpha_agent.cognition.models import CognitiveEventKind
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
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.BELIEF_ARCHIVED}
    )

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: object,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        del log
        now = datetime.now(UTC)
        active = sorted(
            projections.get_typed(BeliefProjection).list_active(),
            key=lambda item: str(item.id),
        )
        pending = after_cursor_wrap(
            active,
            str(checkpoint.metadata.get("last_belief_id", "")),
            lambda item: item.id,
        )
        emitted = 0
        for belief in pending:
            valid_until = _valid_until(str(belief.applicability))
            if valid_until is not None and valid_until < now:
                event = emit_projected(
                    emitter,
                    projections,
                    CognitiveEventKind.BELIEF_ARCHIVED,
                    config=config,
                    payload={"belief_id": str(belief.id), "reason": "valid_until_expired"},
                    rationale="Archived expired belief.",
                )
                emitted += 1 if event is not None or getattr(config, "dry_run", False) else 0
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


def _valid_until(raw: str) -> datetime | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not parsed.get("valid_until"):
        return None
    return datetime.fromisoformat(str(parsed["valid_until"]).replace("Z", "+00:00"))
