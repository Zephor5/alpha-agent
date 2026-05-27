"""Resolve queued belief conflicts with the current subject value lens."""

from __future__ import annotations

from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import (
    after_cursor_wrap,
    emit_projected,
    report,
    trigger,
)
from alpha_agent.cognition.models import (
    BeliefId,
    CognitiveEvent,
    CognitiveEventKind,
    belief_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.models.value import ValueLens
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.value.lens import default_value_lens, lens_to_record, load_lens
from alpha_agent.cognition.value.resolver import resolve_conflict


class ResolveQueuedConflictsWorker:
    name: ClassVar[str] = "resolve_queued_conflicts"
    trigger: ClassVar[ScheduleTrigger] = trigger(
        5,
        6,
        {CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED},
        1,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED}
    )

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        belief_projection = projections.get_typed(BeliefProjection)
        conflicts = sorted(
            log.iter(kinds=[CognitiveEventKind.CONSOLIDATION_CONFLICT_QUEUED]),
            key=lambda event: str(event.id),
        )
        conflicts = after_cursor_wrap(
            conflicts,
            str(checkpoint.metadata.get("last_conflict_event_id", "")),
            lambda event: event.id,
        )
        lens = _current_lens(log)
        emitted = 0
        for event in conflicts:
            if _already_handled(log, event):
                if coordinator.yield_to_higher_priority():
                    return report(
                        self.name,
                        checkpoint,
                        inspected=len(conflicts),
                        emitted=emitted,
                        yielded=True,
                        metadata={"last_conflict_event_id": str(event.id)},
                    )
                continue
            belief_ids = _belief_ids(event)
            if len(belief_ids) != 2:
                emitted += _emit_human_review(
                    emitter,
                    projections,
                    config,
                    event,
                    belief_ids,
                    "queued conflict must reference exactly two beliefs",
                )
                continue
            left = belief_projection.get_by_id(belief_ids[0])
            right = belief_projection.get_by_id(belief_ids[1])
            if left is None or right is None:
                emitted += _emit_human_review(
                    emitter,
                    projections,
                    config,
                    event,
                    belief_ids,
                    "queued conflict references a missing belief",
                )
            else:
                resolution = resolve_conflict(left, right, lens)
                if resolution.tie:
                    emitted += _emit_human_review(
                        emitter,
                        projections,
                        config,
                        event,
                        belief_ids,
                        "tie under current value lens",
                    )
                else:
                    superseded = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.BELIEF_SUPERSEDED,
                        config=config,
                        inputs=[belief_ref(BeliefId(item)) for item in belief_ids],
                        payload={
                            "old_belief_id": str(resolution.loser_id),
                            "new_belief_id": str(resolution.winner_id),
                            "reason": "value_lens_conflict_resolution",
                            "conflict_event_id": str(event.id),
                            "decisive_value_kinds": [
                                value.value for value in resolution.by_lens_priority
                            ],
                            "value_lens_explanation": resolution.rationale,
                            "value_lens": lens_to_record(lens),
                            "resolution_margin": resolution.margin,
                        },
                        rationale="Resolved queued conflict using the subject value lens.",
                    )
                    emitted += (
                        1 if superseded is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(conflicts),
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_conflict_event_id": str(event.id)},
                )
        return report(self.name, checkpoint, inspected=len(conflicts), emitted=emitted, metadata={})


def _current_lens(log: EventLog) -> ValueLens:
    store = getattr(log, "store", None)
    return load_lens(store, str(SUBJECT_SELF)) if store is not None else default_value_lens()


def _belief_ids(event: CognitiveEvent) -> list[str]:
    raw = event.payload.get("belief_ids")
    if isinstance(raw, list):
        return [str(item) for item in raw if item is not None]
    left = event.payload.get("left_belief_id")
    right = event.payload.get("right_belief_id")
    return [str(item) for item in (left, right) if item is not None]


def _already_handled(log: EventLog, event: CognitiveEvent) -> bool:
    handled_kinds = {
        CognitiveEventKind.BELIEF_SUPERSEDED,
        CognitiveEventKind.CONFLICT_KEPT_FOR_HUMAN_REVIEW,
    }
    for candidate in log.iter(kinds=handled_kinds):
        if str(candidate.payload.get("conflict_event_id")) == str(event.id):
            return True
    return False


def _emit_human_review(
    emitter: EventEmitter,
    projections: ProjectionRegistry,
    config: object,
    event: CognitiveEvent,
    belief_ids: list[str],
    reason: str,
) -> int:
    emitted = emit_projected(
        emitter,
        projections,
        CognitiveEventKind.CONFLICT_KEPT_FOR_HUMAN_REVIEW,
        config=config,
        inputs=[belief_ref(BeliefId(item)) for item in belief_ids],
        payload={
            "belief_ids": belief_ids,
            "conflict_event_id": str(event.id),
            "reason": reason,
        },
        rationale="Kept queued belief conflict for human review.",
    )
    return 1 if emitted is not None or getattr(config, "dry_run", False) else 0
