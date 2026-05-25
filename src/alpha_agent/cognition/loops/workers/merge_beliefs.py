"""Merge equivalent active beliefs."""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import ScheduleTrigger, WorkerCheckpoint, WorkerReport
from alpha_agent.cognition.loops.workers._common import (
    after_cursor_wrap,
    emit_projected,
    normalize_text,
    report,
    trigger,
)
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class MergeBeliefsWorker:
    name: ClassVar[str] = "merge_beliefs"
    trigger: ClassVar[ScheduleTrigger] = trigger(
        30, 24, {CognitiveEventKind.BELIEF_FORMED}, 5
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.BELIEF_FORMED, CognitiveEventKind.BELIEF_SUPERSEDED}
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
        projection = projections.get_typed(BeliefProjection)
        groups = defaultdict(list)
        active = projection.list_active()
        for belief in active:
            about_key = tuple(sorted((ref.kind, ref.id) for ref in belief.about))
            key = (
                about_key,
                belief.object,
                belief.cognitive_type.value,
                normalize_text(belief.content),
            )
            groups[key].append(belief)

        emitted = 0
        inspected = len(active)
        pending_groups = after_cursor_wrap(
            sorted(groups.items(), key=lambda item: _group_cursor(item[0])),
            str(checkpoint.metadata.get("last_merge_key", "")),
            lambda item: _group_cursor(item[0]),
        )
        for key, beliefs in pending_groups:
            if len(beliefs) >= 2:
                keeper = max(
                    beliefs,
                    key=lambda item: (item.confidence, str(item.held_since), str(item.id)),
                )
                for old in sorted(beliefs, key=lambda item: str(item.id)):
                    if old.id == keeper.id:
                        continue
                    event = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.BELIEF_SUPERSEDED,
                        config=config,
                        payload={
                            "old_belief_id": str(old.id),
                            "new_belief_id": str(keeper.id),
                            "reason": "equivalent_active_belief",
                        },
                        rationale="Merged equivalent active beliefs.",
                    )
                    emitted += (
                        1 if event is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=inspected,
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_merge_key": _group_cursor(key)},
                )
        return report(self.name, checkpoint, inspected=inspected, emitted=emitted, metadata={})


def _group_cursor(key: object) -> str:
    return repr(key)
