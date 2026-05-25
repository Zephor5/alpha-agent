"""Learn deterministic procedures from repeated successful decisions."""

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
    stable_id,
    trigger,
)
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    ExpectedFeedback,
    Procedure,
    ProcedureId,
    Step,
    TriggerPattern,
)
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class LearnProcedureWorker:
    name: ClassVar[str] = "learn_procedure"
    trigger: ClassVar[ScheduleTrigger] = trigger(
        30,
        24,
        {CognitiveEventKind.DECIDED, CognitiveEventKind.RECEIVED_FEEDBACK},
        10,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.DECIDED, CognitiveEventKind.RECEIVED_FEEDBACK}
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
        decisions = {
            str(event.payload.get("tick_id")): event
            for event in log.iter(kinds=[CognitiveEventKind.DECIDED])
            if event.payload.get("tick_id")
        }
        successes = []
        for event in log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK]):
            tick_id = str(event.payload.get("tick_id"))
            if event.payload.get("matched_expected") is True and tick_id in decisions:
                successes.append((decisions[tick_id], event))

        groups = defaultdict(list)
        for decision, feedback in successes:
            pattern = _pattern(decision.payload)
            if pattern:
                groups[pattern].append((decision, feedback))

        projection = projections.get_typed(ProcedureProjection)
        threshold = max(1, int(getattr(config, "procedure_success_threshold", 3)))
        emitted = 0
        pending_groups = after_cursor_wrap(
            sorted(groups.items()),
            str(checkpoint.metadata.get("last_pattern", "")),
            lambda item: item[0],
        )
        for pattern, items in pending_groups:
            if len(items) >= threshold:
                procedure_id = stable_id("procedure", pattern)
                if projection.get(ProcedureId(procedure_id)) is None:
                    learned_from = [decision.id for decision, _ in items]
                    procedure = Procedure(
                        id=ProcedureId(procedure_id),
                        trigger=TriggerPattern(pattern),
                        steps=[Step(f"repeat action pattern: {pattern}")],
                        expected_outcome=ExpectedFeedback("matched_expected_feedback"),
                        learned_from=learned_from,
                        success_count=len(items),
                        failure_count=0,
                        confidence=min(0.95, 0.5 + 0.1 * len(items)),
                    )
                    event = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.PROCEDURE_LEARNED,
                        config=config,
                        payload={
                            "procedure": procedure.to_record(),
                            "name": f"Repeat {pattern}",
                            "learned_from_event_ids": [str(item) for item in learned_from],
                        },
                        rationale=(
                            "Learned deterministic procedure from repeated successful decisions."
                        ),
                    )
                    emitted += (
                        1 if event is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(successes),
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_pattern": pattern},
                )
        latest = successes[-1][1] if successes else None
        return report(
            self.name,
            checkpoint,
            inspected=len(successes),
            emitted=emitted,
            last_event=latest,
            metadata={},
        )


def _pattern(payload: object) -> str:
    if not isinstance(payload, dict):
        return ""
    action = normalize_text(payload.get("action", ""))
    message = normalize_text(payload.get("message", ""))
    if action and message:
        return f"{action}:{message[:80]}"
    return action
