"""Scheduler-compatible deterministic L3 self-model reflector."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, ClassVar, cast

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import report
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    Instant,
    LoopPriority,
    NLStatement,
    SelfModel,
    subject_ref,
)
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.reflectors.l3_aggregators import (
    AggregationWindow,
    SelfModelAggregator,
    default_aggregators,
)
from alpha_agent.utils.time import utc_now_iso

EMIT_THROTTLE = timedelta(hours=12)


class ReflectorL3:
    """Aggregate long-window runtime data into Subject.SelfModel."""

    name: ClassVar[str] = "reflector_l3"
    priority: ClassVar[LoopPriority] = LoopPriority.L3
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(hours=24),
        max_interval=None,
        watches=frozenset(
            {
                CognitiveEventKind.REFLECTED,
                CognitiveEventKind.BIAS_DETECTED,
                CognitiveEventKind.STRATEGY_CHANGED,
                CognitiveEventKind.BELIEF_SUPERSEDED,
                CognitiveEventKind.PROCEDURE_LEARNED,
                CognitiveEventKind.VALUE_LENS_SHIFTED,
            }
        ),
        min_new_events=50,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = trigger.watches

    def __init__(self, aggregators: list[SelfModelAggregator] | None = None):
        self.aggregators = list(aggregators or default_aggregators())

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        now = Instant(getattr(config, "now", None) or utc_now_iso())
        if self._throttled(log, now):
            return report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                notes=["throttled"],
            )
        subject = projections.get_typed(SubjectProjection).current()
        before = subject.self_model
        window = AggregationWindow(
            since=Instant(str(_parse_time(now) - timedelta(days=30))),
            until=now,
        )
        updates: dict[str, object] = {}
        aggregators_run: list[str] = []
        for aggregator in self.aggregators:
            updates[aggregator.field_name] = aggregator.compute(
                subject_ref(subject.id),
                log,
                projections,
                window,
            )
            aggregators_run.append(aggregator.field_name)
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(aggregators_run),
                    emitted=0,
                    yielded=True,
                    metadata={},
                )

        after = replace(before, **cast(Any, updates))
        if before == after:
            return report(
                self.name,
                checkpoint,
                inspected=len(aggregators_run),
                emitted=0,
                notes=["unchanged"],
            )

        diff = _diff(before, after)
        payload = {
            "before": before.to_record(),
            "after": after.to_record(),
            "diff": diff,
            "window": {"since": str(window.since), "until": str(window.until)},
            "aggregators_run": aggregators_run,
            "subject": SubjectProjection.subject_with_self_model(subject, after).to_record(),
        }
        event = _emit_projected(
            emitter=emitter,
            projections=projections,
            config=config,
            payload=payload,
            rationale="L3 updated the subject self-model from deterministic aggregators.",
            timestamp=now,
        )
        return report(
            self.name,
            checkpoint,
            inspected=len(aggregators_run),
            emitted=0 if event is None else 1,
            last_event=event,
        )

    def run_once(
        self,
        *,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter | None = None,
        coordinator: YieldingCoordinator | None = None,
        config: object | None = None,
    ) -> WorkerReport:
        class _NoYieldCoordinator:
            def yield_to_higher_priority(self) -> bool:
                return False

        return self.run(
            log,
            projections,
            emitter or EventEmitter(log),
            coordinator or _NoYieldCoordinator(),
            config or object(),
            WorkerCheckpoint(worker_name=self.name),
        )

    def _throttled(self, log: EventLog, now: Instant) -> bool:
        latest = None
        for event in log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]):
            latest = event
        if latest is None:
            return False
        return _parse_time(now) - _parse_time(latest.timestamp) < EMIT_THROTTLE


def _diff(before: SelfModel, after: SelfModel) -> dict[str, object]:
    before_record = before.to_record()
    after_record = after.to_record()
    return {
        key: {"before": before_record.get(key), "after": after_record.get(key)}
        for key in sorted(after_record)
        if before_record.get(key) != after_record.get(key)
    }


def _parse_time(value: Instant) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _emit_projected(
    *,
    emitter: EventEmitter,
    projections: ProjectionRegistry,
    config: object,
    payload: dict[str, object],
    rationale: str,
    timestamp: Instant,
) -> CognitiveEvent | None:
    if bool(getattr(config, "dry_run", False)):
        return None
    event = emitter.emit(
        CognitiveEventKind.SELF_MODEL_UPDATED,
        rationale=NLStatement(rationale),
        payload=payload,
        timestamp=timestamp,
    )
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event
