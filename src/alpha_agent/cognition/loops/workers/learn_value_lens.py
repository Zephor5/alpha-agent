"""Minimal deterministic learning loop for value-lens sensitivity."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import report, trigger
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, ValueKind
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import (
    StrategyProjection,
    strategy_is_active_for_domain,
)
from alpha_agent.cognition.value.lens import load_lens, normalize_lens, save_lens


class LearnValueLensWorker:
    name: ClassVar[str] = "learn_value_lens"
    trigger: ClassVar[ScheduleTrigger] = trigger(
        60,
        24,
        {CognitiveEventKind.BELIEF_SUPERSEDED},
        5,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.BELIEF_SUPERSEDED}
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
        if _lens_learning_frozen(projections):
            return report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                notes=["lens learning frozen by active strategy"],
                metadata={},
            )
        event_log_order = list(log.iter(kinds=[CognitiveEventKind.BELIEF_SUPERSEDED]))
        post_checkpoint = _events_after_processed_checkpoint(
            event_log_order,
            str(checkpoint.last_processed_event_id or ""),
        )
        superseded = _after_event_cursor_wrap(
            post_checkpoint,
            str(checkpoint.metadata.get("last_superseded_event_id", "")),
        )
        counter: Counter[ValueKind] = Counter()
        for event in superseded:
            kinds = _decisive_kinds(event.payload.get("decisive_value_kinds"))
            if kinds:
                counter[kinds[0]] += 1
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(superseded),
                    emitted=0,
                    yielded=True,
                    metadata={"last_superseded_event_id": str(event.id)},
                )
        if not counter:
            return report(self.name, checkpoint, inspected=len(superseded), emitted=0, metadata={})

        threshold = max(1, int(getattr(config, "value_lens_learning_threshold", 5)))
        winner, count = sorted(counter.items(), key=lambda item: (-item[1], item[0].value))[0]
        now = str(post_checkpoint[-1].timestamp) if post_checkpoint else None
        if count < threshold or _recent_shift_exists(log, now):
            return report(self.name, checkpoint, inspected=len(superseded), emitted=0, metadata={})

        store = getattr(log, "store", None)
        if store is None:
            return report(
                self.name,
                checkpoint,
                inspected=len(superseded),
                emitted=0,
                notes=["value lens learning requires a SQLite-backed event log"],
                metadata={},
            )
        before = load_lens(store, str(SUBJECT_SELF))
        delta = float(getattr(config, "value_lens_sensitivity_delta", 0.1))
        after = normalize_lens(
            before.__class__(
                priorities=before.priorities,
                weights=before.weights,
                sensitivity={
                    **before.sensitivity,
                    winner: round(float(before.sensitivity.get(winner, 1.0)) + delta, 3),
                },
            )
        )
        emitted = 1
        if not bool(getattr(config, "dry_run", False)):
            event = save_lens(
                store,
                emitter,
                after,
                subject_id=str(SUBJECT_SELF),
                trigger=f"learn_value_lens observed {count} {winner.value} wins",
                before=before,
            )
            for projection in projections.all():
                if event.kind in projection.handles:
                    projection.apply(event)
        return report(
            self.name,
            checkpoint,
            inspected=len(superseded),
            emitted=emitted,
            metadata={},
        )


def _decisive_kinds(raw: object) -> list[ValueKind]:
    if not isinstance(raw, list):
        return []
    return [ValueKind(str(item)) for item in raw if item is not None]


def _lens_learning_frozen(projections: ProjectionRegistry) -> bool:
    try:
        projection = projections.get_typed(StrategyProjection)
    except KeyError:
        return False
    return strategy_is_active_for_domain(
        projection.active(domain="lens_learning"),
        "freeze_lens_learning_for_24h",
        "lens_learning",
    )


def _events_after_processed_checkpoint(
    events: list[CognitiveEvent],
    checkpoint_event_id: str,
) -> list[CognitiveEvent]:
    if not checkpoint_event_id:
        return events
    for index, event in enumerate(events):
        if str(event.id) == checkpoint_event_id:
            return events[index + 1:]
    return events


def _after_event_cursor_wrap(events: list[CognitiveEvent], cursor: str) -> list[CognitiveEvent]:
    if not cursor:
        return events
    for index, event in enumerate(events):
        if str(event.id) == cursor:
            return events[index + 1:] + events[: index + 1]
    return events


def _recent_shift_exists(log: EventLog, now: str | None) -> bool:
    latest = None
    for event in log.iter(kinds=[CognitiveEventKind.VALUE_LENS_SHIFTED]):
        latest = event
    if latest is None:
        return False
    now_time = _parse_time(now or str(latest.timestamp))
    shifted_at = _parse_time(str(latest.timestamp))
    return now_time - shifted_at < timedelta(hours=24)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
