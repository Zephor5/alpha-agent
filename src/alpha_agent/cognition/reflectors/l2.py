"""Scheduler-compatible deterministic L2 reflector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
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
from alpha_agent.cognition.loops.workers._common import emit_projected, report
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    StrategyId,
    StrategyOverride,
)
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import StrategyProjection
from alpha_agent.cognition.reflectors.l2_rules import RULES
from alpha_agent.cognition.reflectors.l2_rules._common import StrategyCandidate
from alpha_agent.utils.time import utc_now_iso


class ReflectorL2:
    name: ClassVar[str] = "reflector_l2"
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(minutes=1),
        max_interval=timedelta(hours=6),
        watches=frozenset({CognitiveEventKind.REFLECTED, CognitiveEventKind.BIAS_DETECTED}),
        min_new_events=1,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.REFLECTED, CognitiveEventKind.BIAS_DETECTED}
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
        reflection_projection = projections.get_typed(ReflectionProjection)
        strategy_projection = projections.get_typed(StrategyProjection)
        now = Instant(utc_now_iso())
        reflections = reflection_projection.list_recent(last=200)
        events = list(
            log.iter(
                kinds=[
                    CognitiveEventKind.RECEIVED_FEEDBACK,
                    CognitiveEventKind.VALUE_LENS_SHIFTED,
                    CognitiveEventKind.BELIEF_FORMED,
                ]
            )
        )
        active = strategy_projection.active(now=now)
        emitted = 0
        for rule in RULES:
            candidate = rule(reflections, events, active)
            if candidate is not None and len(active) < 5:
                strategy = _strategy_from_candidate(candidate, now)
                event = emit_projected(
                    emitter,
                    projections,
                    CognitiveEventKind.STRATEGY_CHANGED,
                    config=config,
                    payload={
                        "strategy": strategy.to_record(),
                        "triggered_by_reflection_ids": candidate[
                            "triggered_by_reflection_ids"
                        ],
                        "triggered_by_rule": candidate["rule"],
                    },
                    rationale=f"L2 rule {candidate['rule']} changed reactive strategy.",
                )
                if event is not None or getattr(config, "dry_run", False):
                    emitted += 1
                    active.append(strategy)
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(RULES),
                    emitted=emitted,
                    yielded=True,
                    metadata={},
                )
        return report(
            self.name,
            checkpoint,
            inspected=len(RULES),
            emitted=emitted,
            metadata={},
        )


def _strategy_from_candidate(candidate: StrategyCandidate, now: Instant) -> StrategyOverride:
    valid_until = Instant(str(_parse_time(now) + timedelta(hours=24)))
    record: dict[str, object] = {
        "name": candidate["strategy_name"],
        "payload": candidate["payload"],
        "target_stages": candidate["target_stages"],
        "set_at": str(now),
        "valid_until": str(valid_until),
    }
    return StrategyOverride(
        id=StrategyId(_stable_strategy_id(candidate["rule"], record)),
        name=str(candidate["strategy_name"]),
        payload=candidate["payload"],
        target_stages=list(candidate["target_stages"]),
        set_by="reflector_l2",
        set_at=now,
        valid_until=valid_until,
    )


def _stable_strategy_id(rule: object, record: Mapping[str, object]) -> str:
    digest = hashlib.sha1(
        json.dumps([rule, record], ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"strategy:{digest}"


def _parse_time(value: Instant) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
