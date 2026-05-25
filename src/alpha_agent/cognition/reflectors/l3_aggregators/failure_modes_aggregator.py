"""Failure mode aggregation from reflection kinds."""

from __future__ import annotations

from collections import Counter

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import FailurePattern, SubjectRef
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class FailureModesAggregator:
    field_name = "typical_failure_modes"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> list[FailurePattern]:
        del subject, log
        try:
            reflections = projections.get_typed(ReflectionProjection).list_recent(
                last=500,
                since=str(window.since),
                until=str(window.until),
            )
        except KeyError:
            return []
        counts = Counter(str(item.kind) for item in reflections)
        return [
            FailurePattern(f"{kind}:count={count}")
            for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        ]
