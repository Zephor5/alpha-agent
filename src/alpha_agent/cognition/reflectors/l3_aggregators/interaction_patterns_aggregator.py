"""Role-level interaction pattern aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartRole,
    InteractionPattern,
    SubjectRef,
)
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class InteractionPatternsAggregator:
    field_name = "interaction_patterns_by_counterpart_role"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> dict[CounterpartRole, InteractionPattern]:
        del subject
        role_by_counterpart = _role_by_counterpart(projections)
        ticks: Counter[str] = Counter()
        feedback: Counter[str] = Counter()
        success: Counter[str] = Counter()
        tick_role = _tick_role_by_perception(log, window, role_by_counterpart)
        for event in log.iter(
            kinds=[CognitiveEventKind.PERCEIVED],
            since=window.since,
            until=window.until,
        ):
            counterpart_id = _counterpart_id(event.payload.get("from_counterpart"))
            role = role_by_counterpart.get(counterpart_id or "")
            if role is not None:
                ticks[role] += 1
        for event in log.iter(
            kinds=[CognitiveEventKind.RECEIVED_FEEDBACK],
            since=window.since,
            until=window.until,
        ):
            role = _feedback_role(event.payload, role_by_counterpart, tick_role)
            if role is None:
                continue
            feedback[role] += 1
            if bool(event.payload.get("matched_expected")):
                success[role] += 1
        reflections_by_role = _reflection_counts(projections, window)
        result: dict[CounterpartRole, InteractionPattern] = {}
        for role_value in sorted(set(ticks) | set(feedback) | set(reflections_by_role)):
            role = CounterpartRole(role_value)
            total_feedback = feedback[role_value]
            success_rate = (success[role_value] / total_feedback) if total_feedback else 0.0
            result[role] = InteractionPattern(
                f"ticks={ticks[role_value]};feedback={total_feedback};"
                f"success_rate={success_rate:.3f};reflections={reflections_by_role[role_value]}"
            )
        return result


def _role_by_counterpart(projections: ProjectionRegistry) -> dict[str, str]:
    try:
        counterparts = projections.get_typed(CounterpartProjection).list_active()
    except KeyError:
        return {}
    return {str(item.id): item.role.value for item in counterparts}


def _counterpart_id(raw: object) -> str | None:
    if isinstance(raw, dict):
        value = raw.get("id")
        return str(value) if value is not None else None
    return None


def _tick_role_by_perception(
    log: EventLog,
    window: AggregationWindow,
    role_by_counterpart: dict[str, str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for event in log.iter(
        kinds=[CognitiveEventKind.PERCEIVED],
        since=window.since,
        until=window.until,
    ):
        tick_id = event.payload.get("tick_id")
        counterpart_id = _counterpart_id(event.payload.get("from_counterpart"))
        role = role_by_counterpart.get(counterpart_id or "")
        if tick_id is not None and role is not None:
            result[str(tick_id)] = role
    return result


def _feedback_role(
    payload: dict[str, object],
    role_by_counterpart: dict[str, str],
    tick_role: dict[str, str],
) -> str | None:
    raw_role = payload.get("counterpart_role") or payload.get("role")
    if raw_role is not None:
        return str(raw_role)
    tick_id = payload.get("tick_id")
    if tick_id is not None and str(tick_id) in tick_role:
        return tick_role[str(tick_id)]
    counterpart_id = payload.get("counterpart_id")
    if counterpart_id is not None:
        return role_by_counterpart.get(str(counterpart_id))
    raw_counterpart = payload.get("from_counterpart")
    return role_by_counterpart.get(_counterpart_id(raw_counterpart) or "")


def _reflection_counts(
    projections: ProjectionRegistry,
    window: AggregationWindow,
) -> defaultdict[str, int]:
    counts: defaultdict[str, int] = defaultdict(int)
    try:
        reflections = projections.get_typed(ReflectionProjection).list_recent(
            last=500,
            since=str(window.since),
            until=str(window.until),
        )
    except KeyError:
        return counts
    for reflection in reflections:
        value = str(reflection.target)
        if value.startswith("role:"):
            counts[value.split(":", 1)[1]] += 1
    return counts
