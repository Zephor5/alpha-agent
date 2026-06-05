"""Projection registry construction for cognition inspection and workers."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection


def default_projection_registry(event_log: EventLog) -> ProjectionRegistry:
    registry = ProjectionRegistry()
    registry.register(SubjectProjection(event_log))
    registry.register(BeliefProjection(getattr(event_log, "store", None)))
    store = getattr(event_log, "store", None)
    if store is not None:
        registry.register(CounterpartProjection(store))
    registry.register(
        GoalProjection(
            store,
            event_log=event_log,
            auto_rebuild=True,
        )
    )
    return registry


__all__ = ["default_projection_registry"]
