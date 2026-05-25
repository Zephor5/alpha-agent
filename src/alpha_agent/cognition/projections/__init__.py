"""Projection implementations."""
from alpha_agent.cognition.projections.base import Projection, ProjectionView
from alpha_agent.cognition.projections.counterpart import (
    CounterpartProjection,
    CounterpartProjectionView,
)
from alpha_agent.cognition.projections.event_count import EventCountByKind, EventCountByKindView
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.cognition.projections.reflection import (
    ReflectionProjection,
    ReflectionProjectionView,
)
from alpha_agent.cognition.projections.registry import ProjectionRegistry

__all__ = [
    "CounterpartProjection",
    "CounterpartProjectionView",
    "EventCountByKind",
    "EventCountByKindView",
    "GoalProjection",
    "Projection",
    "ProjectionRegistry",
    "ProjectionView",
    "ReflectionProjection",
    "ReflectionProjectionView",
]
