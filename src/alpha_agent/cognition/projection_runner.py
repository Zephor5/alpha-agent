"""Projection replay runner."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class ProjectionRunner:
    """Replay cognitive events into registered projections."""

    def __init__(self, log: EventLog, registry: ProjectionRegistry):
        self.log = log
        self.registry = registry

    def replay_all(self) -> None:
        for projection in self.registry.all():
            projection.reset()
        for event in self.log.iter():
            self.apply_one(event)

    def apply_one(self, event: CognitiveEvent) -> None:
        for projection in self.registry.all():
            if event.kind in projection.handles:
                projection.apply(event)
