"""Shared contracts for L3 SelfModel aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import Instant, SubjectRef
from alpha_agent.cognition.projections.registry import ProjectionRegistry


@dataclass(frozen=True)
class AggregationWindow:
    since: Instant
    until: Instant


class SelfModelAggregator(Protocol):
    field_name: ClassVar[str]

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> Any: ...
