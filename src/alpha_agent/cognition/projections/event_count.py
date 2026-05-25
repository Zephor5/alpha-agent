"""Demo projection that counts events by kind."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.projections.base import Projection


@dataclass(frozen=True)
class EventCountByKindView:
    counts: dict[CognitiveEventKind, int] = field(default_factory=dict)


class EventCountByKind(Projection):
    """Trivial replay target used to prove projection plumbing."""

    name = "event_count_by_kind"
    handles = frozenset(CognitiveEventKind)

    def __init__(self) -> None:
        self._counts: dict[CognitiveEventKind, int] = {}

    def apply(self, event: CognitiveEvent) -> None:
        self._counts[event.kind] = self._counts.get(event.kind, 0) + 1

    def reset(self) -> None:
        self._counts.clear()

    def view(self) -> EventCountByKindView:
        return EventCountByKindView(counts=dict(self._counts))
