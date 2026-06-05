"""Projection protocol and marker view."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind


class ProjectionView(Protocol):
    """Marker protocol for projection views."""


class Projection(ABC):
    """Base class for resettable materialized cognition views."""

    name: ClassVar[str]
    handles: ClassVar[frozenset[CognitiveEventKind]]

    @abstractmethod
    def reset(self) -> None:
        """Clear materialized state."""

    @abstractmethod
    def view(self) -> Any:
        """Return the current projection view."""


class EventProjection(Projection):
    """Projection that materializes state by applying cognitive events."""

    @abstractmethod
    def apply(self, event: CognitiveEvent) -> None:
        """Apply one event."""
