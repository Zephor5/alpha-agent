"""Projection protocol and marker view."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind


class ProjectionView(Protocol):
    """Marker protocol for projection views."""


class Projection(ABC):
    """Base class for idempotently rebuildable projections."""

    name: ClassVar[str]
    handles: ClassVar[frozenset[CognitiveEventKind]]

    @abstractmethod
    def apply(self, event: CognitiveEvent) -> None:
        """Apply one event."""

    @abstractmethod
    def reset(self) -> None:
        """Clear materialized state."""

    @abstractmethod
    def view(self) -> Any:
        """Return the current projection view."""
