"""Projection registry."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

from alpha_agent.cognition.projections.base import Projection

P = TypeVar("P", bound=Projection)


class ProjectionRegistry:
    """Lookup projection instances by name or concrete type."""

    def __init__(self) -> None:
        self._by_name: dict[str, Projection] = {}

    def register(self, projection: Projection) -> None:
        if projection.name in self._by_name:
            raise ValueError(f"projection already registered: {projection.name}")
        self._by_name[projection.name] = projection

    def get(self, name: str) -> Projection:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"unknown projection: {name}") from exc

    def get_typed(self, cls: type[P]) -> P:
        for projection in self._by_name.values():
            if isinstance(projection, cls):
                return projection
        raise KeyError(f"unknown projection type: {cls.__name__}")

    def all(self) -> Iterable[Projection]:
        return tuple(self._by_name.values())
