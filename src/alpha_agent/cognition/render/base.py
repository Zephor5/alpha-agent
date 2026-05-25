"""Renderer contracts for cognition views."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from alpha_agent.cognition.render.view import CognitionView


@dataclass(frozen=True)
class RenderBudget:
    """Best-effort rendering budget.

    Text renderers interpret the values as rough tokens. Inspection renderers use
    the same values as output-size guidance.
    """

    max_tokens: int = 2048
    per_section_tokens: dict[str, int] = field(default_factory=dict)
    style_hints: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderResult:
    """Rendered payload and accounting metadata."""

    payload: Any
    used_tokens: int
    dropped_sections: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class Renderer(Protocol):
    """Pure view-to-payload renderer."""

    name: ClassVar[str]

    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult:
        """Render a cognition view into a provider/debug payload."""
