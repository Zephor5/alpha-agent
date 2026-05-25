"""Data slice consumed by cognition renderers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.models import (
    Belief,
    ContextWindow,
    Counterpart,
    Instant,
    Judgment,
    Reflection,
    Situation,
    Subject,
)
from alpha_agent.llm.base import ChatMessage


@dataclass(frozen=True)
class CognitionView:
    """Immutable render input assembled from projections for one tick."""

    subject: Subject
    counterpart: Counterpart | None
    situation: Situation
    window: ContextWindow
    recalled_beliefs: list[Belief] = field(default_factory=list)
    counterpart_digest: Belief | None = None
    active_judgments: list[Judgment] = field(default_factory=list)
    matched_procedures: list[Any] = field(default_factory=list)
    active_strategies: list[Any] = field(default_factory=list)
    recent_reflections: list[Reflection] = field(default_factory=list)
    assembled_at: Instant = Instant("")
    current_query: str | None = None
    chat_history: list[ChatMessage] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
