"""Stub belief projection for Phase 02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.models import BeliefRef, CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.projections.base import Projection


@dataclass(frozen=True)
class BeliefProjectionView:
    recalled: tuple[BeliefRef, ...] = ()
    status: str = "stub"


class BeliefProjection(Projection):
    """Placeholder until Phase 03 materializes belief recall."""

    name = "belief"
    handles = frozenset(
        {
            CognitiveEventKind.BELIEF_FORMED,
            CognitiveEventKind.BELIEF_STRENGTHENED,
            CognitiveEventKind.BELIEF_WEAKENED,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            CognitiveEventKind.BELIEF_RETRACTED,
        }
    )
    status = "stub"

    def recall(self, *_args: Any, **_kwargs: Any) -> list[BeliefRef]:
        return []

    def apply(self, event: CognitiveEvent) -> None:
        return None

    def reset(self) -> None:
        return None

    def view(self) -> BeliefProjectionView:
        return BeliefProjectionView()
