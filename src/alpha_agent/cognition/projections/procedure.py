"""Stub procedure projection for Phase 02."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, ProcedureRef
from alpha_agent.cognition.projections.base import Projection


@dataclass(frozen=True)
class ProcedureProjectionView:
    matched: tuple[ProcedureRef, ...] = ()
    status: str = "stub"


class ProcedureProjection(Projection):
    """Placeholder until procedures can be learned and matched."""

    name = "procedure"
    handles = frozenset(
        {
            CognitiveEventKind.PROCEDURE_LEARNED,
            CognitiveEventKind.PROCEDURE_STRENGTHENED,
            CognitiveEventKind.PROCEDURE_WEAKENED,
            CognitiveEventKind.PROCEDURE_MATCHED,
        }
    )
    status = "stub"

    def match(self, *_args: Any, **_kwargs: Any) -> list[ProcedureRef]:
        return []

    def apply(self, event: CognitiveEvent) -> None:
        return None

    def reset(self) -> None:
        return None

    def view(self) -> ProcedureProjectionView:
        return ProcedureProjectionView()
