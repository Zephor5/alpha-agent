"""L1 reflector stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import CognitiveEventKind, EventId, NLStatement, Reflection
from alpha_agent.cognition.stages.types import Emitted, Feedback, Outcome


class ReflectorL1:
    """Placeholder L1 audit; real rules start in Phase 05."""

    def audit(
        self,
        feedback: Feedback,
        outcome: Outcome,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[list[Reflection]]:
        reflections: list[Reflection] = []
        event = emitter.emit(
            CognitiveEventKind.REFLECTED,
            rationale=NLStatement("No Phase 02 reflection rules fired."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "reflection_count": len(reflections),
                "matched_expected": feedback.matched_expected,
            },
        )
        return Emitted(reflections, event)
