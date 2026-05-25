"""Revise stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import CognitiveEventKind, EventId, Judgment, NLStatement
from alpha_agent.cognition.stages.types import Emitted, Feedback, Reflection, Revision


class Reviser:
    """Derive Phase 02 revisions without applying long-term belief logic."""

    def derive(
        self,
        feedback: Feedback,
        reflections: list[Reflection],
        judgments: list[Judgment],
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[list[Revision]]:
        revisions: list[Revision] = []
        event = emitter.emit(
            CognitiveEventKind.REVISED,
            rationale=NLStatement("No Phase 02 durable revisions derived."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "revisions": [revision.kind for revision in revisions],
                "judgment_count": len(judgments),
                "reflection_count": len(reflections),
                "matched_expected": feedback.matched_expected,
            },
        )
        return Emitted(revisions, event)
