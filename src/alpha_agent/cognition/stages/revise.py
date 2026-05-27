"""Revise stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    EventId,
    Judgment,
    NLStatement,
    StrategyOverride,
)
from alpha_agent.cognition.projections.strategy import strategy_is_active_for_stage
from alpha_agent.cognition.stages.types import (
    Emitted,
    Feedback,
    Interpretation,
    Reflection,
    Revision,
)


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
        interpretation: Interpretation | None = None,
        strategies: list[StrategyOverride] | None = None,
    ) -> Emitted[list[Revision]]:
        revisions: list[Revision] = []
        pending_confirmation = False
        if (
            interpretation is not None
            and interpretation.stance == "contradicting"
            and strategy_is_active_for_stage(
                strategies or [],
                "require_explicit_confirm_on_contradiction",
                "revise",
            )
        ):
            pending_confirmation = True
            emitter.emit(
                CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION,
                rationale=NLStatement(
                    "Strategy requires confirmation before contradiction update."
                ),
                causal_parents=[causal_parent],
                payload={
                    "tick_id": tick_id,
                    "reason": "strategy:require_explicit_confirm_on_contradiction",
                    "contradict_ids": [
                        ref.id for ref in interpretation.contradicting_beliefs
                    ],
                },
            )
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
                "pending_confirmation": pending_confirmation,
            },
        )
        return Emitted(revisions, event)
