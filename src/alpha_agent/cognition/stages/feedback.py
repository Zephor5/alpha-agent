"""Feedback stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import CognitiveEventKind, Decision, EventId, NLStatement
from alpha_agent.cognition.stages.types import Emitted, Feedback, Outcome


class FeedbackReader:
    """Compare expected feedback against the outcome."""

    def compare(
        self,
        decision: Decision,
        outcome: Outcome,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[Feedback]:
        feedback = Feedback(
            matched_expected=bool(outcome.text) or bool(outcome.tool_results),
            surprises=[] if outcome.text or outcome.tool_results else ["empty_outcome"],
            affected_belief_ids=[],
        )
        event = emitter.emit(
            CognitiveEventKind.RECEIVED_FEEDBACK,
            inputs=[],
            rationale=NLStatement("Observed effector outcome."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "matched_expected": feedback.matched_expected,
                "surprises": list(feedback.surprises),
                "expected_feedback": str(decision.expected_feedback),
            },
        )
        return Emitted(feedback, event)
