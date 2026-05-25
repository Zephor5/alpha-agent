"""Decide stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    Action,
    CognitiveEventKind,
    ContextWindow,
    Decision,
    DecisionId,
    EventId,
    ExpectedFeedback,
    Judgment,
    NLStatement,
    ProcedureRef,
    Reference,
    Subject,
)
from alpha_agent.cognition.stages.types import Emitted
from alpha_agent.utils.ids import new_id


class Decider:
    """Choose the next action for a reactive turn."""

    def decide(
        self,
        judgments: list[Judgment],
        procedures: list[ProcedureRef],
        subject: Subject,
        window: ContextWindow,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[Decision]:
        latest_claim = str(judgments[-1].claim) if judgments else ""
        action = "use_tool" if procedures or "tool" in latest_claim.lower() else "respond"
        decision = Decision(
            id=DecisionId(new_id("decision")),
            action=Action(action),
            payload={
                "message": latest_claim,
                "thread_id": window.thread_id.to_record(),
                "counterpart": window.counterpart.to_record()
                if window.counterpart is not None
                else None,
            },
            justified_by=[Reference("judgment", str(item.id)) for item in judgments],
            expected_feedback=ExpectedFeedback("assistant_response_delivered"),
            fallback=None,
            decided_at=window.assembled_at,
        )
        event = emitter.emit(
            CognitiveEventKind.DECIDED,
            situation=window.situation_at,
            inputs=[Reference("subject", str(subject.id)), *decision.justified_by],
            outputs=[Reference("decision", str(decision.id))],
            rationale=NLStatement("Selected the reactive response action."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "action": str(decision.action),
                "expected_feedback": str(decision.expected_feedback),
                "procedure_count": len(procedures),
            },
        )
        return Emitted(decision, event)
