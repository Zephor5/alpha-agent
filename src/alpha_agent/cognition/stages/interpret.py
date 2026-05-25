"""Interpret stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    BeliefRef,
    CognitiveEventKind,
    ContextWindow,
    EventId,
    NLStatement,
    Reference,
    Subject,
)
from alpha_agent.cognition.stages._payload import ref_ids
from alpha_agent.cognition.stages.types import AttentionFocus, Emitted, Interpretation


class Interpreter:
    """Compare the focus against recalled beliefs."""

    def interpret(
        self,
        focus: AttentionFocus,
        window: ContextWindow,
        recalled: list[BeliefRef],
        subject: Subject,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[Interpretation]:
        text = "\n".join(str(claim) for claim in focus.salient_claims)
        stance = "novel" if not recalled else "consistent"
        interpretation = Interpretation(
            stance=stance,
            supporting_beliefs=list(recalled),
            contradicting_beliefs=[],
            novel_claims=list(focus.salient_claims) if stance == "novel" else [],
            ambiguity_notes=[] if text else ["empty stimulus"],
            source_text=text,
        )
        event = emitter.emit(
            CognitiveEventKind.INTERPRETED,
            situation=window.situation_at,
            inputs=[Reference("subject", str(subject.id))],
            rationale=NLStatement("Interpreted focus against recalled beliefs."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "stance": interpretation.stance,
                "support_ids": ref_ids(interpretation.supporting_beliefs),
                "contradict_ids": ref_ids(interpretation.contradicting_beliefs),
                "novel_claim_count": len(interpretation.novel_claims),
            },
        )
        return Emitted(interpretation, event)
