"""Attend stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    EventId,
    NLStatement,
    Perception,
    Reference,
    Subject,
    ValueKind,
)
from alpha_agent.cognition.stages.types import AttentionFocus, Emitted


class Attender:
    """Extract a minimal foreground focus from a perception."""

    def focus(
        self,
        perception: Perception,
        subject: Subject,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[AttentionFocus]:
        text = str(perception.raw).strip()
        focus = AttentionFocus(
            entities=list(perception.raised_entities),
            salient_claims=[NLStatement(text)] if text else [],
            value_signals={ValueKind.HELPFULNESS: 1.0, ValueKind.HONESTY: 1.0},
        )
        event = emitter.emit(
            CognitiveEventKind.ATTENDED,
            situation=perception.situation,
            inputs=[Reference("perception", str(perception.id))],
            rationale=f"Focused perception for {subject.id}.",
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "focused_entity_count": len(focus.entities),
                "salient_claim_count": len(focus.salient_claims),
            },
        )
        return Emitted(focus, event)
