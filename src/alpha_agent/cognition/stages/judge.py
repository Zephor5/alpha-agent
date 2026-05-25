"""Judge stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    Applicability,
    CognitiveEventKind,
    EventId,
    Judgment,
    JudgmentId,
    NLStatement,
    Reference,
    SituationRef,
    ValueKind,
    ValueLens,
)
from alpha_agent.cognition.stages.types import Emitted, Interpretation
from alpha_agent.utils.ids import new_id


class Judger:
    """Form short-lived judgments from an interpretation."""

    def judge(
        self,
        interpretation: Interpretation,
        value_lens: ValueLens,
        *,
        situation: SituationRef,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[list[Judgment]]:
        claim = interpretation.source_text or "No actionable user message."
        judgments = [
            Judgment(
                id=JudgmentId(new_id("judgment")),
                claim=NLStatement(claim),
                supports=list(interpretation.supporting_beliefs),
                undermined_by=list(interpretation.contradicting_beliefs),
                applicable_under=Applicability("reactive_tick"),
                confidence=0.5 if interpretation.stance == "ambiguous" else 0.8,
                value_weights={ValueKind.HELPFULNESS: 1.0, ValueKind.HONESTY: 1.0},
                formed_in=situation,
                expires_at=None,
            )
        ]
        event = emitter.emit(
            CognitiveEventKind.JUDGED,
            situation=situation,
            inputs=[Reference("value_lens", value_lens.__class__.__name__)],
            outputs=[Reference("judgment", str(item.id)) for item in judgments],
            rationale=NLStatement("Judged interpretation with the current value lens."),
            causal_parents=[causal_parent],
            payload={"tick_id": tick_id, "judgment_count": len(judgments)},
        )
        return Emitted(judgments, event)
