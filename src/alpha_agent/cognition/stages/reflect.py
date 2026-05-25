"""L1 reflector stage for reactive ticks."""

from __future__ import annotations

from typing import Any

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    ContextWindow,
    Counterpart,
    Decision,
    EventId,
    Judgment,
    NLStatement,
    Perception,
    Reference,
    Reflection,
    Subject,
)
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.reflection import target_to_parts
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l1 import AuditContext
from alpha_agent.cognition.reflectors.l1 import ReflectorL1 as RuleReflectorL1
from alpha_agent.cognition.stages.types import (
    AttentionFocus,
    Emitted,
    Feedback,
    Interpretation,
    Outcome,
)


class ReflectorL1:
    """Stage wrapper that runs read-only L1 rules and records their audit events."""

    def __init__(
        self,
        projections: ProjectionRegistry | None = None,
        *,
        rule_reflector: RuleReflectorL1 | None = None,
    ):
        self.projections = projections or ProjectionRegistry()
        self.rule_reflector = rule_reflector or RuleReflectorL1()

    def audit(
        self,
        perception_or_context: Perception | AuditContext,
        focus: AttentionFocus | None = None,
        interpretation: Interpretation | None = None,
        judgments: list[Judgment] | None = None,
        decision: Decision | None = None,
        outcome: Outcome | None = None,
        feedback: Feedback | None = None,
        subject: Subject | None = None,
        window: ContextWindow | None = None,
        *,
        emitter: EventEmitter,
        tick_id: str | None = None,
        causal_parent: EventId,
    ) -> Emitted[list[Reflection]]:
        ctx = self._coerce_context(
            perception_or_context,
            focus,
            interpretation,
            judgments,
            decision,
            outcome,
            feedback,
            subject,
            window,
            emitter=emitter,
            tick_id=tick_id,
        )
        reflections = self.rule_reflector.audit(ctx)
        reflected_event = emitter.emit(
            CognitiveEventKind.REFLECTED,
            inputs=[],
            outputs=[Reference("reflection", str(item.id)) for item in reflections],
            rationale=NLStatement(_rationale(reflections)),
            causal_parents=[causal_parent],
            payload={
                "tick_id": ctx.tick_id,
                "reflection_count": len(reflections),
                "reflection_ids": [str(item.id) for item in reflections],
                "reflections": [item.to_record() for item in reflections],
            },
        )
        for reflection in reflections:
            target_kind, target_id = target_to_parts(reflection.target)
            emitter.emit(
                CognitiveEventKind.BIAS_DETECTED,
                inputs=[Reference("reflection", str(reflection.id))],
                rationale=reflection.finding,
                causal_parents=[reflected_event.id],
                payload={
                    "tick_id": ctx.tick_id,
                    "reflection_id": str(reflection.id),
                    "kind": str(reflection.kind),
                    "severity": str(reflection.severity),
                    "target": {"kind": target_kind, "id": target_id},
                },
            )
        return Emitted(reflections, reflected_event)

    def _coerce_context(
        self,
        perception_or_context: Perception | AuditContext,
        focus: AttentionFocus | None,
        interpretation: Interpretation | None,
        judgments: list[Judgment] | None,
        decision: Decision | None,
        outcome: Outcome | None,
        feedback: Feedback | None,
        subject: Subject | None,
        window: ContextWindow | None,
        *,
        emitter: EventEmitter,
        tick_id: str | None,
    ) -> AuditContext:
        if isinstance(perception_or_context, AuditContext):
            return perception_or_context
        required: dict[str, Any | None] = {
            "focus": focus,
            "interpretation": interpretation,
            "judgments": judgments,
            "decision": decision,
            "outcome": outcome,
            "feedback": feedback,
            "subject": subject,
            "window": window,
            "tick_id": tick_id,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing reflector audit inputs: {', '.join(missing)}")
        assert focus is not None
        assert interpretation is not None
        assert judgments is not None
        assert decision is not None
        assert outcome is not None
        assert feedback is not None
        assert subject is not None
        assert window is not None
        assert tick_id is not None
        return AuditContext(
            tick_id=tick_id,
            perception=perception_or_context,
            focus=focus,
            interpretation=interpretation,
            judgments=judgments,
            decision=decision,
            outcome=outcome,
            feedback=feedback,
            subject=subject,
            counterpart=self._counterpart(window),
            projections=self.projections,
            clock=emitter.clock,
        )

    def _counterpart(self, window: ContextWindow) -> Counterpart | None:
        if window.counterpart is None:
            return None
        try:
            projection = self.projections.get_typed(CounterpartProjection)
        except KeyError:
            return None
        return projection.get(window.counterpart.id)


def _rationale(reflections: list[Reflection]) -> str:
    if not reflections:
        return "No L1 reflection rules fired."
    return f"L1 reflection emitted {len(reflections)} audit finding(s)."
