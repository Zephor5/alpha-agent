from __future__ import annotations

from dataclasses import replace

from alpha_agent.cognition.models import (
    Action,
    Applicability,
    BeliefId,
    Counterpart,
    CounterpartId,
    CounterpartRole,
    Decision,
    DecisionId,
    ExpectedFeedback,
    Instant,
    Judgment,
    JudgmentId,
    NLStatement,
    Perception,
    PerceptionId,
    Reference,
    Relationship,
    SituationId,
    StimulusKind,
    Subject,
    ValueKind,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l1 import AuditContext
from alpha_agent.cognition.stages.types import AttentionFocus, Feedback, Interpretation, Outcome

NOW = Instant("2026-01-01T00:00:00+00:00")
SITUATION = situation_ref(SituationId("situation:current"))


def judgment(
    judgment_id: str = "judgment:1",
    *,
    confidence: float = 0.8,
    value_weights: dict[ValueKind | str, float] | None = None,
    supports: list[Reference] | None = None,
    undermined_by: list[Reference] | None = None,
    applicable_under: str = "reactive_tick",
    claim: str = "Healthy reactive judgment.",
) -> Judgment:
    return Judgment(
        id=JudgmentId(judgment_id),
        claim=NLStatement(claim),
        supports=supports or [],
        undermined_by=undermined_by or [],
        applicable_under=Applicability(applicable_under),
        confidence=confidence,
        value_weights=value_weights or {ValueKind.HELPFULNESS: 1.0},
        formed_in=SITUATION,
        expires_at=None,
    )


def decision(action: str = "respond", *, justified_by: list[Reference] | None = None) -> Decision:
    return Decision(
        id=DecisionId("decision:1"),
        action=Action(action),
        payload={},
        justified_by=justified_by or [Reference("judgment", "judgment:1")],
        expected_feedback=ExpectedFeedback("assistant_response_delivered"),
        fallback=None,
        decided_at=NOW,
    )


def context(
    *,
    interpretation: Interpretation | None = None,
    judgments: list[Judgment] | None = None,
    decision_: Decision | None = None,
    feedback: Feedback | None = None,
    counterpart: Counterpart | None = None,
) -> AuditContext:
    subject = Subject(held_at=NOW)
    perception = Perception(
        id=PerceptionId("perception:1"),
        source_kind=StimulusKind.USER_MESSAGE,
        from_counterpart=None,
        raw="hello",
        surface_intent=[],
        raised_entities=[],
        subject=subject_ref(SUBJECT_SELF),
        situation=SITUATION,
        received_at=NOW,
    )
    focus = AttentionFocus(entities=[], salient_claims=[NLStatement("hello")], value_signals={})
    base_judgment = judgment()
    return AuditContext(
        tick_id="tick:1",
        perception=perception,
        focus=focus,
        interpretation=interpretation
        or Interpretation(
            stance="consistent",
            supporting_beliefs=[],
            contradicting_beliefs=[],
            novel_claims=[],
            ambiguity_notes=[],
            source_text="hello",
        ),
        judgments=judgments or [base_judgment],
        decision=decision_ or decision(justified_by=[Reference("judgment", str(base_judgment.id))]),
        outcome=Outcome(text="ok", tool_calls=[], tool_results=[], raw_llm_response=None),
        feedback=feedback
        or Feedback(matched_expected=True, surprises=[], affected_belief_ids=[]),
        subject=subject,
        counterpart=counterpart,
        projections=ProjectionRegistry(),
        clock=lambda: str(NOW),
        id_factory=_id_factory(),
    )


def novel_interpretation() -> Interpretation:
    return Interpretation(
        stance="novel",
        supporting_beliefs=[],
        contradicting_beliefs=[],
        novel_claims=[NLStatement("A new claim.")],
        ambiguity_notes=[],
        source_text="A new claim.",
    )


def counterpart(trust_level: float = 0.5) -> Counterpart:
    return Counterpart(
        id=CounterpartId("counterpart:user"),
        role=CounterpartRole.USER,
        identity={},
        relationship=Relationship(),
        service_contract=[],
        trust_level=trust_level,
        communication_style=[],
        first_seen_at=NOW,
        last_interaction_at=NOW,
    )


def formed_belief_feedback(belief_id: str = "belief:new") -> Feedback:
    return Feedback(
        matched_expected=True,
        surprises=[],
        affected_belief_ids=[],
        formed_belief_ids=[BeliefId(belief_id)],
    )


def with_counterpart(ctx: AuditContext, value: Counterpart) -> AuditContext:
    return replace(ctx, counterpart=value)


def _id_factory():
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"reflection:{counter}"

    return next_id
