from __future__ import annotations

from alpha_agent.cognition.models import BeliefId
from alpha_agent.cognition.reflectors.rules.premature_novel_auto_form import (
    PrematureNovelAutoFormRule,
)
from alpha_agent.cognition.stages.types import Feedback
from tests.cognition.reflector_helpers import (
    context,
    formed_belief_feedback,
    judgment,
    novel_interpretation,
)


def test_premature_novel_auto_form_triggers_for_low_confidence_novel_belief_formation() -> None:
    ctx = context(
        interpretation=novel_interpretation(),
        judgments=[judgment(confidence=0.4)],
        feedback=formed_belief_feedback("belief:new"),
    )

    reflections = list(PrematureNovelAutoFormRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["premature-novel-auto-form"]
    assert reflections[0].target == "belief:belief:new"


def test_premature_novel_auto_form_does_not_trigger_for_high_confidence_novel_claim() -> None:
    ctx = context(
        interpretation=novel_interpretation(),
        judgments=[judgment(confidence=0.8)],
        feedback=formed_belief_feedback("belief:new"),
    )

    assert list(PrematureNovelAutoFormRule().evaluate(ctx)) == []


def test_premature_novel_auto_form_does_not_trigger_for_non_formation_updates() -> None:
    ctx = context(
        interpretation=novel_interpretation(),
        judgments=[judgment(confidence=0.4)],
        feedback=Feedback(
            matched_expected=True,
            affected_belief_ids=[BeliefId("belief:updated")],
            formed_belief_ids=[],
        ),
    )

    assert list(PrematureNovelAutoFormRule().evaluate(ctx)) == []
