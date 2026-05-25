from __future__ import annotations

from alpha_agent.cognition.reflectors.rules.feedback_surprise import FeedbackSurpriseRule
from alpha_agent.cognition.stages.types import Feedback
from tests.cognition.reflector_helpers import context


def test_feedback_surprise_triggers_when_feedback_does_not_match_expectation() -> None:
    ctx = context(
        feedback=Feedback(
            matched_expected=False,
            surprises=["empty_outcome"],
            affected_belief_ids=[],
        )
    )

    reflections = list(FeedbackSurpriseRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["feedback-surprise"]
    assert reflections[0].severity == "info"
    assert reflections[0].target == "loop_run:tick:1"


def test_feedback_surprise_does_not_trigger_without_surprises() -> None:
    ctx = context(feedback=Feedback(matched_expected=False, surprises=[], affected_belief_ids=[]))

    assert list(FeedbackSurpriseRule().evaluate(ctx)) == []


def test_feedback_surprise_does_not_trigger_when_feedback_matches_expectation() -> None:
    ctx = context(
        feedback=Feedback(
            matched_expected=True,
            surprises=["unexpected_extra_context"],
        )
    )

    assert list(FeedbackSurpriseRule().evaluate(ctx)) == []
