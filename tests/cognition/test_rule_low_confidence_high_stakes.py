from __future__ import annotations

from alpha_agent.cognition.reflectors.rules.low_confidence_high_stakes import (
    LowConfidenceHighStakesRule,
)
from tests.cognition.reflector_helpers import context, judgment


def test_low_confidence_high_stakes_triggers_for_low_confidence_safety_judgment() -> None:
    ctx = context(
        judgments=[
            judgment(
                confidence=0.3,
                value_weights={"existence": 0.8},
            )
        ]
    )

    reflections = list(LowConfidenceHighStakesRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["low-confidence-high-stakes"]
    assert reflections[0].severity == "warning"
    assert reflections[0].target == "judgment:judgment:1"


def test_low_confidence_high_stakes_does_not_trigger_when_stakes_are_low() -> None:
    ctx = context(judgments=[judgment(confidence=0.3, value_weights={"existence": 0.2})])

    assert list(LowConfidenceHighStakesRule().evaluate(ctx)) == []


def test_low_confidence_high_stakes_does_not_trigger_when_confidence_is_high() -> None:
    ctx = context(judgments=[judgment(confidence=0.8, value_weights={"existence": 0.9})])

    assert list(LowConfidenceHighStakesRule().evaluate(ctx)) == []
