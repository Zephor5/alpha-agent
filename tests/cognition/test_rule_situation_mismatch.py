from __future__ import annotations

from alpha_agent.cognition.reflectors.rules.situation_mismatch import SituationMismatchRule
from tests.cognition.reflector_helpers import context, judgment


def test_situation_mismatch_triggers_when_applicability_names_different_situation() -> None:
    ctx = context(judgments=[judgment(applicable_under="situation:other")])

    reflections = list(SituationMismatchRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["situation-mismatch"]
    assert reflections[0].severity == "info"


def test_situation_mismatch_does_not_trigger_for_current_situation() -> None:
    ctx = context(judgments=[judgment(applicable_under="situation:current")])

    assert list(SituationMismatchRule().evaluate(ctx)) == []
