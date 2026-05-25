from __future__ import annotations

from alpha_agent.cognition.models import Reference
from alpha_agent.cognition.reflectors.rules.contradiction_accepted import (
    ContradictionAcceptedRule,
)
from tests.cognition.reflector_helpers import context, judgment


def test_contradiction_accepted_triggers_when_same_belief_supports_and_undermines() -> None:
    belief = Reference("belief", "belief:python")
    ctx = context(judgments=[judgment(supports=[belief], undermined_by=[belief])])

    reflections = list(ContradictionAcceptedRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["contradiction-accepted"]
    assert reflections[0].severity == "blocker"


def test_contradiction_accepted_does_not_trigger_for_disjoint_beliefs() -> None:
    ctx = context(
        judgments=[
            judgment(
                supports=[Reference("belief", "belief:python")],
                undermined_by=[Reference("belief", "belief:rust")],
            )
        ]
    )

    assert list(ContradictionAcceptedRule().evaluate(ctx)) == []
