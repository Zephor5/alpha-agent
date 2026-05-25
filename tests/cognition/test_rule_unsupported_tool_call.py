from __future__ import annotations

from alpha_agent.cognition.models import Reference
from alpha_agent.cognition.reflectors.rules.unsupported_tool_call import UnsupportedToolCallRule
from tests.cognition.reflector_helpers import context, decision, judgment


def test_unsupported_tool_call_triggers_when_no_judgment_requires_a_tool() -> None:
    ctx = context(
        judgments=[judgment(claim="Answer directly.")],
        decision_=decision("use_tool", justified_by=[Reference("judgment", "judgment:1")]),
    )

    reflections = list(UnsupportedToolCallRule().evaluate(ctx))

    assert [item.kind for item in reflections] == ["unsupported-tool-call"]
    assert reflections[0].target == "decision:decision:1"


def test_unsupported_tool_call_does_not_trigger_when_judgment_requires_tool_use() -> None:
    ctx = context(
        judgments=[judgment(claim="Use tool to inspect current state.")],
        decision_=decision("use_tool", justified_by=[Reference("judgment", "judgment:1")]),
    )

    assert list(UnsupportedToolCallRule().evaluate(ctx)) == []
