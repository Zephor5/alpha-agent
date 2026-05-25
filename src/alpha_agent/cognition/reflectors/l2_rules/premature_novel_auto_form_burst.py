"""L2 rule for bursts of novel auto belief formation."""

from __future__ import annotations

from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    Reflection,
    StrategyOverride,
)
from alpha_agent.cognition.reflectors.l2_rules._common import event_window, has_active_strategy

RULE_NAME = "premature-novel-auto-form-burst"
STRATEGY_NAME = "require_confirm_before_novel_form"


def premature_novel_auto_form_burst(
    reflections: list[Reflection],
    events: list[CognitiveEvent],
    active_strategies: list[StrategyOverride],
) -> dict[str, object] | None:
    del reflections
    if has_active_strategy(active_strategies, STRATEGY_NAME):
        return None
    formed = [
        event
        for event in event_window(events, hours=1)
        if event.kind == CognitiveEventKind.BELIEF_FORMED
        and (
            event.payload.get("auto_formed_novel") is True
            or event.payload.get("origin") == "novel_auto_form"
        )
    ]
    if len(formed) < 5:
        return None
    return {
        "rule": RULE_NAME,
        "strategy_name": STRATEGY_NAME,
        "target_stages": ["interpret"],
        "payload": {"formed_count": len(formed)},
        "triggered_by_reflection_ids": [],
    }
