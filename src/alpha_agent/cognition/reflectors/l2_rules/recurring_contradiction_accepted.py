"""L2 rule for recurring contradiction reflections."""

from __future__ import annotations

from collections import Counter

from alpha_agent.cognition.models import CognitiveEvent, Reflection, StrategyOverride
from alpha_agent.cognition.reflectors.l2_rules._common import (
    StrategyCandidate,
    has_active_strategy,
    within_window,
)

RULE_NAME = "recurring-contradiction-accepted"
STRATEGY_NAME = "require_explicit_confirm_on_contradiction"


def recurring_contradiction_accepted(
    reflections: list[Reflection],
    events: list[CognitiveEvent],
    active_strategies: list[StrategyOverride],
) -> StrategyCandidate | None:
    del events
    if has_active_strategy(active_strategies, STRATEGY_NAME):
        return None
    window = [
        item
        for item in within_window(reflections, minutes=30)
        if "contradiction" in str(item.kind)
    ]
    counts = Counter(str(item.kind) for item in window)
    if not counts:
        return None
    kind, count = counts.most_common(1)[0]
    if count < 3:
        return None
    triggered = [str(item.id) for item in window if str(item.kind) == kind]
    return {
        "rule": RULE_NAME,
        "strategy_name": STRATEGY_NAME,
        "target_stages": ["revise"],
        "payload": {"reflection_kind": kind, "count": count},
        "triggered_by_reflection_ids": triggered,
    }
