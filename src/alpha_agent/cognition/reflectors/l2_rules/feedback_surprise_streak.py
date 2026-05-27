"""L2 rule for repeated feedback misses under the same trigger."""

from __future__ import annotations

from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    Reflection,
    StrategyOverride,
)
from alpha_agent.cognition.reflectors.l2_rules._common import (
    StrategyCandidate,
    has_active_strategy,
)

RULE_NAME = "feedback-surprise-streak"
STRATEGY_NAME = "disable_auto_procedure_match_for_trigger"


def feedback_surprise_streak(
    reflections: list[Reflection],
    events: list[CognitiveEvent],
    active_strategies: list[StrategyOverride],
) -> StrategyCandidate | None:
    del reflections
    if has_active_strategy(active_strategies, STRATEGY_NAME):
        return None
    streaks: dict[str, list[CognitiveEvent]] = {}
    for event in events:
        if event.kind != CognitiveEventKind.RECEIVED_FEEDBACK:
            continue
        trigger = str(event.payload.get("trigger") or event.payload.get("action") or "")
        if not trigger:
            continue
        if event.payload.get("matched_expected") is False:
            streaks.setdefault(trigger, []).append(event)
        else:
            streaks[trigger] = []
    candidates = [
        (trigger, group)
        for trigger, group in streaks.items()
        if len(group) >= 5
    ]
    if not candidates:
        return None
    trigger, group = sorted(candidates, key=lambda item: (-len(item[1]), item[0]))[0]
    return _candidate(trigger, group)


def _candidate(trigger: str, group: list[CognitiveEvent]) -> StrategyCandidate:
    return {
        "rule": RULE_NAME,
        "strategy_name": STRATEGY_NAME,
        "target_stages": ["decide"],
        "payload": {"trigger": trigger, "count": len(group)},
        "triggered_by_reflection_ids": [],
    }
