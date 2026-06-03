"""L2 rule for repeated value-lens shifts."""

from __future__ import annotations

import json
from collections import Counter
from numbers import Real

from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    Reflection,
    StrategyOverride,
)
from alpha_agent.cognition.reflectors.l2_rules._common import (
    StrategyCandidate,
    event_window,
    has_active_strategy,
)

RULE_NAME = "lens-shift-flap"
STRATEGY_NAME = "freeze_lens_learning_for_24h"


def lens_shift_flap(
    reflections: list[Reflection],
    events: list[CognitiveEvent],
    active_strategies: list[StrategyOverride],
) -> StrategyCandidate | None:
    del reflections
    if has_active_strategy(active_strategies, STRATEGY_NAME):
        return None
    directions = [
        direction
        for event in event_window(events, hours=24)
        if event.kind == CognitiveEventKind.VALUE_LENS_SHIFTED
        for direction in [_shift_direction(event)]
        if direction
    ]
    if not directions:
        return None
    direction, count = sorted(Counter(directions).items(), key=lambda item: (-item[1], item[0]))[0]
    if count < 3:
        return None
    return {
        "rule": RULE_NAME,
        "strategy_name": STRATEGY_NAME,
        "target_domains": ["lens_learning"],
        "payload": {"shift_count": count, "direction": direction},
        "triggered_by_reflection_ids": [],
    }


def _shift_direction(event: CognitiveEvent) -> str:
    before = _mapping(event.payload.get("before"))
    after = _mapping(event.payload.get("after"))
    if before and after:
        priority_direction = _priority_direction(before, after)
        if priority_direction:
            return priority_direction
        sensitivity_direction = _sensitivity_direction(before, after)
        if sensitivity_direction:
            return sensitivity_direction
    trigger = str(event.payload.get("trigger") or "").strip()
    return f"trigger:{trigger}" if trigger else ""


def _priority_direction(before: dict[str, object], after: dict[str, object]) -> str:
    before_priority = before.get("priorities") or before.get("priority")
    after_priority = after.get("priorities") or after.get("priority")
    if before_priority == after_priority or after_priority is None:
        return ""
    return (
        "priority:"
        f"{json.dumps(before_priority, sort_keys=True)}->"
        f"{json.dumps(after_priority, sort_keys=True)}"
    )


def _sensitivity_direction(before: dict[str, object], after: dict[str, object]) -> str:
    before_sensitivity = _mapping(before.get("sensitivity"))
    after_sensitivity = _mapping(after.get("sensitivity"))
    keys = sorted(set(before_sensitivity) | set(after_sensitivity))
    directions: list[str] = []
    for key in keys:
        before_value = _float_value(before_sensitivity.get(key), 1.0)
        after_value = _float_value(after_sensitivity.get(key), 1.0)
        if after_value > before_value:
            directions.append(f"sensitivity:{key}:up")
        elif after_value < before_value:
            directions.append(f"sensitivity:{key}:down")
    return "|".join(directions)


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _float_value(value: object, default: float) -> float:
    return float(value) if isinstance(value, Real | str) else default
