"""Shared helpers for deterministic L2 rules."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any, TypedDict

from alpha_agent.cognition.models import CognitiveEvent, Reflection, StrategyOverride


class StrategyCandidate(TypedDict):
    rule: str
    strategy_name: str
    target_stages: list[str]
    payload: dict[str, Any]
    triggered_by_reflection_ids: list[str]


def parse_time(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def latest_time(values: Iterable[object]) -> datetime | None:
    parsed = [parse_time(item) for item in values if str(item)]
    return max(parsed) if parsed else None


def within_window(items: list[Reflection], *, minutes: int) -> list[Reflection]:
    latest = latest_time(item.created_at for item in items)
    if latest is None:
        return []
    floor = latest - timedelta(minutes=minutes)
    return [item for item in items if parse_time(item.created_at) >= floor]


def event_window(events: list[CognitiveEvent], *, hours: int) -> list[CognitiveEvent]:
    latest = latest_time(item.timestamp for item in events)
    if latest is None:
        return []
    floor = latest - timedelta(hours=hours)
    return [item for item in events if parse_time(item.timestamp) >= floor]


def has_active_strategy(strategies: list[StrategyOverride], name: str) -> bool:
    return any(strategy.name == name for strategy in strategies)
