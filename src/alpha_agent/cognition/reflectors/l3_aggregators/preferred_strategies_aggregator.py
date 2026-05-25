"""Preferred strategy references from L2-emitted strategy history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEventKind, StrategyRef, SubjectRef
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class PreferredStrategiesAggregator:
    field_name = "preferred_strategies"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> list[StrategyRef]:
        del subject, projections, window
        expired_at = _expired_at_by_strategy(log)
        durations: list[_StrategyDuration] = []
        for event in log.iter(kinds=[CognitiveEventKind.STRATEGY_CHANGED]):
            raw = event.payload.get("strategy")
            if not isinstance(raw, dict) or raw.get("set_by") != "reflector_l2":
                continue
            strategy_id = str(raw.get("id") or "")
            if not strategy_id:
                continue
            set_at = str(raw.get("set_at") or event.timestamp)
            valid_until = str(raw.get("valid_until") or set_at)
            end = min(
                _parse_time(valid_until),
                expired_at.get(strategy_id, _parse_time(valid_until)),
            )
            start = _parse_time(set_at)
            seconds = max(0.0, (end - start).total_seconds())
            durations.append(_StrategyDuration(strategy_id, seconds, set_at))
        return [
            StrategyRef("strategy", item.strategy_id)
            for item in sorted(
                durations,
                key=lambda item: (-item.active_seconds, item.set_at, item.strategy_id),
            )
        ][:12]


@dataclass(frozen=True)
class _StrategyDuration:
    strategy_id: str
    active_seconds: float
    set_at: str


def _expired_at_by_strategy(log: EventLog) -> dict[str, datetime]:
    result: dict[str, datetime] = {}
    for event in log.iter(kinds=[CognitiveEventKind.STRATEGY_EXPIRED]):
        strategy_id = event.payload.get("strategy_id")
        if strategy_id is None:
            continue
        result.setdefault(str(strategy_id), _parse_time(str(event.timestamp)))
    return result


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
