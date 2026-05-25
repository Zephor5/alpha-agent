"""Value tradeoff aggregation from resolved belief supersede events."""

from __future__ import annotations

from collections import Counter

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEventKind, SubjectRef, ValueTradeoff
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.reflectors.l3_aggregators.protocol import AggregationWindow


class TradeoffAggregator:
    field_name = "typical_value_tradeoffs"

    def compute(
        self,
        subject: SubjectRef,
        log: EventLog,
        projections: ProjectionRegistry,
        window: AggregationWindow,
    ) -> list[ValueTradeoff]:
        del subject, projections
        counts: Counter[str] = Counter()
        for event in log.iter(
            kinds=[CognitiveEventKind.BELIEF_SUPERSEDED],
            since=window.since,
            until=window.until,
        ):
            raw = event.payload.get("decisive_value_kinds") or []
            if isinstance(raw, list):
                counts.update(str(item) for item in raw)
        return [
            ValueTradeoff(f"{kind}:count={count}")
            for kind, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
        ]
