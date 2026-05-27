"""Promote repeated deterministic judgments into beliefs."""

from __future__ import annotations

from collections import defaultdict
from typing import ClassVar

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import (
    active_belief,
    after_cursor_wrap,
    emit_projected,
    normalize_text,
    report,
    stable_id,
    trigger,
)
from alpha_agent.cognition.models import BeliefId, CognitiveEventKind, CognitiveType
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry


class PromoteJudgmentWorker:
    name: ClassVar[str] = "promote_judgment"
    trigger: ClassVar[ScheduleTrigger] = trigger(5, 6, {CognitiveEventKind.JUDGED}, 10)
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.JUDGED, CognitiveEventKind.BELIEF_FORMED}
    )

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        events = list(log.iter(kinds=[CognitiveEventKind.JUDGED]))
        window = max(1, int(getattr(config, "judgment_repeat_window", 20)))
        threshold = max(1, int(getattr(config, "judgment_repeat_threshold", 3)))
        groups = defaultdict(list)
        for event in events[-window:]:
            for claim in _claims(event.payload):
                groups[normalize_text(claim)].append((claim, event))

        projection = projections.get_typed(BeliefProjection)
        emitted = 0
        pending_groups = after_cursor_wrap(
            sorted(groups.items()),
            str(checkpoint.metadata.get("last_claim", "")),
            lambda item: item[0],
        )
        for normalized, items in pending_groups:
            if normalized and len(items) >= threshold:
                belief_id = stable_id("belief:judgment", normalized)
                if projection.get_by_id(BeliefId(belief_id)) is None:
                    content, first_event = items[0]
                    belief = active_belief(
                        belief_id,
                        content,
                        object_=normalized[:120],
                        cognitive_type=CognitiveType.META,
                        confidence=min(0.95, 0.5 + 0.1 * len(items)),
                        sources=[],
                        held_since=first_event.timestamp,
                    )
                    formed = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.BELIEF_FORMED,
                        config=config,
                        payload={
                            "belief": belief.to_record(),
                            "promoted_from": [str(item.id) for _, item in items],
                        },
                        rationale="Promoted repeated judgment into belief.",
                    )
                    emitted += (
                        1 if formed is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(events),
                    emitted=emitted,
                    last_event=events[-1] if events else None,
                    yielded=True,
                    metadata={"last_claim": normalized},
                )
        return report(
            self.name,
            checkpoint,
            inspected=len(events),
            emitted=emitted,
            last_event=events[-1] if events else None,
            metadata={},
        )


def _claims(payload: dict[str, object]) -> list[str]:
    if isinstance(payload.get("claim"), str):
        return [str(payload["claim"])]
    raw = payload.get("judgments")
    if not isinstance(raw, list):
        return []
    claims: list[str] = []
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("claim"), str):
            claims.append(str(item["claim"]))
    return claims
