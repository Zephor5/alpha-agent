"""Maintain one digest belief per active counterpart."""

from __future__ import annotations

from collections.abc import Sequence
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
    belief_source_refs,
    emit_projected,
    report,
    stable_id,
    trigger,
)
from alpha_agent.cognition.models import (
    Belief,
    CognitiveEventKind,
    CognitiveType,
    CounterpartId,
    counterpart_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry

DIGEST_OBJECT_PREFIX = "counterpart_digest:"


class SummarizeCounterpartWorker:
    name: ClassVar[str] = "summarize_counterpart"
    trigger: ClassVar[ScheduleTrigger] = trigger(
        30,
        24 * 7,
        {CognitiveEventKind.BELIEF_FORMED, CognitiveEventKind.BELIEF_SUPERSEDED},
        3,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset(
        {CognitiveEventKind.BELIEF_FORMED, CognitiveEventKind.BELIEF_SUPERSEDED}
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
        del log
        counterparts = sorted(
            projections.get_typed(CounterpartProjection).list_active(),
            key=lambda item: str(item.id),
        )
        counterparts = after_cursor_wrap(
            counterparts,
            str(checkpoint.metadata.get("last_counterpart_id", "")),
            lambda item: item.id,
        )
        belief_projection = projections.get_typed(BeliefProjection)
        emitted = 0
        for counterpart in counterparts:
            ref = counterpart_ref(CounterpartId(str(counterpart.id)))
            beliefs = belief_projection.recall_about(ref)
            source_beliefs = [
                item for item in beliefs if not str(item.object).startswith(DIGEST_OBJECT_PREFIX)
            ]
            old_digest = _active_digest(beliefs, str(counterpart.id))
            source_ids = [str(item.id) for item in source_beliefs]
            if not _should_digest(source_ids, old_digest, config):
                if coordinator.yield_to_higher_priority():
                    return report(
                        self.name,
                        checkpoint,
                        inspected=len(counterparts),
                        emitted=emitted,
                        yielded=True,
                        metadata={"last_counterpart_id": str(counterpart.id)},
                    )
                continue
            content = _digest_content(source_beliefs)
            digest_id = stable_id("belief:digest", str(counterpart.id), source_ids, content)
            if old_digest is None or str(old_digest.id) != digest_id:
                digest = active_belief(
                    digest_id,
                    content,
                    about=[ref],
                    object_=f"{DIGEST_OBJECT_PREFIX}{counterpart.id}",
                    cognitive_type=CognitiveType.CONCEPT,
                    confidence=0.7,
                    sources=belief_source_refs(source_ids),
                    held_since=source_beliefs[-1].held_since,
                )
                formed = emit_projected(
                    emitter,
                    projections,
                    CognitiveEventKind.BELIEF_FORMED,
                    config=config,
                    payload={"belief": digest.to_record(), "digest_source_ids": source_ids},
                    rationale="Formed deterministic counterpart digest belief.",
                )
                emitted += 1 if formed is not None or getattr(config, "dry_run", False) else 0
                if old_digest is not None:
                    superseded = emit_projected(
                        emitter,
                        projections,
                        CognitiveEventKind.BELIEF_SUPERSEDED,
                        config=config,
                        payload={
                            "old_belief_id": str(old_digest.id),
                            "new_belief_id": digest_id,
                            "reason": "counterpart_digest_refreshed",
                        },
                        rationale="Superseded previous counterpart digest.",
                    )
                    emitted += (
                        1 if superseded is not None or getattr(config, "dry_run", False) else 0
                    )
            if coordinator.yield_to_higher_priority():
                return report(
                    self.name,
                    checkpoint,
                    inspected=len(counterparts),
                    emitted=emitted,
                    yielded=True,
                    metadata={"last_counterpart_id": str(counterpart.id)},
                )
        return report(
            self.name,
            checkpoint,
            inspected=len(counterparts),
            emitted=emitted,
            metadata={},
        )


def _active_digest(beliefs: Sequence[Belief], counterpart_id: str) -> Belief | None:
    object_id = f"{DIGEST_OBJECT_PREFIX}{counterpart_id}"
    digests = [belief for belief in beliefs if str(belief.object) == object_id]
    return max(digests, key=lambda item: str(item.held_since)) if digests else None


def _should_digest(source_ids: list[str], old_digest: Belief | None, config: object) -> bool:
    if old_digest is None:
        return len(source_ids) >= int(getattr(config, "counterpart_digest_min_beliefs", 5))
    old_source_ids = {source.id for source in old_digest.sources if source.kind == "belief"}
    new_count = len(set(source_ids) - old_source_ids)
    return new_count >= int(getattr(config, "counterpart_digest_min_new_beliefs", 3))


def _digest_content(beliefs: Sequence[Belief]) -> str:
    ordered = sorted(
        beliefs,
        key=lambda item: (item.cognitive_type.value, -float(item.confidence), str(item.id)),
    )
    parts = [
        f"{belief.cognitive_type.value}: {belief.content}"
        for belief in ordered[:8]
    ]
    return "Counterpart digest: " + " | ".join(parts)
