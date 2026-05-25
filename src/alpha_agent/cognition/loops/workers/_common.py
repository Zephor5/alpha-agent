"""Shared deterministic helpers for consolidation workers."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import timedelta
from typing import Any

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import ScheduleTrigger, WorkerCheckpoint, WorkerReport
from alpha_agent.cognition.models import (
    Applicability,
    Belief,
    BeliefId,
    CognitiveEvent,
    CognitiveEventKind,
    CognitiveType,
    EvidenceRef,
    Instant,
    Lifecycle,
    NLStatement,
    Reference,
    Role,
    SituationId,
    UpdatePolicy,
    belief_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.value.profile_derivation import derive_value_profile


def trigger(
    minutes: int,
    max_hours: int | None,
    watches: set[CognitiveEventKind],
    min_new_events: int,
) -> ScheduleTrigger:
    return ScheduleTrigger(
        min_interval=timedelta(minutes=minutes),
        max_interval=timedelta(hours=max_hours) if max_hours is not None else None,
        watches=frozenset(watches),
        min_new_events=min_new_events,
    )


def emit_projected(
    emitter: EventEmitter,
    projections: ProjectionRegistry,
    kind: CognitiveEventKind,
    *,
    config: object,
    payload: dict[str, Any],
    rationale: str,
    inputs: list[Reference] | None = None,
    outputs: list[Reference] | None = None,
    causal_parents: list[Any] | None = None,
) -> CognitiveEvent | None:
    if bool(getattr(config, "dry_run", False)):
        return None
    event = emitter.emit(
        kind,
        inputs=inputs or [],
        outputs=outputs or [],
        rationale=NLStatement(rationale),
        causal_parents=causal_parents or [],
        payload=payload,
    )
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


def report(
    worker: str,
    checkpoint: WorkerCheckpoint,
    *,
    inspected: int,
    emitted: int,
    notes: list[str] | None = None,
    last_event: CognitiveEvent | None = None,
    yielded: bool = False,
    metadata: dict[str, object] | None = None,
) -> WorkerReport:
    last_processed_event_id = (
        last_event.id if last_event is not None else checkpoint.last_processed_event_id
    )
    return WorkerReport(
        worker=worker,
        inspected=inspected,
        emitted=emitted,
        notes=notes or [],
        yielded_to_higher_priority=yielded,
        new_checkpoint=WorkerCheckpoint(
            worker_name=worker,
            last_run_at=checkpoint.last_run_at,
            last_processed_event_id=last_processed_event_id,
            last_status="yielded" if yielded else "ok",
            metadata=metadata if metadata is not None else checkpoint.metadata,
        ),
    )


def normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).casefold()).strip()


def stable_id(prefix: str, *parts: object) -> str:
    digest = hashlib.sha1(
        json.dumps(parts, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}:{digest}"


def active_belief(
    belief_id: str,
    content: str,
    *,
    about: list[Reference] | None = None,
    object_: str,
    cognitive_type: CognitiveType,
    confidence: float,
    sources: list[EvidenceRef] | None = None,
    held_since: Instant,
) -> Belief:
    return Belief(
        id=BeliefId(belief_id),
        subject=subject_ref(SUBJECT_SELF),
        about=list(about or []),
        object=object_,
        content=NLStatement(content),
        cognitive_type=cognitive_type,
        structure=None,
        sources=list(sources or []),
        confidence=confidence,
        applicability=Applicability("{}"),
        value_profile=derive_value_profile(
            content,
            cognitive_type=cognitive_type,
            entities=about or [],
        ),
        relations=[],
        formed_in=situation_ref(SituationId("situation:consolidation")),
        holder_role=Role("agent"),
        action_orientation=[],
        update_policy=UpdatePolicy("{}"),
        status=Lifecycle("active"),
        held_since=held_since,
    )


def belief_source_refs(belief_ids: list[str]) -> list[EvidenceRef]:
    return [belief_ref(BeliefId(item)) for item in belief_ids]


def latest_event(log: EventLog, kinds: set[CognitiveEventKind]) -> CognitiveEvent | None:
    last = None
    for event in log.iter(kinds=kinds):
        last = event
    return last


def payload_fingerprint(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def after_cursor_wrap[T](items: list[T], cursor: str, key: Any) -> list[T]:
    """Return sorted work after cursor, then wrap to the lower/equal prefix."""

    if not cursor:
        return items
    greater = [item for item in items if str(key(item)) > cursor]
    lower_or_equal = [item for item in items if str(key(item)) <= cursor]
    return greater + lower_or_equal
