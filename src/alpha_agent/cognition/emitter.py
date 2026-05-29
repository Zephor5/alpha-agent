"""Convenience wrapper for emitting cognitive events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    ActorRef,
    CognitiveEvent,
    CognitiveEventKind,
    EventId,
    Instant,
    NLStatement,
    Reference,
    SituationRef,
    SubjectRef,
    actor_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.payload_contract import validate_event_payload
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class EventEmitter:
    """Fill default cognitive event fields before appending to an EventLog."""

    def __init__(
        self,
        log: EventLog,
        *,
        subject: SubjectRef | None = None,
        subject_version: int = 1,
        actor: ActorRef | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], str] | None = None,
    ):
        self.log = log
        self.subject = subject or subject_ref(SUBJECT_SELF)
        self.subject_version = subject_version
        self.actor = actor or actor_ref("cognition")
        self.id_factory = id_factory or (lambda: new_id("cogevt"))
        self.clock = clock or utc_now_iso

    def emit(
        self,
        kind: CognitiveEventKind,
        *,
        situation: SituationRef | None = None,
        inputs: list[Reference] | None = None,
        outputs: list[Reference] | None = None,
        rationale: NLStatement | str = "",
        actor: ActorRef | None = None,
        causal_parents: list[EventId] | None = None,
        payload: dict[str, Any] | None = None,
        timestamp: Instant | None = None,
    ) -> CognitiveEvent:
        event_payload = payload or {}
        validate_event_payload(kind, event_payload)
        event = CognitiveEvent(
            id=EventId(self.id_factory()),
            kind=kind,
            subject=self.subject,
            subject_version=self.subject_version,
            situation=situation,
            inputs=inputs or [],
            outputs=outputs or [],
            rationale=NLStatement(str(rationale)),
            timestamp=timestamp or Instant(self.clock()),
            actor=actor or self.actor,
            causal_parents=causal_parents or [],
            payload=event_payload,
        )
        self.log.append(event)
        return event
