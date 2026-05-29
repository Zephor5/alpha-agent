"""Perceive stage for reactive ticks."""

from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartId,
    CounterpartRef,
    NLStatement,
    Perception,
    PerceptionId,
    Reference,
    Situation,
    SituationId,
    SocialContext,
    Stimulus,
    Subject,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.stages._payload import digest_payload
from alpha_agent.cognition.stages.types import Emitted
from alpha_agent.utils.ids import new_id


class Perceiver:
    """Turn a raw stimulus into a subject-scoped perception."""

    def perceive(
        self,
        stimulus: Stimulus,
        subject: Subject,
        *,
        emitter: EventEmitter,
        tick_id: str,
    ) -> Emitted[Perception]:
        situation = Situation(
            id=SituationId(new_id("sit")),
            social=SocialContext(
                present_counterparts=[stimulus.source] if stimulus.source is not None else []
            ),
        )
        perception = Perception(
            id=PerceptionId(new_id("perception")),
            source_kind=stimulus.kind,
            from_counterpart=stimulus.source,
            raw=stimulus.payload,
            surface_intent=[],
            raised_entities=_raised_entities(stimulus.source),
            subject=subject_ref(subject.id),
            situation=situation_ref(situation.id),
            received_at=stimulus.received_at,
        )
        event = emitter.emit(
            CognitiveEventKind.PERCEIVED,
            situation=perception.situation,
            outputs=[Reference("perception", str(perception.id)), perception.situation],
            rationale=NLStatement("Stimulus perceived for reactive tick."),
            payload={
                "tick_id": tick_id,
                "stimulus_kind": stimulus.kind.value,
                "payload_digest": digest_payload(stimulus.payload),
                "thread_id": stimulus.thread_id.to_record(),
                "perception": perception.to_record(),
                "source_refs": [ref.to_record() for ref in stimulus.source_refs],
                "session_id": _source_ref_id(stimulus.source_refs, "session"),
                "user_message_id": _source_ref_id(stimulus.source_refs, "session_message"),
                "from_counterpart": stimulus.source.to_record()
                if stimulus.source is not None
                else None,
                "present_counterparts": [
                    item.to_record() for item in situation.social.present_counterparts
                ],
            },
        )
        return Emitted(perception, event)


def _raised_entities(source: CounterpartRef | None) -> list[Reference]:
    if source is None:
        return []
    return [counterpart_ref(CounterpartId(source.id))]


def _source_ref_id(source_refs: list[Reference], kind: str) -> str | None:
    return next((ref.id for ref in source_refs if ref.kind == kind), None)
