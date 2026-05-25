from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind, EventId
from alpha_agent.cognition.stages.reflect import ReflectorL1
from tests.cognition.reflector_helpers import context, judgment


def test_reflect_stage_emits_reflected_event_even_when_no_reflections_fire() -> None:
    log = InMemoryEventLog()

    emitted = ReflectorL1().audit(
        context(),
        emitter=EventEmitter(log, id_factory=_id_factory(), clock=lambda: "2026-01-01"),
        causal_parent=EventId("event:feedback"),
    )

    assert emitted.value == []
    events = list(log.iter())
    assert [event.kind for event in events] == [CognitiveEventKind.REFLECTED]
    assert events[0].payload["reflection_count"] == 0
    assert events[0].payload["reflection_ids"] == []


def test_reflect_stage_emits_bias_detected_event_for_each_reflection() -> None:
    log = InMemoryEventLog()
    ctx = context(judgments=[judgment(confidence=0.3, value_weights={"existence": 0.8})])

    emitted = ReflectorL1().audit(
        ctx,
        emitter=EventEmitter(log, id_factory=_id_factory(), clock=lambda: "2026-01-01"),
        causal_parent=EventId("event:feedback"),
    )

    events = list(log.iter())
    assert [event.kind for event in events] == [
        CognitiveEventKind.REFLECTED,
        CognitiveEventKind.BIAS_DETECTED,
    ]
    assert events[0].payload["reflection_count"] == 1
    assert events[0].payload["reflection_ids"] == [str(emitted.value[0].id)]
    assert events[0].payload["reflections"] == [emitted.value[0].to_record()]
    assert events[1].payload["reflection_id"] == str(emitted.value[0].id)
    assert events[1].payload["target"] == {"kind": "judgment", "id": "judgment:1"}


def _id_factory():
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"event:{counter}"

    return next_id
