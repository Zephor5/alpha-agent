from __future__ import annotations

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.subject import SubjectProjection


def test_default_projections_return_expected_shapes() -> None:
    log = InMemoryEventLog()
    registry = default_projection_registry(log)
    emitter = EventEmitter(log, id_factory=_id_factory(), clock=_clock_factory())
    for index, message in enumerate(["one", "two", "three"], start=1):
        _apply_all(
            registry,
            emitter.emit(
                CognitiveEventKind.PERCEIVED,
                payload=_perceived_payload(index=index, message=message),
            ),
        )

    subject = registry.get_typed(SubjectProjection).current()
    window = ContextWindowProjection(log, recent_limit=2).get("s1", subject)

    assert [perception.raw for perception in window.foreground] == ["two", "three"]
    assert window.background is None
    assert window.recalled == []
    assert registry.get_typed(BeliefProjection).recall(BeliefRecallParams()) == []
    assert registry.get_typed(ProcedureProjection).status == "materialized"
    assert registry.get_typed(ProcedureProjection).match("anything") == []


def _apply_all(registry, event: CognitiveEvent) -> None:
    for projection in registry.all():
        if event.kind in projection.handles:
            projection.apply(event)


def _perceived_payload(*, index: int, message: str) -> dict[str, object]:
    return {
        "turn_id": f"turn_{index}",
        "session_id": "s1",
        "stimulus_kind": "user_message",
        "source": {"kind": "session", "id": "s1"},
        "from_counterpart": None,
        "source_refs": [
            {"kind": "session", "id": "s1"},
            {"kind": "session_message", "id": f"msg_{index}"},
        ],
        "content_digest": f"digest-{index}",
        "content_length": len(message),
        "perception": {
            "id": f"perception:{index}",
            "source_kind": "user_message",
            "from_counterpart": None,
            "raw": message,
            "surface_intent": [],
            "raised_entities": [],
            "subject": {"kind": "subject", "id": "subject:self"},
            "situation": {"kind": "situation", "id": f"situation:{index}"},
            "received_at": "2026-01-01T00:00:00+00:00",
        },
    }


def _id_factory():
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return f"event:{counter}"

    return next_id


def _clock_factory():
    counter = 0

    def now() -> str:
        nonlocal counter
        counter += 1
        return f"2026-01-01T00:00:{counter:02d}+00:00"

    return now
