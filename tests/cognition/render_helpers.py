from __future__ import annotations

from typing import Any, cast

from alpha_agent.cognition.models import (
    ContextWindow,
    Counterpart,
    CounterpartId,
    CounterpartRole,
    Instant,
    Perception,
    PerceptionId,
    Relationship,
    Situation,
    SituationId,
    StimulusKind,
    StyleHint,
    Subject,
    ThreadId,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.render import CognitionView


def perception(raw: str) -> Perception:
    return Perception(
        id=PerceptionId("perception:1"),
        source_kind=StimulusKind.USER_MESSAGE,
        from_counterpart=None,
        raw=raw,
        surface_intent=[],
        raised_entities=[],
        subject=subject_ref(Subject().id),
        situation=situation_ref(SituationId("situation:test")),
        received_at=Instant("2026-01-01T00:00:00+00:00"),
    )


def window(*, raw: str = "hello") -> ContextWindow:
    subject = Subject()
    situation = Situation(id=SituationId("situation:test"))
    return ContextWindow(
        thread_id=ThreadId.from_session("s1"),
        counterpart=None,
        foreground=[perception(raw)],
        background=None,
        recalled=[],
        recent_judgments=[],
        matched_procedures=[],
        subject_at=subject_ref(subject.id),
        situation_at=situation_ref(situation.id),
        assembled_at=Instant("2026-01-01T00:00:00+00:00"),
    )


def counterpart(
    *,
    role: CounterpartRole = CounterpartRole.USER,
    trust_level: float = 0.8,
    style_value: str = "direct",
) -> Counterpart:
    return Counterpart(
        id=CounterpartId("counterpart:user-a"),
        role=role,
        identity={"display_name": "User A"},
        relationship=Relationship(),
        service_contract=[],
        trust_level=trust_level,
        communication_style=[StyleHint(kind="tone", value=style_value, confidence=0.9)],
        first_seen_at=Instant("2026-01-01T00:00:00+00:00"),
        last_interaction_at=Instant("2026-01-01T00:00:00+00:00"),
    )


def view(**kwargs: Any) -> CognitionView:
    subject = Subject()
    situation = Situation(id=SituationId("situation:test"))
    base = {
        "subject": subject,
        "counterpart": None,
        "situation": situation,
        "window": window(),
        "assembled_at": Instant("2026-01-01T00:00:00+00:00"),
        "current_query": "hello",
    }
    base.update(kwargs)
    return CognitionView(**cast(Any, base))


def counterpart_window() -> ContextWindow:
    base = window()
    return ContextWindow.from_record(
        {
            **base.to_record(),
            "counterpart": counterpart_ref(CounterpartId("counterpart:user-a")).to_record(),
        },
    )
