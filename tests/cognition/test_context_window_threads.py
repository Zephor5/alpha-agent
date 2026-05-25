from __future__ import annotations

import pytest

from alpha_agent.cognition.models import Instant, Stimulus, StimulusKind, Subject, ThreadId
from alpha_agent.cognition.threads import StimulusRouter


def test_stimulus_router_routes_conversation_stimuli_to_session_threads() -> None:
    stimulus = Stimulus(
        kind=StimulusKind.USER_MESSAGE,
        source=None,
        payload={"source_metadata": {"platform": "test", "user_id": "u1"}},
        thread_id=ThreadId.from_session("placeholder"),
        received_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    routed = StimulusRouter.route(stimulus, session_id="s1")

    assert routed == ThreadId.from_session(
        "s1",
        {"platform": "test", "user_id": "u1"},
    )


def test_stimulus_router_routes_internal_stimuli_to_cognition_threads() -> None:
    subject = Subject()
    self_signal = Stimulus(
        kind=StimulusKind.SELF_SIGNAL,
        source=None,
        payload={"goal_id": "Review Plan"},
        thread_id=ThreadId.from_session("placeholder"),
        received_at=Instant("2026-01-01T00:00:00+00:00"),
    )
    clock_tick = Stimulus(
        kind=StimulusKind.CLOCK_TICK,
        source=None,
        payload={},
        thread_id=ThreadId.from_session("placeholder"),
        received_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    assert StimulusRouter.route(self_signal, subject_id=subject.id) == ThreadId.cognition(
        subject.id,
        "Review Plan",
    )
    assert StimulusRouter.route(clock_tick, subject_id=subject.id) == ThreadId.cognition(
        subject.id,
        "clock",
    )


def test_stimulus_router_requires_session_for_conversation_stimuli() -> None:
    stimulus = Stimulus(
        kind=StimulusKind.WEBHOOK,
        source=None,
        payload={},
        thread_id=ThreadId.from_session("placeholder"),
        received_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    with pytest.raises(ValueError, match="session_id is required"):
        StimulusRouter.route(stimulus)
