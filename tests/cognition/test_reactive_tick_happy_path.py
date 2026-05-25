from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    Stimulus,
    StimulusKind,
    ThreadId,
)
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.tools.registry import ToolRegistry


def test_reactive_tick_emits_nine_events_with_causal_chain() -> None:
    log = InMemoryEventLog()
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=_StaticProvider("hello back"),
        tools=ToolRegistry(),
    )

    result = controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=None,
            payload="hello",
            thread_id=ThreadId.from_session("s1"),
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=ThreadId.from_session("s1"),
    )

    assert result.response_text == "hello back"
    events = list(log.iter())
    assert [event.kind for event in events] == [
        CognitiveEventKind.PERCEIVED,
        CognitiveEventKind.ATTENDED,
        CognitiveEventKind.INTERPRETED,
        CognitiveEventKind.JUDGED,
        CognitiveEventKind.DECIDED,
        CognitiveEventKind.ACTED,
        CognitiveEventKind.RECEIVED_FEEDBACK,
        CognitiveEventKind.REFLECTED,
        CognitiveEventKind.REVISED,
    ]
    assert events[0].causal_parents == []
    for previous, current in zip(events[:-1], events[1:], strict=True):
        assert current.causal_parents == [previous.id]


class _StaticProvider:
    name = "static"

    def __init__(self, response: str):
        self.response = response

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content=self.response, model="test", provider=self.name)
