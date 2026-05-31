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
from alpha_agent.tools.default import build_tool_registry


def test_reactive_tick_emits_nine_events_with_causal_chain() -> None:
    log = InMemoryEventLog()
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=_StaticProvider("hello back"),
        tools=build_tool_registry(),
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

    judged = [event for event in events if event.kind == CognitiveEventKind.JUDGED][0]
    decided = [event for event in events if event.kind == CognitiveEventKind.DECIDED][0]
    acted = [event for event in events if event.kind == CognitiveEventKind.ACTED][0]
    feedback = [
        event for event in events if event.kind == CognitiveEventKind.RECEIVED_FEEDBACK
    ][0]
    revised = [event for event in events if event.kind == CognitiveEventKind.REVISED][0]

    assert judged.payload["claim"] == "hello"
    assert judged.payload["judgments"][0]["claim"] == "hello"
    assert decided.payload["message"] == "hello"
    assert decided.payload["decision"]["payload"]["message"] == "hello"
    assert acted.payload["decision_id"] == decided.payload["decision"]["id"]
    assert acted.payload["tool_call_ids"] == []
    assert acted.payload["response_text_digest"]
    assert feedback.payload["decision_id"] == decided.payload["decision"]["id"]
    assert feedback.payload["acted_event_id"] == str(acted.id)
    assert revised.payload["judgment_ids"] == [judged.payload["judgments"][0]["id"]]
    assert revised.payload["feedback_event_id"] == str(feedback.id)


class _StaticProvider:
    name = "static"

    def __init__(self, response: str):
        self.response = response

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content=self.response, model="test", provider=self.name)
