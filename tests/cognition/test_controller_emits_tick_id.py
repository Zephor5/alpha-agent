from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import Instant, Stimulus, StimulusKind, ThreadId
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.tools.default import build_tool_registry


def test_controller_emits_same_tick_id_on_all_reactive_events() -> None:
    log = InMemoryEventLog()
    thread_id = ThreadId.from_session("s1")
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=_StaticProvider(),
        tools=build_tool_registry(),
    )

    result = controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=None,
            payload="same tick",
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=thread_id,
    )

    tick_ids = {event.payload["tick_id"] for event in log.iter()}
    assert tick_ids == {result.debug["tick_id"]}


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
