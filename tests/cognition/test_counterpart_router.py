from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    Stimulus,
    StimulusKind,
    ThreadId,
)
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.runtime.counterpart_router import CounterpartRouter
from alpha_agent.tools.registry import ToolRegistry


def test_counterpart_router_first_observed_no_duplicate_and_perception_source() -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log)
    router = CounterpartRouter(log)
    source_metadata = {"platform": "test", "user_id": "u1"}

    first = router.upsert_from_source_metadata(source_metadata, emitter=emitter)
    second = router.upsert_from_source_metadata(source_metadata, emitter=emitter)

    assert first == second
    assert [
        event.kind for event in log.iter(kinds=[CognitiveEventKind.COUNTERPART_FIRST_OBSERVED])
    ] == [CognitiveEventKind.COUNTERPART_FIRST_OBSERVED]

    thread_id = ThreadId.from_session("s1", source_metadata)
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=_StaticProvider(),
        tools=ToolRegistry(),
        emitter=emitter,
    )
    controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=first,
            payload="from user",
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=thread_id,
    )

    perceived = [event for event in log.iter(kinds=[CognitiveEventKind.PERCEIVED])][0]
    assert perceived.payload["from_counterpart"] == first.to_record()


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
