from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import Instant, Stimulus, StimulusKind, ThreadId
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.tools.registry import ToolRegistry


def test_stub_projections_return_expected_phase_02_shapes() -> None:
    log = InMemoryEventLog()
    registry = default_projection_registry(log)
    controller = CognitiveController(
        event_log=log,
        projections=registry,
        llm=_StaticProvider(),
        tools=ToolRegistry(),
    )
    thread_id = ThreadId.from_session("s1")
    for message in ["one", "two", "three"]:
        controller.reactive_tick(
            stimulus=Stimulus(
                kind=StimulusKind.USER_MESSAGE,
                source=None,
                payload=message,
                thread_id=thread_id,
                received_at=Instant("2026-01-01T00:00:00+00:00"),
            ),
            thread_id=thread_id,
        )

    subject = registry.get_typed(SubjectProjection).current()
    window = ContextWindowProjection(log, recent_limit=2).get(thread_id, subject)

    assert [perception.raw for perception in window.foreground] == ["two", "three"]
    assert window.background is None
    assert window.recalled == []
    assert registry.get_typed(BeliefProjection).status == "stub"
    assert registry.get_typed(BeliefProjection).recall("anything") == []
    assert registry.get_typed(ProcedureProjection).status == "stub"
    assert registry.get_typed(ProcedureProjection).match("anything") == []


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
