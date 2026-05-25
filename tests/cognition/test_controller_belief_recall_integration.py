from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import (
    Belief,
    ContextWindow,
    CounterpartId,
    Instant,
    Stimulus,
    StimulusKind,
    ThreadId,
    counterpart_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.stages.interpret import Interpreter
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.tools.registry import ToolRegistry
from tests.cognition.test_belief_projection_apply import belief


def test_controller_recalls_after_context_window_supplies_counterpart() -> None:
    log = InMemoryEventLog()
    recalled_belief = belief("belief:ctx", "User A prefers Python.")
    belief_projection = _RecordingBeliefProjection(recalled_belief)
    interpreter = _RecordingInterpreter()
    registry = ProjectionRegistry()
    registry.register(SubjectProjection(log))
    registry.register(belief_projection)
    registry.register(ProcedureProjection())
    registry.register(ContextWindowProjection(log))
    controller = CognitiveController(
        event_log=log,
        projections=registry,
        llm=_StaticProvider(),
        tools=ToolRegistry(),
        interpreter=interpreter,
    )
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))

    controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=counterpart,
            payload="hello",
            thread_id=ThreadId.from_session("s1"),
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=ThreadId.from_session("s1"),
    )

    assert belief_projection.last_params is not None
    assert belief_projection.last_params.counterpart == counterpart
    assert interpreter.last_window is not None
    assert [item.id for item in interpreter.last_window.recalled] == ["belief:ctx"]


class _RecordingBeliefProjection(BeliefProjection):
    def __init__(self, recalled_belief: Belief):
        super().__init__()
        self.recalled_belief = recalled_belief
        self.last_params: BeliefRecallParams | None = None

    def recall(
        self,
        params: BeliefRecallParams | object,
        **_kwargs: object,
    ) -> list[Belief]:
        assert isinstance(params, BeliefRecallParams)
        self.last_params = params
        return [self.recalled_belief]


class _RecordingInterpreter(Interpreter):
    def __init__(self) -> None:
        self.last_window: ContextWindow | None = None

    def interpret(self, *args, **kwargs):
        self.last_window = args[1]
        return super().interpret(*args, **kwargs)


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
