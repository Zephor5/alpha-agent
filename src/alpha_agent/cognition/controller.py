"""Reactive cognition controller."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import Stimulus, ThreadId
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.stages import (
    Attender,
    Decider,
    Effector,
    FeedbackReader,
    Interpreter,
    Judger,
    Perceiver,
    ReflectorL1,
    Reviser,
)
from alpha_agent.cognition.stages.types import Outcome
from alpha_agent.llm.base import LLMProvider
from alpha_agent.tools.registry import ToolRegistry


@dataclass(frozen=True)
class LoopResult:
    response_text: str
    decision: Any
    outcome: Outcome
    reflections: list[Any]
    debug: dict[str, Any] = field(default_factory=dict)


class CognitiveController:
    """Run one complete 9-step reactive cognition tick."""

    def __init__(
        self,
        event_log: EventLog,
        projections: ProjectionRegistry | None,
        llm: LLMProvider,
        tools: ToolRegistry,
        *,
        emitter: EventEmitter | None = None,
        effector: Effector | None = None,
        perceiver: Perceiver | None = None,
        attender: Attender | None = None,
        interpreter: Interpreter | None = None,
        judger: Judger | None = None,
        decider: Decider | None = None,
        feedback_reader: FeedbackReader | None = None,
        reflector: ReflectorL1 | None = None,
        reviser: Reviser | None = None,
    ):
        self.event_log = event_log
        self.projections = projections or default_projection_registry(event_log)
        self.emitter = emitter or EventEmitter(event_log)
        self.perceiver = perceiver or Perceiver()
        self.attender = attender or Attender()
        self.interpreter = interpreter or Interpreter()
        self.judger = judger or Judger()
        self.decider = decider or Decider()
        self.effector = effector or Effector(llm_provider=llm, tool_registry=tools)
        self.feedback_reader = feedback_reader or FeedbackReader()
        self.reflector = reflector or ReflectorL1()
        self.reviser = reviser or Reviser()

    def reactive_tick(self, stimulus: Stimulus, thread_id: ThreadId) -> LoopResult:
        tick_id = str(uuid.uuid4())
        subject = self.projections.get_typed(SubjectProjection).current()

        perceived = self.perceiver.perceive(
            stimulus,
            subject,
            emitter=self.emitter,
            tick_id=tick_id,
        )
        self._apply_projection(perceived.event)
        attended = self.attender.focus(
            perceived.value,
            subject,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=perceived.event.id,
        )
        self._apply_projection(attended.event)

        context_projection = self.projections.get_typed(ContextWindowProjection)
        context_window = context_projection.get(
            thread_id,
            subject,
        )
        belief_projection = self.projections.get_typed(BeliefProjection)
        recalled = belief_projection.recall(
            BeliefRecallParams(focus=attended.value, counterpart=context_window.counterpart)
        )
        context_window = context_projection.attach_recalled(context_window, recalled)
        interpreted = self.interpreter.interpret(
            attended.value,
            context_window,
            recalled,
            subject,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=attended.event.id,
        )
        self._apply_projection(interpreted.event)
        judged = self.judger.judge(
            interpreted.value,
            subject.value_lens,
            situation=context_window.situation_at,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=interpreted.event.id,
        )
        self._apply_projection(judged.event)
        procedures = self.projections.get_typed(ProcedureProjection).match(
            interpreted.value,
            subject=subject,
            window=context_window,
        )
        decided = self.decider.decide(
            judged.value,
            procedures,
            subject,
            context_window,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=judged.event.id,
        )
        self._apply_projection(decided.event)
        acted = self.effector.execute(
            decided.value,
            context_window,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=decided.event.id,
        )
        self._apply_projection(acted.event)
        feedback = self.feedback_reader.compare(
            decided.value,
            acted.value,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=acted.event.id,
        )
        self._apply_projection(feedback.event)
        reflected = self.reflector.audit(
            feedback.value,
            acted.value,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=feedback.event.id,
        )
        self._apply_projection(reflected.event)
        revised = self.reviser.derive(
            feedback.value,
            reflected.value,
            judged.value,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=reflected.event.id,
        )
        self._apply_projection(revised.event)

        return LoopResult(
            response_text=acted.value.text or "",
            decision=decided.value,
            outcome=acted.value,
            reflections=reflected.value,
            debug={
                **acted.value.debug,
                "busy": False,
                "tick_id": tick_id,
                "event_ids": [
                    perceived.event.id,
                    attended.event.id,
                    interpreted.event.id,
                    judged.event.id,
                    decided.event.id,
                    acted.event.id,
                    feedback.event.id,
                    reflected.event.id,
                    revised.event.id,
                ],
            },
        )

    def _apply_projection(self, event: Any) -> None:
        for projection in self.projections.all():
            if event.kind in projection.handles:
                projection.apply(event)


def default_projection_registry(event_log: EventLog) -> ProjectionRegistry:
    registry = ProjectionRegistry()
    registry.register(SubjectProjection(event_log))
    registry.register(
        BeliefProjection(
            getattr(event_log, "store", None),
            event_log=event_log,
            auto_rebuild=True,
        )
    )
    registry.register(ProcedureProjection())
    registry.register(ContextWindowProjection(event_log))
    return registry
