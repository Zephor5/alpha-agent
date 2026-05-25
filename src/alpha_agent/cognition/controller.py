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
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import (
    StrategyProjection,
    strategy_applies_to_counterpart,
)
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.render.build_view import build_view, situation_from_ref
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
        self.reflector = reflector or ReflectorL1(self.projections)
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
        active_strategies = self._active_strategies(perceived.value.from_counterpart)
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
            strategies=active_strategies,
        )
        self._apply_projection(interpreted.event)
        judged = self.judger.judge(
            interpreted.value,
            subject.value_lens,
            situation=context_window.situation_at,
            thread_id=thread_id,
            emitter=self.emitter,
            tick_id=tick_id,
            causal_parent=interpreted.event.id,
        )
        self._apply_projection(judged.event)
        procedures = []
        if not _disable_procedure_match(active_strategies, interpreted.value.source_text):
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
        view = build_view(
            thread_id=thread_id,
            situation=situation_from_ref(context_window.situation_at),
            projections=self.projections,
            window=context_window,
            recalled_beliefs=recalled,
            matched_procedures=procedures,
            current_query=str(decided.value.payload.get("message", ""))
            if isinstance(decided.value.payload, dict)
            else None,
        )
        acted = self.effector.execute(
            decided.value,
            view,
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
            perceived.value,
            attended.value,
            interpreted.value,
            judged.value,
            decided.value,
            acted.value,
            feedback.value,
            subject,
            context_window,
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
            interpretation=interpreted.value,
            strategies=active_strategies,
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

    def _active_strategies(self, counterpart: Any) -> list[Any]:
        try:
            projection = self.projections.get_typed(StrategyProjection)
        except KeyError:
            return []
        return [
            strategy
            for strategy in projection.active()
            if strategy_applies_to_counterpart(strategy, counterpart)
        ]


def _disable_procedure_match(strategies: list[Any], source_text: str) -> bool:
    text = source_text.casefold()
    for strategy in strategies:
        if strategy.name != "disable_auto_procedure_match_for_trigger":
            continue
        if strategy.target_stages and "decide" not in strategy.target_stages:
            continue
        trigger = str(strategy.payload.get("trigger") or "").casefold()
        if not trigger or trigger in text:
            return True
    return False


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
    store = getattr(event_log, "store", None)
    if store is not None:
        registry.register(CounterpartProjection(store))
    registry.register(
        ProcedureProjection(
            store,
            event_log=event_log,
            auto_rebuild=True,
        )
    )
    registry.register(ContextWindowProjection(event_log))
    registry.register(
        ReflectionProjection(
            store,
            event_log=event_log,
            auto_rebuild=True,
        )
    )
    registry.register(
        StrategyProjection(
            store,
            event_log=event_log,
            auto_rebuild=True,
        )
    )
    return registry
