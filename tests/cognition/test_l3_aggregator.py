from __future__ import annotations

from pathlib import Path

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CognitiveType,
    CounterpartId,
    ExpectedFeedback,
    Instant,
    NLStatement,
    Procedure,
    ProcedureId,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    Severity,
    Step,
    StrategyId,
    StrategyOverride,
    TriggerPattern,
    counterpart_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import StrategyProjection
from alpha_agent.cognition.reflectors.l3_aggregators import (
    AggregationWindow,
    CapabilitiesAggregator,
    FailureModesAggregator,
    InteractionPatternsAggregator,
    PreferredStrategiesAggregator,
    StablePreferencesAggregator,
    TradeoffAggregator,
)
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, counterpart_payload, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_l3_aggregators_derive_self_model_fields(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    window = AggregationWindow(
        since=Instant("2025-12-01T00:00:00+00:00"),
        until=Instant("2026-01-31T00:00:00+00:00"),
    )
    subject = subject_ref(SUBJECT_SELF)
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.PROCEDURE_LEARNED,
        {
            "procedure": Procedure(
                id=ProcedureId("procedure:summarize"),
                trigger=TriggerPattern("summarize"),
                steps=[Step("read"), Step("answer")],
                expected_outcome=NLStatement("summary sent"),
                learned_from=[],
                success_count=5,
                failure_count=1,
                confidence=0.85,
            ).to_record()
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.REFLECTED,
        {
            "tick_id": "tick-1",
            "reflections": [
                Reflection(
                    id=ReflectionId("reflection:one"),
                    level="L1",
                    kind=ReflectionKind("feedback-surprise"),
                    severity=Severity("warning"),
                    target=ReflectionTarget("role:user"),
                    finding=NLStatement("missed expectation"),
                    suggested_remedy=ExpectedFeedback("ask for confirmation"),
                    created_at=Instant("2026-01-01T00:00:00+00:00"),
                ).to_record(),
                Reflection(
                    id=ReflectionId("reflection:two"),
                    level="L1",
                    kind=ReflectionKind("feedback-surprise"),
                    severity=Severity("info"),
                    target=ReflectionTarget("role:user"),
                    finding=NLStatement("missed expectation again"),
                    suggested_remedy=ExpectedFeedback("ask for confirmation"),
                    created_at=Instant("2026-01-01T00:01:00+00:00"),
                ).to_record(),
            ],
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {
            "strategy": StrategyOverride(
                id=StrategyId("strategy:l2-short"),
                name="require_confirm_before_novel_form",
                payload={},
                target_stages=["interpret"],
                set_by="reflector_l2",
                set_at=Instant("2026-01-01T00:00:00+00:00"),
                valid_until=Instant("2026-01-05T00:00:00+00:00"),
            ).to_record()
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_EXPIRED,
        {"strategy_id": "strategy:l2-short"},
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {
            "strategy": StrategyOverride(
                id=StrategyId("strategy:l2-confirm"),
                name="require_explicit_confirm_on_contradiction",
                payload={},
                target_stages=["revise"],
                set_by="reflector_l2",
                set_at=Instant("2026-01-01T00:00:00+00:00"),
                valid_until=Instant("2026-01-02T00:00:00+00:00"),
            ).to_record()
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.STRATEGY_CHANGED,
        {
            "strategy": StrategyOverride(
                id=StrategyId("strategy:l2-long"),
                name="freeze_lens_learning_for_24h",
                payload={},
                target_stages=["revise"],
                set_by="reflector_l2",
                set_at=Instant("2026-01-02T00:00:00+00:00"),
                valid_until=Instant("2026-01-04T00:00:00+00:00"),
            ).to_record()
        },
    )
    stable_belief = belief(
        "belief:stable-value",
        "The agent values concise truthful answers.",
        confidence=0.9,
    ).to_record()
    stable_belief["cognitive_type"] = CognitiveType.VALUE.value
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.BELIEF_FORMED,
        {"belief": stable_belief},
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.BELIEF_SUPERSEDED,
        {
            "old_belief_id": "belief:old",
            "new_belief_id": "belief:new",
            "decisive_value_kinds": ["honesty"],
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
        counterpart_payload("counterpart:user-a"),
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.PERCEIVED,
        {
            "tick_id": "tick-user-a",
            "from_counterpart": counterpart_ref(CounterpartId("counterpart:user-a")).to_record(),
        },
    )
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.RECEIVED_FEEDBACK,
        {"tick_id": "tick-user-a", "matched_expected": True},
    )

    assert CapabilitiesAggregator().compute(subject, log, projections, window) == {
        "summarize": "confidence=0.850;success=5;failure=1"
    }
    assert FailureModesAggregator().compute(subject, log, projections, window) == [
        "feedback-surprise:count=2"
    ]
    preferred = PreferredStrategiesAggregator().compute(subject, log, projections, window)
    stable = StablePreferencesAggregator().compute(subject, log, projections, window)
    assert [item.to_record() for item in preferred] == [
        {"kind": "strategy", "id": "strategy:l2-long"},
        {"kind": "strategy", "id": "strategy:l2-confirm"},
        {"kind": "strategy", "id": "strategy:l2-short"},
    ]
    assert [item.to_record() for item in stable] == [
        {"kind": "belief", "id": "belief:stable-value"}
    ]
    assert TradeoffAggregator().compute(subject, log, projections, window) == [
        "honesty:count=1"
    ]
    assert InteractionPatternsAggregator().compute(subject, log, projections, window) == {
        "user": "ticks=1;feedback=1;success_rate=1.000;reflections=2"
    }


def _runtime(tmp_path: Path):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    projections = ProjectionRegistry()
    projections.register(ProcedureProjection(store))
    projections.register(ReflectionProjection(store))
    projections.register(StrategyProjection(store))
    projections.register(BeliefProjection(store))
    projections.register(CounterpartProjection(store))
    return store, log, projections, emitter


def _emit_apply(
    emitter: EventEmitter,
    projections: ProjectionRegistry,
    kind: CognitiveEventKind,
    payload: dict[str, object],
):
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event
