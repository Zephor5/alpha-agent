from __future__ import annotations

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    NLStatement,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
)
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory


def test_reset_and_replay_rebuilds_equivalent_reflection_view(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = ReflectionProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    reflection = _reflection("reflection:1", "low-confidence-high-stakes", "warning")
    projection.apply(
        emit(
            log,
            CognitiveEventKind.REFLECTED,
            payload={
                "turn_id": "turn_1",
                "session_id": "s1",
                "reflection_count": 1,
                "reflection_ids": [str(reflection.id)],
                "targets": [{"kind": "reflection", "id": str(reflection.id)}],
                "reflections": [reflection.to_record()],
            },
            event_ids=event_ids,
            clock=clock,
        )
    )
    before = [item.to_record() for item in projection.list_recent(last=10)]

    registry = ProjectionRegistry()
    rebuilt = ReflectionProjection(store)
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()

    assert [item.to_record() for item in rebuilt.list_recent(last=10)] == before
    ProjectionRunner(log, registry).replay_all()
    assert [item.to_record() for item in rebuilt.list_recent(last=10)] == before


def test_reflection_projection_filters_by_severity_kind_and_target(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = ReflectionProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    warning = _reflection("reflection:1", "unsupported-tool-call", "warning")
    info = _reflection("reflection:2", "feedback-surprise", "info", target="loop_run:turn_1")
    projection.apply(
        emit(
            log,
            CognitiveEventKind.REFLECTED,
            payload={
                "turn_id": "turn_1",
                "session_id": "s1",
                "reflection_count": 2,
                "reflection_ids": [str(warning.id), str(info.id)],
                "targets": [
                    {"kind": "reflection", "id": str(warning.id)},
                    {"kind": "reflection", "id": str(info.id)},
                ],
                "reflections": [warning.to_record(), info.to_record()],
            },
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert [item.id for item in projection.by_severity("warning")] == ["reflection:1"]
    assert [item.id for item in projection.by_kind("feedback-surprise")] == ["reflection:2"]
    assert [item.id for item in projection.for_target("loop_run", "turn_1")] == [
        "reflection:2"
    ]
    assert [item.id for item in projection.list_recent(last=1)] == ["reflection:2"]


def _reflection(
    reflection_id: str,
    kind: str,
    severity: str,
    *,
    target: str = "decision:decision:1",
) -> Reflection:
    return Reflection(
        id=ReflectionId(reflection_id),
        level="L1",
        kind=ReflectionKind(kind),
        severity=Severity(severity),
        target=ReflectionTarget(target),
        finding=NLStatement("finding"),
        suggested_remedy=RemedyHint("remedy"),
        created_at=Instant(f"2026-01-01T00:00:0{reflection_id[-1]}+00:00"),
    )
