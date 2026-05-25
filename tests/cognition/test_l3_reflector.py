from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.coordinator import LoopAcquireRequest
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops import (
    CheckpointStore,
    ConsolidationConfig,
    ConsolidationLoop,
    Scheduler,
)
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    LoopPriority,
    NLStatement,
    Procedure,
    ProcedureId,
    Step,
    TriggerPattern,
)
from alpha_agent.cognition.reflectors.l3 import ReflectorL3
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory


def test_reflector_l3_emits_self_model_update_and_throttles(tmp_path: Path) -> None:
    store, log, projections, emitter = _runtime(tmp_path)
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.PROCEDURE_LEARNED,
        {
            "procedure": Procedure(
                id=ProcedureId("procedure:triage"),
                trigger=TriggerPattern("triage"),
                steps=[Step("classify")],
                expected_outcome=NLStatement("issue classified"),
                learned_from=[],
                success_count=3,
                failure_count=0,
                confidence=0.9,
            ).to_record()
        },
    )
    l3 = ReflectorL3()

    first = l3.run_once(
        log=log,
        projections=projections,
        emitter=emitter,
        config=SimpleNamespace(now="2026-01-02T00:00:00+00:00"),
    )
    second = l3.run_once(
        log=log,
        projections=projections,
        emitter=emitter,
        config=SimpleNamespace(now="2026-01-02T00:10:00+00:00"),
    )

    assert first.emitted == 1
    assert second.emitted == 0
    assert second.notes == ["throttled"]
    assert [event.kind for event in log.iter()][-1] == CognitiveEventKind.SELF_MODEL_UPDATED
    assert len(list(log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]))) == 1


def test_reflector_l3_only_emits_self_model_updated(tmp_path: Path) -> None:
    _store, log, projections, emitter = _runtime(tmp_path)
    _emit_apply(
        emitter,
        projections,
        CognitiveEventKind.PROCEDURE_LEARNED,
        {
            "procedure": Procedure(
                id=ProcedureId("procedure:answer"),
                trigger=TriggerPattern("answer"),
                steps=[Step("respond")],
                expected_outcome=NLStatement("answer sent"),
                learned_from=[],
                success_count=2,
                failure_count=0,
                confidence=0.8,
            ).to_record()
        },
    )
    before = list(log.iter())

    ReflectorL3().run_once(
        log=log,
        projections=projections,
        emitter=emitter,
        config=SimpleNamespace(now="2026-01-03T00:00:00+00:00"),
    )
    after = list(log.iter())[len(before):]

    assert [event.kind for event in after] == [CognitiveEventKind.SELF_MODEL_UPDATED]


def test_consolidation_scheduler_runs_l3_with_l3_priority(tmp_path: Path) -> None:
    store, log, projections, _emitter = _runtime(tmp_path)
    coordinator = _PriorityRecordingCoordinator()

    reports = ConsolidationLoop(
        scheduler=Scheduler(log, CheckpointStore(store)),
        coordinator=coordinator,
        log=log,
        projections=projections,
        config=ConsolidationConfig(),
        workers=[ReflectorL3()],
    ).run_once()

    assert [item.worker for item in reports] == ["reflector_l3"]
    assert coordinator.priorities == [LoopPriority.L3]


def _runtime(tmp_path: Path):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    projections = default_projection_registry(log)
    return store, log, projections, emitter


def _emit_apply(
    emitter: EventEmitter,
    projections,
    kind: CognitiveEventKind,
    payload: dict[str, object],
):
    event = emitter.emit(kind, payload=payload)
    for projection in projections.all():
        if event.kind in projection.handles:
            projection.apply(event)
    return event


class _PriorityRecordingCoordinator:
    def __init__(self) -> None:
        self.priorities: list[LoopPriority] = []

    def acquire(self, req: LoopAcquireRequest):
        class _Context:
            def __enter__(_self):
                self.priorities.append(req.priority)

            def __exit__(_self, exc_type, exc, tb):
                return False

        return _Context()

    def yield_to_higher_priority(self) -> bool:
        return False
