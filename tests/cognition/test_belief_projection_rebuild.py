from __future__ import annotations

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_reset_and_replay_rebuilds_equivalent_belief_view(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()

    for item in [
        belief("belief:old", "User prefers Python."),
        belief("belief:new", "User prefers Rust.", object_="rust"),
    ]:
        projection.apply(
            emit(
                log,
                CognitiveEventKind.BELIEF_FORMED,
                payload={"belief": item.to_record()},
                event_ids=event_ids,
                clock=clock,
            )
        )
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            payload={"old_belief_id": "belief:old", "new_belief_id": "belief:new"},
            event_ids=event_ids,
            clock=clock,
        )
    )
    before = [item.to_record() for item in projection.list_active()]

    registry = ProjectionRegistry()
    rebuilt = BeliefProjection(store)
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()

    assert [item.to_record() for item in rebuilt.list_active()] == before
    ProjectionRunner(log, registry).replay_all()
    assert [item.to_record() for item in rebuilt.list_active()] == before


def test_default_projection_auto_rebuilds_empty_materialized_view_from_event_log(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    formed = emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"belief": belief("belief:auto", "User prefers Python.").to_record()},
    )
    projection = BeliefProjection(store)
    projection.apply(formed)
    assert [item.id for item in projection.list_active()] == ["belief:auto"]
    projection.reset()

    rebuilt = BeliefProjection(store, event_log=log, auto_rebuild=True)

    assert [item.id for item in rebuilt.list_active()] == ["belief:auto"]
