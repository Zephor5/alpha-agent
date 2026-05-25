from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind, CounterpartRole
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, counterpart_payload, id_factory


def test_counterpart_projection_lifecycle_and_rebuild(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())
    counterpart_id = "counterpart:user-a"

    emitter.emit(
        CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
        payload=counterpart_payload(counterpart_id, role=CounterpartRole.USER),
    )
    emitter.emit(
        CognitiveEventKind.COUNTERPART_IDENTIFIED,
        payload={"counterpart_id": counterpart_id, "identity": {"platform_id": "u-1"}},
    )
    emitter.emit(
        CognitiveEventKind.COUNTERPART_RELATIONSHIP_CHANGED,
        payload={"counterpart_id": counterpart_id, "relationship": "served_by_agent"},
    )
    emitter.emit(
        CognitiveEventKind.TRUST_UPDATED,
        payload={"counterpart_id": counterpart_id, "trust_level": 0.75},
    )

    registry = ProjectionRegistry()
    projection = CounterpartProjection(store)
    registry.register(projection)
    ProjectionRunner(log, registry).replay_all()
    before = projection.get(counterpart_id)

    assert before is not None
    assert before.first_seen_at == "2026-01-01T00:00:01+00:00"
    assert before.identity == {"display_name": "User A", "platform_id": "u-1"}
    assert before.relationship.kind == "served_by_agent"
    assert before.trust_level == 0.75
    assert projection.by_role(CounterpartRole.USER) == [before]

    projection.reset()
    assert projection.list_active() == []
    ProjectionRunner(log, registry).replay_all()

    assert projection.get(counterpart_id) == before
