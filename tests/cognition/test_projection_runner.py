from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.event_count import EventCountByKind
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from tests.cognition.helpers import clock_factory, id_factory


def test_projection_runner_idempotent_rebuild_and_late_projection() -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())
    emitter.emit(CognitiveEventKind.PERCEIVED)
    emitter.emit(CognitiveEventKind.JUDGED)
    emitter.emit(CognitiveEventKind.JUDGED)

    registry = ProjectionRegistry()
    projection = EventCountByKind()
    registry.register(projection)
    runner = ProjectionRunner(log, registry)

    runner.replay_all()
    first_view = projection.view()
    runner.replay_all()

    assert projection.view() == first_view
    assert projection.view().counts[CognitiveEventKind.JUDGED] == 2

    class LateEventCountByKind(EventCountByKind):
        name = "late_event_count_by_kind"

    late_projection = LateEventCountByKind()
    registry.register(late_projection)
    runner.replay_all()

    assert late_projection.view() == projection.view()
