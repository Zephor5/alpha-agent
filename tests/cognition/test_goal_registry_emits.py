from __future__ import annotations

from pathlib import Path

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.goals import GoalRegistry
from alpha_agent.cognition.models import CognitiveEventKind, GoalId
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory


def test_goal_registry_emits_events_and_materializes_projection(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    projection = GoalProjection(store)
    registry = GoalRegistry(log, emitter=emitter, projection=projection)

    set_event = registry.set_goal(
        description="answer pending user question",
        target_outcome="clear answer sent",
        priority=4,
        goal_id=GoalId("goal:pending-question"),
    )
    progressed = registry.progress("goal:pending-question", note="noted", drive_progress=False)
    satisfied = registry.satisfy("goal:pending-question", evidence="answer accepted")

    goal = projection.get("goal:pending-question")
    events = list(log.iter())

    assert [event.kind for event in events] == [
        CognitiveEventKind.GOAL_SET,
        CognitiveEventKind.GOAL_PROGRESSED,
        CognitiveEventKind.GOAL_SATISFIED,
    ]
    assert set_event.payload["source"] == "user"
    assert progressed.payload["drive_progress"] is False
    assert satisfied.payload["evidence"] == "answer accepted"
    assert goal is not None
    assert goal.status == "satisfied"
    assert goal.description == "answer pending user question"
    assert goal.target_outcome == "clear answer sent"
    assert goal.priority == 4
    assert goal.last_drive_at is None
