from __future__ import annotations

from pathlib import Path

import pytest

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.goals import GoalRegistry
from alpha_agent.cognition.models import GoalId
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory


def test_goal_projection_rebuilds_from_event_log(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    registry = GoalRegistry(log, emitter=emitter, projection=GoalProjection(store))

    registry.set_goal(
        description="draft response",
        target_outcome="ready to send",
        priority=8,
        goal_id=GoalId("goal:draft"),
    )
    registry.progress("goal:draft", note="self_signal completed", drive_progress=True)
    registry.abandon("goal:draft", reason="question withdrawn")

    projection = GoalProjection(store)
    projection.reset()
    rebuilt = GoalProjection(store, event_log=log, auto_rebuild=True)
    goal = rebuilt.get("goal:draft")

    assert goal is not None
    assert goal.status == "abandoned"
    assert goal.last_drive_at == "2026-01-01T00:00:02+00:00"
    assert rebuilt.active() == []


def test_goal_projection_enforces_active_goal_cap_on_replay(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    registry = GoalRegistry(log, emitter=emitter)

    registry.set_goal(description="first", goal_id=GoalId("goal:first"))
    registry.set_goal(description="second", goal_id=GoalId("goal:second"))
    registry.set_goal(description="third", goal_id=GoalId("goal:third"))

    replayed = GoalProjection(store, event_log=log, auto_rebuild=True, active_limit=2)

    assert [str(goal.id) for goal in replayed.active()] == ["goal:first", "goal:second"]
    assert [event.payload["goal"]["id"] for event in log.iter()] == [
        "goal:first",
        "goal:second",
        "goal:third",
    ]


def test_goal_registry_enforces_active_goal_cap_before_emitting(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    projection = GoalProjection(store, active_limit=2)
    registry = GoalRegistry(log, emitter=emitter, projection=projection, active_limit=2)

    registry.set_goal(description="first", goal_id=GoalId("goal:first"))
    registry.set_goal(description="second", goal_id=GoalId("goal:second"))
    with pytest.raises(ValueError, match="active goal limit exceeded"):
        registry.set_goal(description="third", goal_id=GoalId("goal:third"))

    assert [str(goal.id) for goal in projection.active()] == ["goal:first", "goal:second"]
    assert len(list(log.iter())) == 2
