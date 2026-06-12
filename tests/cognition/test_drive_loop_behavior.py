from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pytest

from alpha_agent.cognition.coordinator import LockBusy, LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.goals import GoalRegistry
from alpha_agent.cognition.loops import DriveConfig, DriveLoop
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartId,
    GoalId,
    Instant,
    counterpart_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore
from alpha_agent.utils.system_reminder import inline_system_reminder
from tests.cognition.helpers import clock_factory, id_factory


def test_drive_loop_triggers_runtime_self_signal_turn_and_updates_goal_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    store, log, emitter, loop = _drive_runtime(tmp_path, enabled=False)
    registry = GoalRegistry(
        log,
        emitter=emitter,
        projection=GoalProjection(store),
    )
    registry.set_goal(
        description="answer pending user question",
        target_outcome="clear answer sent",
        priority=5,
        goal_id=GoalId("goal:pending"),
    )

    report = loop.run_once(force=True)
    events = list(log.iter())
    perceived = [
        event for event in events if event.kind == CognitiveEventKind.PERCEIVED
    ][-1]
    messages = store.list_session_messages("internal:goal:goal:pending")
    session_record = store.get_session_record("internal:goal:goal:pending")
    goal = loop.projections.get_typed(GoalProjection).get("goal:pending")
    linked = [event for event in events if str(event.id) in report.linked_event_ids]

    assert report.triggered is True
    assert str(report.selected_goal_id) == "goal:pending"
    assert [message.kind for message in messages] == [
        "system_reminder",
        "user_message",
        "assistant_message",
    ]
    assert messages[0].raw_content == inline_system_reminder(
        "started at: 2026-01-01T08:00+08:00"
    )
    assert messages[1].raw_content.startswith("[self_signal]")
    assert "goal_id: goal:pending" in messages[1].raw_content
    assert perceived.payload["stimulus_kind"] == "self_signal"
    assert perceived.payload["session_id"] == "internal:goal:goal:pending"
    assert perceived.payload["from_counterpart"] is None
    assert store.get_session_counterpart("internal:goal:goal:pending") is None
    assert session_record is not None
    assert session_record.timezone == "Asia/Shanghai"
    assert perceived.payload["turn_id"]
    assert [event.kind for event in linked] == [
        CognitiveEventKind.PERCEIVED,
        CognitiveEventKind.ACTED,
        CognitiveEventKind.TURN_SOURCES_RECORDED,
        CognitiveEventKind.GOAL_PROGRESSED,
    ]
    assert goal is not None
    assert goal.last_drive_at is not None
    assert any(event.kind == CognitiveEventKind.GOAL_PROGRESSED for event in events)


def test_drive_loop_cooldown_prevents_repeated_trigger(tmp_path: Path) -> None:
    store, log, emitter, loop = _drive_runtime(
        tmp_path,
        enabled=True,
        now_values=[
            "2026-01-01T00:00:10+00:00",
            "2026-01-01T00:00:20+00:00",
        ],
        goal_cooldown_seconds=3600,
    )
    registry = GoalRegistry(log, emitter=emitter, projection=GoalProjection(store))
    registry.set_goal(description="keep thinking", goal_id=GoalId("goal:think"))

    first = loop.run_once()
    second = loop.run_once()

    assert first.triggered is True
    assert second.triggered is False
    assert second.skipped_reason == "no_eligible_goal"
    assert len(
        [
            event
            for event in log.iter(kinds=[CognitiveEventKind.GOAL_PROGRESSED])
            if event.payload.get("drive_progress") is True
        ]
    ) == 1


def test_drive_loop_carries_goal_counterpart_into_self_signal_turn(
    tmp_path: Path,
) -> None:
    store, log, emitter, loop = _drive_runtime(tmp_path, enabled=False)
    registry = GoalRegistry(log, emitter=emitter, projection=GoalProjection(store))
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))
    registry.set_goal(
        description="answer user A",
        goal_id=GoalId("goal:user-a"),
        for_counterpart=counterpart,
    )

    report = loop.run_once(force=True)
    perceived = [
        event
        for event in log.iter(kinds=[CognitiveEventKind.PERCEIVED])
        if event.payload.get("session_id") == "internal:goal:goal:user-a"
    ][-1]
    messages = store.list_session_messages("internal:goal:goal:user-a")
    binding = store.get_session_counterpart("internal:goal:goal:user-a")

    assert report.triggered is True
    assert "for_counterpart: counterpart:user-a" in messages[1].raw_content
    assert perceived.payload["stimulus_kind"] == "self_signal"
    assert perceived.payload["from_counterpart"] == {
        "kind": "counterpart",
        "id": "counterpart:user-a",
    }
    assert binding is not None
    assert binding.counterpart_id == "counterpart:user-a"


def test_drive_loop_disabled_by_default_but_force_runs(tmp_path: Path) -> None:
    store, log, emitter, loop = _drive_runtime(tmp_path, enabled=False)
    registry = GoalRegistry(log, emitter=emitter, projection=GoalProjection(store))
    registry.set_goal(description="manual only", goal_id=GoalId("goal:manual"))

    skipped = loop.run_once()
    forced = loop.run_once(force=True)

    assert skipped.triggered is False
    assert skipped.skipped_reason == "disabled"
    assert forced.triggered is True


def test_drive_loop_drops_self_signal_when_runtime_turn_is_busy(tmp_path: Path) -> None:
    store, log, emitter, loop = _drive_runtime(
        tmp_path,
        enabled=True,
        coordinator=_RuntimeBusyCoordinator(),
    )
    registry = GoalRegistry(log, emitter=emitter, projection=GoalProjection(store))
    registry.set_goal(description="retry later", goal_id=GoalId("goal:retry"))

    report = loop.run_once()
    events = list(log.iter())
    goal = loop.projections.get_typed(GoalProjection).get("goal:retry")

    assert report.triggered is False
    assert report.dropped is True
    assert report.skipped_reason == "runtime_turn_busy"
    assert store.list_session_messages("internal:goal:goal:retry") == []
    assert [
        event for event in events if event.kind == CognitiveEventKind.PERCEIVED
    ] == []
    assert [
        event
        for event in events
        if event.kind == CognitiveEventKind.GOAL_PROGRESSED
        and event.payload.get("drive_progress") is True
    ] == []
    assert goal is not None
    assert goal.status == "active"
    assert goal.last_drive_at is None


def _drive_runtime(
    tmp_path: Path,
    *,
    enabled: bool,
    now_values: list[str] | None = None,
    goal_cooldown_seconds: int = 3600,
    coordinator: LoopCoordinator | None = None,
):
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory("evt"), clock=clock_factory())
    projections = ProjectionRegistry()
    projections.register(GoalProjection(store, event_log=log, auto_rebuild=True))
    coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        event_log=log,
        coordinator=coordinator,
        local_clock=lambda: datetime.fromisoformat("2026-01-01T08:00:10+08:00"),
    )
    clock = _sequence_clock(now_values or ["2026-01-01T00:00:10+00:00"])
    loop = DriveLoop(
        log=log,
        projections=projections,
        runtime_turn_runner=agent,
        coordinator=coordinator,
        emitter=emitter,
        config=DriveConfig(
            enabled=enabled,
            goal_cooldown_seconds=goal_cooldown_seconds,
        ),
        clock=clock,
    )
    return store, log, emitter, loop


class _RuntimeBusyCoordinator(LoopCoordinator):
    def __init__(self) -> None:
        super().__init__(SUBJECT_SELF)

    @contextmanager
    def try_acquire(self, req: LoopAcquireRequest) -> Iterator[None]:
        if req.loop_name == "runtime_turn":
            raise LockBusy("runtime_turn", Instant("2026-01-01T00:00:00+00:00"))
        with super().try_acquire(req):
            yield


def _sequence_clock(values: list[str]):
    index = 0

    def now() -> str:
        nonlocal index
        value = values[min(index, len(values) - 1)]
        index += 1
        return value

    return now
