"""Synchronous v1 Drive Loop."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.goals import GoalRegistry
from alpha_agent.cognition.loops.scheduler import ScheduleTrigger
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Goal,
    GoalId,
    Instant,
    LoopPriority,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.goal import ACTIVE_GOAL_LIMIT, GoalProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class DriveConfig:
    enabled: bool = False
    interval_seconds: int = 300
    goal_cooldown_seconds: int = 3600
    active_goal_limit: int = ACTIVE_GOAL_LIMIT


@dataclass(frozen=True)
class DriveReport:
    selected_goal_id: GoalId | None
    triggered: bool
    dropped: bool = False
    skipped_reason: str = ""
    linked_event_ids: list[str] = field(default_factory=list)


class DriveLoop:
    """Turn one eligible active goal into a cognition self-signal."""

    trigger = ScheduleTrigger(
        min_interval=timedelta(minutes=5),
        max_interval=None,
        watches=frozenset(
            {
                CognitiveEventKind.GOAL_SET,
                CognitiveEventKind.GOAL_PROGRESSED,
                CognitiveEventKind.ACTED,
            }
        ),
        min_new_events=1,
    )

    def __init__(
        self,
        *,
        log: EventLog,
        projections: ProjectionRegistry,
        runtime_turn_runner: Any,
        coordinator: LoopCoordinator | None = None,
        emitter: EventEmitter | None = None,
        config: DriveConfig | None = None,
        clock: Callable[[], str] | None = None,
    ):
        self.log = log
        self.projections = projections
        self.runtime_turn_runner = runtime_turn_runner
        self.coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
        self.emitter = emitter or EventEmitter(log)
        self.config = config or DriveConfig()
        self.clock = clock or utc_now_iso

    def run_once(self, *, force: bool = False) -> DriveReport:
        if not self.config.enabled and not force:
            return DriveReport(None, triggered=False, skipped_reason="disabled")

        now = Instant(self.clock())
        goal: Goal | None = None
        drive_req = LoopAcquireRequest(
            loop_name="drive",
            priority=LoopPriority.DRIVE,
            max_chunk_duration=timedelta(seconds=10),
        )
        with self.coordinator.acquire(drive_req):
            goal = self._select_goal(now)
            if goal is None:
                return DriveReport(None, triggered=False, skipped_reason="no_eligible_goal")

        result = self._run_runtime_turn(goal)
        if _is_busy_turn(result):
            return DriveReport(
                goal.id,
                triggered=False,
                dropped=True,
                skipped_reason="runtime_turn_busy",
            )

        linked_event_ids = _linked_event_ids(result)
        registry = GoalRegistry(
            self.log,
            emitter=self.emitter,
            projection=self.projections.get_typed(GoalProjection),
            active_limit=self.config.active_goal_limit,
        )
        progress = registry.progress(
            goal.id,
            note="Drive Loop triggered a runtime self-signal turn.",
            linked_event_ids=linked_event_ids,
            drive_progress=True,
        )
        return DriveReport(
            goal.id,
            triggered=True,
            linked_event_ids=[*linked_event_ids, str(progress.id)],
        )

    def _select_goal(self, now: Instant) -> Goal | None:
        projection = self.projections.get_typed(GoalProjection)
        for goal in projection.active():
            if self._cooldown_elapsed(goal, now):
                return goal
        return None

    def _cooldown_elapsed(self, goal: Goal, now: Instant) -> bool:
        if goal.last_drive_at is None:
            return True
        elapsed = _parse_instant(now) - _parse_instant(goal.last_drive_at)
        return elapsed >= timedelta(seconds=self.config.goal_cooldown_seconds)

    def _run_runtime_turn(self, goal: Goal) -> Any:
        session_id = _goal_session_id(goal.id)
        message = _self_signal_message(goal)
        source_metadata = {
            "source": "drive_loop",
            "stimulus_kind": "self_signal",
            "goal_id": str(goal.id),
        }
        respond = getattr(self.runtime_turn_runner, "respond", None)
        if callable(respond):
            return respond(
                message,
                session_id=session_id,
                source_metadata=source_metadata,
            )
        if callable(self.runtime_turn_runner):
            return self.runtime_turn_runner(
                message,
                session_id,
                source_metadata,
            )
        raise TypeError("runtime_turn_runner must be AlphaAgent-like or callable")


def _is_busy_turn(result: Any) -> bool:
    debug = getattr(result, "debug", None)
    return isinstance(debug, Mapping) and debug.get("busy") is True


def _linked_event_ids(result: Any) -> list[str]:
    debug = getattr(result, "debug", None)
    if not isinstance(debug, Mapping):
        return []
    event_ids: list[str] = []
    for key in ("turn_received_event_id", "acted_event_id", "turn_sources_event_id"):
        value = debug.get(key)
        if value is not None and str(value):
            event_ids.append(str(value))
    event_ids.extend(_string_list(debug.get("tool_cognitive_event_ids")))
    return event_ids


def _goal_session_id(goal_id: GoalId | str) -> str:
    return f"internal:goal:{goal_id}"


def _self_signal_message(goal: Goal) -> str:
    lines = [
        "[self_signal]",
        f"goal_id: {goal.id}",
        "drive_reason: active goal needs progress",
        f"goal_description: {goal.description}",
    ]
    if goal.target_outcome:
        lines.append(f"target_outcome: {goal.target_outcome}")
    return "\n".join(lines)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if item is not None and str(item)]


def _parse_instant(value: Instant) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
