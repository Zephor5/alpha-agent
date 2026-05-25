"""Synchronous v1 Drive Loop."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from alpha_agent.cognition.controller import CognitiveController, LoopResult
from alpha_agent.cognition.coordinator import LockBusy, LoopAcquireRequest, LoopCoordinator
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
    Stimulus,
    StimulusKind,
    ThreadId,
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
                CognitiveEventKind.RECEIVED_FEEDBACK,
            }
        ),
        min_new_events=1,
    )

    def __init__(
        self,
        *,
        log: EventLog,
        projections: ProjectionRegistry,
        controller: CognitiveController,
        coordinator: LoopCoordinator | None = None,
        emitter: EventEmitter | None = None,
        config: DriveConfig | None = None,
        clock: Callable[[], str] | None = None,
    ):
        self.log = log
        self.projections = projections
        self.controller = controller
        self.coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
        self.emitter = emitter or EventEmitter(log)
        self.config = config or DriveConfig()
        self.clock = clock or utc_now_iso

    def run_once(self, *, force: bool = False) -> DriveReport:
        if not self.config.enabled and not force:
            return DriveReport(None, triggered=False, skipped_reason="disabled")

        now = Instant(self.clock())
        goal: Goal | None = None
        stimulus: Stimulus | None = None
        thread_id: ThreadId | None = None
        drive_req = LoopAcquireRequest(
            loop_name="drive",
            priority=LoopPriority.DRIVE,
            max_chunk_duration=timedelta(seconds=10),
        )
        with self.coordinator.acquire(drive_req):
            goal = self._select_goal(now)
            if goal is None:
                return DriveReport(None, triggered=False, skipped_reason="no_eligible_goal")
            thread_id = ThreadId.cognition(SUBJECT_SELF, topic=str(goal.id))
            stimulus = Stimulus(
                kind=StimulusKind.SELF_SIGNAL,
                source=goal.for_counterpart,
                payload={
                    "goal_id": str(goal.id),
                    "drive_reason": "active goal needs progress",
                    "goal_description": goal.description,
                    "target_outcome": goal.target_outcome,
                },
                thread_id=thread_id,
                received_at=now,
            )

        reactive_req = LoopAcquireRequest(
            loop_name="reactive",
            priority=LoopPriority.REACTIVE,
            max_chunk_duration=timedelta(seconds=30),
        )
        try:
            with self.coordinator.try_acquire(reactive_req):
                result = self.controller.reactive_tick(stimulus, thread_id)
        except LockBusy:
            return DriveReport(
                goal.id,
                triggered=False,
                dropped=True,
                skipped_reason="reactive_busy",
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
            note="Drive Loop triggered a self_signal reactive tick.",
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


def _linked_event_ids(result: LoopResult) -> list[str]:
    raw = result.debug.get("event_ids")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw]


def _parse_instant(value: Instant) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
