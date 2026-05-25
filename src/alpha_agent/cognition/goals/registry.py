"""Event-sourced goal lifecycle registry."""

from __future__ import annotations

from collections.abc import Iterable

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    BeliefId,
    CognitiveEvent,
    CognitiveEventKind,
    CounterpartRef,
    Goal,
    GoalId,
    Instant,
)
from alpha_agent.cognition.models.goal import GoalSource
from alpha_agent.cognition.projections.goal import ACTIVE_GOAL_LIMIT, GoalProjection
from alpha_agent.utils.ids import new_id


class GoalRegistry:
    """Emit goal lifecycle events and keep an optional projection current."""

    def __init__(
        self,
        log: EventLog,
        *,
        emitter: EventEmitter | None = None,
        projection: GoalProjection | None = None,
        active_limit: int = ACTIVE_GOAL_LIMIT,
    ):
        self.log = log
        self.emitter = emitter or EventEmitter(log)
        self.projection = projection
        self.active_limit = active_limit

    def set_goal(
        self,
        *,
        description: str,
        target_outcome: str = "",
        priority: int = 0,
        source: GoalSource = "user",
        linked_belief_ids: Iterable[BeliefId | str] | None = None,
        for_counterpart: CounterpartRef | None = None,
        goal_id: GoalId | str | None = None,
    ) -> CognitiveEvent:
        """Set or replace an active goal."""

        if self.projection is not None:
            existing = self.projection.get(str(goal_id)) if goal_id is not None else None
            if (
                (existing is None or existing.status != "active")
                and len(self.projection.active()) >= self.active_limit
            ):
                raise ValueError(f"active goal limit exceeded: {self.active_limit}")
        now = Instant(self.emitter.clock())
        goal = Goal(
            id=GoalId(str(goal_id or new_id("goal"))),
            description=description,
            target_outcome=target_outcome,
            priority=priority,
            status="active",
            source=source,
            linked_belief_ids=[
                BeliefId(str(item)) for item in (linked_belief_ids or [])
            ],
            for_counterpart=for_counterpart,
            created_at=now,
            updated_at=now,
        )
        return self._emit_apply(
            CognitiveEventKind.GOAL_SET,
            payload={"goal": goal.to_record(), "source": source},
            timestamp=now,
        )

    def satisfy(self, goal_id: GoalId | str, *, evidence: str) -> CognitiveEvent:
        """Mark a goal as satisfied."""

        return self._emit_apply(
            CognitiveEventKind.GOAL_SATISFIED,
            payload={"goal_id": str(goal_id), "evidence": evidence},
        )

    def abandon(self, goal_id: GoalId | str, *, reason: str) -> CognitiveEvent:
        """Mark a goal as abandoned."""

        return self._emit_apply(
            CognitiveEventKind.GOAL_ABANDONED,
            payload={"goal_id": str(goal_id), "reason": reason},
        )

    def progress(
        self,
        goal_id: GoalId | str,
        *,
        note: str,
        linked_event_ids: Iterable[str] | None = None,
        drive_progress: bool = False,
    ) -> CognitiveEvent:
        """Record progress on a goal."""

        return self._emit_apply(
            CognitiveEventKind.GOAL_PROGRESSED,
            payload={
                "goal_id": str(goal_id),
                "note": note,
                "linked_event_ids": list(linked_event_ids or []),
                "drive_progress": drive_progress,
            },
        )

    def _emit_apply(
        self,
        kind: CognitiveEventKind,
        *,
        payload: dict[str, object],
        timestamp: Instant | None = None,
    ) -> CognitiveEvent:
        event = self.emitter.emit(kind, payload=payload, timestamp=timestamp)
        if self.projection is not None:
            self.projection.apply(event)
        return event
