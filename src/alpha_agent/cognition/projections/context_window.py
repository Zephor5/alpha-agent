"""Stub context window projection for Phase 02."""

from __future__ import annotations

from dataclasses import dataclass, replace

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import (
    Belief,
    BeliefId,
    BeliefRef,
    CognitiveEvent,
    CognitiveEventKind,
    ContextWindow,
    Instant,
    Perception,
    SituationId,
    Subject,
    belief_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.thread import ThreadId
from alpha_agent.cognition.projections.base import Projection
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class ContextWindowProjectionView:
    status: str = "stub"


class ContextWindowProjection(Projection):
    """Assemble recent perceived events as foreground context."""

    name = "context_window"
    handles = frozenset({CognitiveEventKind.PERCEIVED})
    status = "stub"

    def __init__(self, event_log: EventLog, *, recent_limit: int = 8):
        self.event_log = event_log
        self.recent_limit = max(1, recent_limit)

    def get(
        self,
        thread_id: ThreadId,
        subject: Subject,
        at: Instant | None = None,
    ) -> ContextWindow:
        perceived = [
            event
            for event in self.event_log.iter(kinds=[CognitiveEventKind.PERCEIVED], until=at)
            if self._same_thread(event, thread_id)
        ][-self.recent_limit :]
        foreground = [self._perception_from_event(event) for event in perceived]
        situation = (
            foreground[-1].situation
            if foreground
            else situation_ref(SituationId("situation:stub"))
        )
        counterpart = foreground[-1].from_counterpart if foreground else None
        return ContextWindow(
            thread_id=thread_id,
            counterpart=counterpart,
            foreground=foreground,
            background=None,
            recalled=[],
            recent_judgments=[],
            matched_procedures=[],
            subject_at=subject_ref(subject.id),
            situation_at=situation,
            assembled_at=at or Instant(utc_now_iso()),
            metadata={"status": self.status},
        )

    def attach_recalled(
        self,
        window: ContextWindow,
        recalled: list[Belief | BeliefRef],
    ) -> ContextWindow:
        return replace(window, recalled=[_belief_reference(item) for item in recalled])

    def apply(self, event: CognitiveEvent) -> None:
        return None

    def reset(self) -> None:
        return None

    def view(self) -> ContextWindowProjectionView:
        return ContextWindowProjectionView()

    def _same_thread(self, event: CognitiveEvent, thread_id: ThreadId) -> bool:
        raw = event.payload.get("thread_id")
        if not isinstance(raw, dict):
            return True
        return raw.get("kind") == thread_id.kind.value and raw.get("key") == thread_id.key

    def _perception_from_event(self, event: CognitiveEvent) -> Perception:
        raw = event.payload.get("perception")
        if not isinstance(raw, dict):
            raise ValueError(f"perceived event missing perception payload: {event.id}")
        return Perception.from_record(raw)


def _belief_reference(value: Belief | BeliefRef) -> BeliefRef:
    if isinstance(value, Belief):
        return belief_ref(BeliefId(str(value.id)))
    return value
