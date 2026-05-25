"""Belief evidence-chain renderer."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView

_EVIDENCE_KINDS = {
    CognitiveEventKind.BELIEF_FORMED,
    CognitiveEventKind.BELIEF_STRENGTHENED,
    CognitiveEventKind.BELIEF_WEAKENED,
    CognitiveEventKind.BELIEF_SUPERSEDED,
    CognitiveEventKind.BELIEF_RETRACTED,
    CognitiveEventKind.BELIEF_ARCHIVED,
}


class EvidenceRenderer:
    """Render the event evidence chain for one belief id."""

    name = "evidence"

    def __init__(self, event_log: EventLog, *, belief_id: str):
        self.event_log = event_log
        self.belief_id = belief_id

    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult:
        events = _belief_events(self.event_log, self.belief_id)
        lines = [f"Evidence for {self.belief_id}"]
        if view.counterpart is not None:
            lines.append(f"counterpart: {view.counterpart.id}")
        for event in events[: max(1, budget.max_tokens)]:
            lines.append(_format_event(event))
        if not events:
            lines.append("(none)")
        payload = "\n".join(lines)
        return RenderResult(payload=payload, used_tokens=len(payload) // 4)


def _belief_events(event_log: EventLog, belief_id: str) -> list[CognitiveEvent]:
    return [
        event
        for event in event_log.iter(kinds=_EVIDENCE_KINDS)
        if belief_id in _event_belief_ids(event)
    ]


def _event_belief_ids(event: CognitiveEvent) -> set[str]:
    ids = {
        str(value)
        for value in (
            event.payload.get("belief_id"),
            event.payload.get("id"),
            event.payload.get("old_belief_id"),
            event.payload.get("new_belief_id"),
            event.payload.get("superseded_belief_id"),
        )
        if value is not None
    }
    raw_belief = event.payload.get("belief")
    if isinstance(raw_belief, dict) and raw_belief.get("id") is not None:
        ids.add(str(raw_belief["id"]))
    return ids


def _format_event(event: CognitiveEvent) -> str:
    inputs = ", ".join(f"{item.kind}:{item.id}" for item in event.inputs) or "-"
    outputs = ", ".join(f"{item.kind}:{item.id}" for item in event.outputs) or "-"
    return (
        f"- {event.timestamp} {event.kind.value} event={event.id} "
        f"inputs={inputs} outputs={outputs}"
    )
