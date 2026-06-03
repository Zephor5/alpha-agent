"""Belief evidence-chain renderer."""

from __future__ import annotations

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView

_EVIDENCE_KINDS = {
    CognitiveEventKind.MEMORY_PROPOSED,
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
    events = list(event_log.iter(kinds=_EVIDENCE_KINDS))
    direct_events = [
        event for event in events if belief_id in _event_belief_ids(event)
    ]
    proposal_ids = {
        proposal_id
        for event in direct_events
        if (proposal_id := _event_proposal_id(event)) is not None
    }
    causal_parent_ids = {
        str(parent)
        for event in direct_events
        for parent in event.causal_parents
    }
    return [
        event
        for event in events
        if belief_id in _event_belief_ids(event)
        or (
            (proposal_id := _event_proposal_id(event)) is not None
            and proposal_id in proposal_ids
        )
        or str(event.id) in causal_parent_ids
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
    for field in ("target_belief_ids", "conflict_belief_ids"):
        raw_ids = event.payload.get(field)
        if isinstance(raw_ids, list):
            ids.update(str(value) for value in raw_ids if value is not None)
    raw_belief = event.payload.get("belief")
    if isinstance(raw_belief, dict) and raw_belief.get("id") is not None:
        ids.add(str(raw_belief["id"]))
    return ids


def _event_proposal_id(event: CognitiveEvent) -> str | None:
    raw = event.payload.get("proposal_id")
    return str(raw) if raw is not None else None


def _format_event(event: CognitiveEvent) -> str:
    inputs = ", ".join(f"{item.kind}:{item.id}" for item in event.inputs) or "-"
    outputs = ", ".join(f"{item.kind}:{item.id}" for item in event.outputs) or "-"
    details = _format_payload_details(event.payload)
    detail_text = f" {' '.join(details)}" if details else ""
    return (
        f"- {event.timestamp} {event.kind.value} event={event.id} "
        f"inputs={inputs} outputs={outputs}{detail_text}"
    )


def _format_payload_details(payload: dict[str, object]) -> list[str]:
    details: list[str] = []
    for field in (
        "proposal_id",
        "operation",
        "belief_id",
        "old_belief_id",
        "new_belief_id",
        "tool_call_id",
    ):
        value = payload.get(field)
        if value is not None:
            details.append(f"{field}={value}")
    target_ids = payload.get("target_belief_ids")
    if isinstance(target_ids, list):
        details.append(
            "target_belief_ids="
            + ",".join(str(value) for value in target_ids if value is not None)
        )
    gate = payload.get("gate")
    if isinstance(gate, dict):
        decision = gate.get("decision", "")
        reason = gate.get("reason", "")
        details.append(f"gate={decision}:{reason}")
    for field in ("reason", "evidence"):
        value = payload.get(field)
        if isinstance(value, str) and value:
            details.append(f"{field}={value}")
    return details
