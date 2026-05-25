"""Tick-to-tick cognition event diff renderer."""

from __future__ import annotations

from collections import defaultdict

from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind
from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView

_DIFF_KINDS = {
    "belief": {
        CognitiveEventKind.BELIEF_FORMED,
        CognitiveEventKind.BELIEF_STRENGTHENED,
        CognitiveEventKind.BELIEF_WEAKENED,
        CognitiveEventKind.BELIEF_SUPERSEDED,
        CognitiveEventKind.BELIEF_RETRACTED,
        CognitiveEventKind.BELIEF_ARCHIVED,
    },
    "lens": {CognitiveEventKind.VALUE_LENS_SHIFTED},
    "strategy": {
        CognitiveEventKind.STRATEGY_CHANGED,
        CognitiveEventKind.STRATEGY_EXPIRED,
    },
}


class DiffRenderer:
    """Render deterministic event-kind deltas between two ticks."""

    name = "diff"

    def __init__(self, event_log: EventLog, *, tick_id_a: str, tick_id_b: str):
        self.event_log = event_log
        self.tick_id_a = tick_id_a
        self.tick_id_b = tick_id_b

    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult:
        before = _collect(self.event_log, self.tick_id_a)
        after = _collect(self.event_log, self.tick_id_b)
        lines = [f"Diff {self.tick_id_a} -> {self.tick_id_b}"]
        for section in ("belief", "lens", "strategy"):
            removed, added = _section_delta(before[section], after[section])
            lines.append(f"{section}:")
            lines.extend(f"  - {item}" for item in removed[: budget.max_tokens])
            lines.extend(f"  + {item}" for item in added[: budget.max_tokens])
            if not removed and not added:
                lines.append("  (no change)")
        payload = "\n".join(lines)
        return RenderResult(payload=payload, used_tokens=len(payload) // 4)


def _collect(event_log: EventLog, tick_id: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for event in event_log.iter():
        if str(event.payload.get("tick_id") or "") != tick_id:
            continue
        section = _section(event)
        if section is not None:
            grouped[section].append(_event_label(event))
    return grouped


def _section(event: CognitiveEvent) -> str | None:
    for section, kinds in _DIFF_KINDS.items():
        if event.kind in kinds:
            return section
    return None


def _section_delta(before: list[str], after: list[str]) -> tuple[list[str], list[str]]:
    before_set = set(before)
    after_set = set(after)
    return sorted(before_set - after_set), sorted(after_set - before_set)


def _event_label(event: CognitiveEvent) -> str:
    belief_id = event.payload.get("belief_id") or event.payload.get("id")
    if belief_id is None and isinstance(event.payload.get("belief"), dict):
        belief_id = event.payload["belief"].get("id")
    target = (
        belief_id
        or event.payload.get("strategy_id")
        or event.payload.get("lens_id")
        or event.id
    )
    return f"{event.kind.value}:{target}"
