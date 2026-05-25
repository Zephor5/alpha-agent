from __future__ import annotations

from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.cognition.render import DiffRenderer, RenderBudget
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.render_helpers import view
from tests.cognition.test_belief_projection_apply import belief


def test_diff_renderer_lists_belief_lens_and_strategy_changes() -> None:
    log = InMemoryEventLog()
    event_ids = id_factory()
    clock = clock_factory()
    emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"tick_id": "tick:a", "belief": belief("belief:old", "Old.").to_record()},
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"tick_id": "tick:b", "belief": belief("belief:new", "New.").to_record()},
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.VALUE_LENS_SHIFTED,
        payload={"tick_id": "tick:b", "lens_id": "lens:1"},
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.STRATEGY_CHANGED,
        payload={"tick_id": "tick:b", "strategy_id": "strategy:1"},
        event_ids=event_ids,
        clock=clock,
    )

    rendered = DiffRenderer(log, tick_id_a="tick:a", tick_id_b="tick:b").render(
        view(),
        RenderBudget(),
    )

    assert "- belief_formed:belief:old" in rendered.payload
    assert "+ belief_formed:belief:new" in rendered.payload
    assert "+ value_lens_shifted:lens:1" in rendered.payload
    assert "+ strategy_changed:strategy:1" in rendered.payload
