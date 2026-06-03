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
    old = emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"belief": belief("belief:old", "Old.").to_record()},
        event_ids=event_ids,
        clock=clock,
    )
    new = emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"belief": belief("belief:new", "New.").to_record()},
        event_ids=event_ids,
        clock=clock,
    )
    lens = emit(
        log,
        CognitiveEventKind.VALUE_LENS_SHIFTED,
        payload={"lens_id": "lens:1"},
        event_ids=event_ids,
        clock=clock,
    )
    strategy = emit(
        log,
        CognitiveEventKind.STRATEGY_CHANGED,
        payload={"strategy_id": "strategy:1"},
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.TURN_SOURCES_RECORDED,
        payload=_turn_sources_payload("turn_a", [str(old.id)]),
        event_ids=event_ids,
        clock=clock,
    )
    emit(
        log,
        CognitiveEventKind.TURN_SOURCES_RECORDED,
        payload=_turn_sources_payload(
            "turn_b",
            [str(new.id), str(lens.id), str(strategy.id)],
        ),
        event_ids=event_ids,
        clock=clock,
    )

    rendered = DiffRenderer(log, turn_id_a="turn_a", turn_id_b="turn_b").render(
        view(),
        RenderBudget(),
    )

    assert "- belief_formed:belief:old" in rendered.payload
    assert "+ belief_formed:belief:new" in rendered.payload
    assert "+ value_lens_shifted:lens:1" in rendered.payload
    assert "+ strategy_changed:strategy:1" in rendered.payload


def _turn_sources_payload(turn_id: str, event_ids: list[str]) -> dict[str, object]:
    return {
        "turn_id": turn_id,
        "session_id": "s1",
        "user_message_id": f"user:{turn_id}",
        "assistant_message_id": f"assistant:{turn_id}",
        "provider_tool_message_ids": [],
        "provider_tool_trace_ids": [],
        "llm_call_ids": [],
        "llm_trace_ids": [],
        "cognitive_event_ids": event_ids,
        "tool_cognitive_event_ids": [],
    }
