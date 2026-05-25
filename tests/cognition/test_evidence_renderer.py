from __future__ import annotations

from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind, Reference
from alpha_agent.cognition.render import EvidenceRenderer, RenderBudget
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.render_helpers import view
from tests.cognition.test_belief_projection_apply import belief


def test_evidence_renderer_traces_formed_and_superseded_chain() -> None:
    log = InMemoryEventLog()
    event_ids = id_factory()
    clock = clock_factory()
    formed = emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"belief": belief("belief:old", "Old.").to_record()},
        event_ids=event_ids,
        clock=clock,
    )
    superseded = emit(
        log,
        CognitiveEventKind.BELIEF_SUPERSEDED,
        payload={"old_belief_id": "belief:old", "new_belief_id": "belief:new"},
        event_ids=event_ids,
        clock=clock,
    )
    object.__setattr__(formed, "inputs", [Reference("perception", "perception:1")])
    object.__setattr__(superseded, "inputs", [Reference("perception", "perception:2")])

    rendered = EvidenceRenderer(log, belief_id="belief:old").render(view(), RenderBudget())

    assert "belief_formed" in rendered.payload
    assert "belief_superseded" in rendered.payload
    assert "perception:perception:1" in rendered.payload
    assert "perception:perception:2" in rendered.payload
