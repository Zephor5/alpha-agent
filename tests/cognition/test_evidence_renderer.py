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
    proposed = emit(
        log,
        CognitiveEventKind.MEMORY_PROPOSED,
        payload={
            "turn_id": "turn-1",
            "session_id": "s1",
            "proposal_id": "proposal-1",
            "tool_call_id": "call-memory",
            "proposal": {
                "operation": "replace",
                "targets": ["belief:old"],
            },
            "derived_about": [],
            "source_refs": [],
            "audit_refs": [],
            "gate": {"decision": "accepted", "reason": "accepted_replace"},
            "operation": "replace",
            "target_belief_ids": ["belief:old"],
            "reason": "User changed the preference.",
            "evidence": "User said to replace the old preference.",
        },
        event_ids=event_ids,
        clock=clock,
    )
    superseded = emit(
        log,
        CognitiveEventKind.BELIEF_SUPERSEDED,
        payload={
            "turn_id": "turn-1",
            "session_id": "s1",
            "proposal_id": "proposal-1",
            "origin": "memory_propose",
            "operation": "replace",
            "target_belief_ids": ["belief:old"],
            "reason": "User changed the preference.",
            "evidence": "User said to replace the old preference.",
            "tool_call_id": "call-memory",
            "old_belief_id": "belief:old",
            "new_belief_id": "belief:new",
            "belief": belief("belief:new", "New.").to_record(),
        },
        event_ids=event_ids,
        clock=clock,
    )
    object.__setattr__(formed, "inputs", [Reference("perception", "perception:1")])
    object.__setattr__(proposed, "outputs", [Reference("memory_proposal", "proposal-1")])
    object.__setattr__(superseded, "inputs", [Reference("perception", "perception:2")])

    rendered = EvidenceRenderer(log, belief_id="belief:old").render(view(), RenderBudget())

    assert "belief_formed" in rendered.payload
    assert "memory_proposed" in rendered.payload
    assert "belief_superseded" in rendered.payload
    assert "operation=replace" in rendered.payload
    assert "target_belief_ids=belief:old" in rendered.payload
    assert "old_belief_id=belief:old" in rendered.payload
    assert "new_belief_id=belief:new" in rendered.payload
    assert "gate=accepted:accepted_replace" in rendered.payload
    assert "evidence=User said to replace the old preference." in rendered.payload
    assert "perception:perception:1" in rendered.payload
    assert "perception:perception:2" in rendered.payload
