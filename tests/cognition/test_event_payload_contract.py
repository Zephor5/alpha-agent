from __future__ import annotations

import pytest

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.cognition.payload_contract import EventPayloadValidationError
from tests.cognition.helpers import clock_factory, id_factory


@pytest.mark.parametrize(
    ("kind", "payload", "missing"),
    [
        (
            CognitiveEventKind.PERCEIVED,
            {
                "turn_id": "turn-1",
                "session_id": "s1",
                "stimulus_kind": "user_message",
                "source": {"kind": "session", "id": "s1"},
                "source_refs": [],
                "content_digest": "abc",
            },
            "content_length",
        ),
        (
            CognitiveEventKind.ACTED,
            {
                "turn_id": "turn-1",
                "session_id": "s1",
                "assistant_message_id": "msg-1",
                "response_text_digest": "abc",
                "response_text_length": 2,
                "llm_call_ids": [],
                "llm_trace_ids": [],
                "tool_call_ids": [],
                "tool_names": [],
                "tool_result_trace_ids": [],
            },
            "tool_cognitive_event_ids",
        ),
        (
            CognitiveEventKind.MEMORY_PROPOSED,
            {
                "turn_id": "turn-1",
                "session_id": "s1",
                "proposal_id": "proposal-1",
                "tool_call_id": "call-1",
                "proposal": {},
                "derived_about": [],
                "source_refs": [],
                "audit_refs": [],
            },
            "gate",
        ),
        (
            CognitiveEventKind.TURN_SOURCES_RECORDED,
            {
                "turn_id": "turn-1",
                "session_id": "s1",
                "user_message_id": "msg-user",
                "assistant_message_id": "msg-assistant",
                "provider_tool_message_ids": [],
                "provider_tool_trace_ids": [],
                "llm_call_ids": [],
                "llm_trace_ids": [],
                "cognitive_event_ids": [],
            },
            "tool_cognitive_event_ids",
        ),
        (
            CognitiveEventKind.RECEIVED_FEEDBACK,
            {
                "turn_id": "turn-1",
                "session_id": "s1",
                "feedback_kind": "external",
            },
            "matched_expected",
        ),
    ],
)
def test_key_event_payload_contract_rejects_missing_consumed_fields(
    kind: CognitiveEventKind,
    payload: dict[str, object],
    missing: str,
) -> None:
    log = InMemoryEventLog()
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())

    with pytest.raises(EventPayloadValidationError, match=missing):
        emitter.emit(kind, payload=payload)

    assert log.length() == 0
