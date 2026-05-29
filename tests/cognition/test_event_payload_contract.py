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
            {"tick_id": "tick-1", "thread_id": {"kind": "conversation", "key": "s1"}},
            "perception",
        ),
        (CognitiveEventKind.JUDGED, {"tick_id": "tick-1"}, "claim"),
        (
            CognitiveEventKind.DECIDED,
            {"tick_id": "tick-1", "action": "respond"},
            "message",
        ),
        (CognitiveEventKind.ACTED, {"tick_id": "tick-1"}, "decision_id"),
        (
            CognitiveEventKind.RECEIVED_FEEDBACK,
            {"tick_id": "tick-1"},
            "matched_expected",
        ),
        (
            CognitiveEventKind.RECEIVED_FEEDBACK,
            {"matched_expected": True},
            "tick_id",
        ),
        (CognitiveEventKind.REVISED, {"tick_id": "tick-1"}, "judgment_ids"),
        (
            CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION,
            {"tick_id": "tick-1", "reason": "strategy"},
            "contradict_ids",
        ),
        (
            CognitiveEventKind.CONTEXT_COMPRESSED,
            {
                "thread_id": {"kind": "conversation", "key": "s1"},
                "absorbed_perception_ids": ["perception-1"],
                "summary": "old context",
                "compression_policy": "deterministic_v1",
            },
            "produced_summary_id",
        ),
        (CognitiveEventKind.PROCEDURE_LEARNED, {"name": "Repeat"}, "procedure"),
        (CognitiveEventKind.BELIEF_FORMED, {"source": "worker"}, "belief"),
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
