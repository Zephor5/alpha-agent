from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from tests.cognition.helpers import clock_factory, emit, id_factory


def test_memory_event_log_append_iter_and_length() -> None:
    log = InMemoryEventLog()
    ids = id_factory()
    clock = clock_factory()

    first = emit(log, CognitiveEventKind.PERCEIVED, event_ids=ids, clock=clock)
    second = emit(
        log,
        CognitiveEventKind.RECEIVED_FEEDBACK,
        payload={
            "turn_id": "turn-1",
            "session_id": "s1",
            "feedback_kind": "external",
            "matched_expected": True,
        },
        event_ids=ids,
        clock=clock,
    )

    assert log.length() == 2
    assert log.get(first.id) == first
    assert list(log.iter()) == [first, second]
    assert list(log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK])) == [second]
    assert list(log.iter(kinds=[])) == []
