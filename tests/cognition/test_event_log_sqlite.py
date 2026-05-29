from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, id_factory, perceived_payload


def test_sqlite_event_log_append_replay_and_persistence(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log, id_factory=id_factory(), clock=clock_factory())

    for index in range(1000):
        emitter.emit(
            CognitiveEventKind.PERCEIVED,
            payload=perceived_payload(index=index, raw=f"message-{index}"),
        )

    reopened = SQLiteEventLog(StateStore(tmp_path / "alpha.db"))
    events = list(reopened.iter())

    assert reopened.length() == 1000
    assert [event.payload["index"] for event in events] == list(range(1000))
    assert reopened.get(events[123].id).payload["index"] == 123
    assert list(reopened.iter(kinds=[])) == []
