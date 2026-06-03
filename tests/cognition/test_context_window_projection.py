from __future__ import annotations

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    CounterpartId,
    Reference,
    StimulusKind,
    Subject,
    counterpart_ref,
)
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore


def test_context_window_foreground_roll_keeps_last_k_perceptions(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=3)
    subject = Subject()

    for index in range(8):
        event = _perceived_event(log, "s1", f"message-{index}", subject=subject)
        projection.apply(event)

    window = projection.get("s1", subject)

    assert [str(perception.id) for perception in window.foreground] == [
        "perception:turn-message-5",
        "perception:turn-message-6",
        "perception:turn-message-7",
    ]


def test_context_window_foreground_is_keyed_by_session_id(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=5)
    subject = Subject()

    projection.apply(_perceived_event(log, "a", "a-local", subject=subject))
    projection.apply(_perceived_event(log, "a", "a-gateway", subject=subject))
    projection.apply(_perceived_event(log, "b", "b-only", subject=subject))

    assert [str(item.id) for item in projection.get("a", subject).foreground] == [
        "perception:turn-a-local",
        "perception:turn-a-gateway",
    ]
    assert [str(item.id) for item in projection.get("b", subject).foreground] == [
        "perception:turn-b-only"
    ]


def test_context_window_recovers_raw_from_source_session_message(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=5)
    subject = Subject()
    message = log.store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="original user text",
    )
    event = EventEmitter(log).emit(
        CognitiveEventKind.PERCEIVED,
        outputs=[Reference("perception", "perception:turn-raw")],
        payload={
            "turn_id": "turn-raw",
            "session_id": "s1",
            "stimulus_kind": StimulusKind.USER_MESSAGE.value,
            "source": {"kind": "session", "id": "s1"},
            "from_counterpart": None,
            "source_refs": [
                {"kind": "session", "id": "s1"},
                {"kind": "session_message", "id": message.id},
            ],
            "content_digest": "digest-raw",
            "content_length": len(message.raw_content),
        },
    )

    projection.apply(event)

    [perception] = projection.get("s1", subject).foreground
    assert perception.raw == "original user text"


def test_context_window_counterpart_link_lists_sessions(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log)
    subject = Subject()
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))

    projection.apply(
        _perceived_event(log, "s1", "one", subject=subject, counterpart=counterpart)
    )
    projection.apply(
        _perceived_event(log, "s2", "two", subject=subject, counterpart=counterpart)
    )

    assert projection.get("s1", subject).counterpart == counterpart
    assert projection.list_sessions_by_counterpart(counterpart) == ["s1", "s2"]


def test_context_window_anchor_survives_foreground_roll(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=4)
    subject = Subject()

    anchor_event = _perceived_event(log, "s1", "anchor", subject=subject)
    projection.apply(anchor_event)
    anchor_id = "perception:turn-anchor"
    projection.mark_anchor("s1", anchor_id)

    for index in range(10):
        projection.apply(_perceived_event(log, "s1", f"later-{index}", subject=subject))

    window = projection.get("s1", subject)

    assert [str(perception.id) for perception in window.foreground] == [
        "perception:turn-anchor",
        "perception:turn-later-7",
        "perception:turn-later-8",
        "perception:turn-later-9",
    ]


def test_context_window_rebuild_from_event_log(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=3)
    subject = Subject()

    first = _perceived_event(log, "s1", "first", subject=subject)
    second = _perceived_event(log, "s1", "second", subject=subject)
    projection.apply(first)
    projection.apply(second)
    projection.mark_anchor("s1", "perception:turn-first")
    before = projection.get("s1", subject)

    with log.store.immediate_transaction() as conn:
        conn.execute("DROP TABLE context_window_view")

    rebuilt = ContextWindowProjection(log, recent_limit=3)
    registry = ProjectionRegistry()
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()

    after = rebuilt.get("s1", subject)

    assert [str(item.id) for item in after.foreground] == [
        str(item.id) for item in before.foreground
    ]
    assert after.counterpart == before.counterpart


def _sqlite_log(tmp_path) -> SQLiteEventLog:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return SQLiteEventLog(store)


def _perceived_event(
    log: SQLiteEventLog,
    session_id: str,
    payload: str,
    *,
    subject: Subject,
    counterpart=None,
):
    del subject
    turn_id = f"turn-{payload}"
    counterpart_record = counterpart.to_record() if counterpart is not None else None
    return EventEmitter(log).emit(
        CognitiveEventKind.PERCEIVED,
        outputs=[Reference("perception", f"perception:{turn_id}")],
        payload={
            "turn_id": turn_id,
            "session_id": session_id,
            "stimulus_kind": StimulusKind.USER_MESSAGE.value,
            "source": {"kind": "session", "id": session_id},
            "from_counterpart": counterpart_record,
            "source_refs": [
                {"kind": "session", "id": session_id},
                {"kind": "session_message", "id": f"msg-{payload}"},
            ],
            "content_digest": f"digest-{payload}",
            "content_length": len(payload),
        },
    )
