from __future__ import annotations

from alpha_agent.cognition.controller import CognitiveController
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    Applicability,
    Belief,
    BeliefId,
    CognitiveEventKind,
    CognitiveType,
    CounterpartId,
    Instant,
    Lifecycle,
    NLStatement,
    Role,
    SituationId,
    Stimulus,
    StimulusKind,
    Subject,
    SubjectId,
    ThreadId,
    UpdatePolicy,
    ValueProfile,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.context_window import ContextWindowProjection
from alpha_agent.cognition.projections.procedure import ProcedureProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.stages.effector import Effector
from alpha_agent.cognition.stages.perceive import Perceiver
from alpha_agent.cognition.stages.types import Outcome
from alpha_agent.llm.base import ChatMessage, LLMResponse
from alpha_agent.state.store import StateStore
from alpha_agent.tools.registry import ToolRegistry


def test_context_window_foreground_roll_keeps_last_k_perceptions(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=3)
    thread_id = ThreadId.from_session("s1")
    subject = Subject()

    for index in range(8):
        event = _perceived_event(log, thread_id, f"message-{index}", subject=subject)
        projection.apply(event)

    window = projection.get(thread_id, subject)

    assert [perception.raw for perception in window.foreground] == [
        "message-5",
        "message-6",
        "message-7",
    ]


def test_context_window_thread_isolation(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=5)
    subject = Subject()
    thread_a = ThreadId.from_session("a")
    thread_b = ThreadId.from_session("b")

    projection.apply(_perceived_event(log, thread_a, "a-only", subject=subject))
    projection.apply(_perceived_event(log, thread_b, "b-only", subject=subject))

    assert [item.raw for item in projection.get(thread_a, subject).foreground] == ["a-only"]
    assert [item.raw for item in projection.get(thread_b, subject).foreground] == ["b-only"]


def test_context_window_counterpart_link_and_cognition_none(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log)
    subject = Subject()
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))
    first_thread = ThreadId.from_session("s1")
    second_thread = ThreadId.from_session("s2")
    cognition_thread = ThreadId.cognition(subject.id, "clock")

    projection.apply(
        _perceived_event(log, first_thread, "one", subject=subject, source=counterpart)
    )
    projection.apply(
        _perceived_event(log, second_thread, "two", subject=subject, source=counterpart)
    )
    projection.apply(_perceived_event(log, cognition_thread, "tick", subject=subject))

    assert projection.get(first_thread, subject).counterpart == counterpart
    assert projection.get(cognition_thread, subject).counterpart is None
    assert projection.list_threads_by_counterpart(counterpart) == [first_thread, second_thread]


def test_context_window_anchor_survives_foreground_roll(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=4)
    thread_id = ThreadId.from_session("s1")
    subject = Subject()

    anchor_event = _perceived_event(log, thread_id, "anchor", subject=subject)
    projection.apply(anchor_event)
    anchor_id = anchor_event.payload["perception"]["id"]
    projection.mark_anchor(thread_id, anchor_id)

    for index in range(10):
        projection.apply(_perceived_event(log, thread_id, f"later-{index}", subject=subject))

    window = projection.get(thread_id, subject)

    assert [perception.raw for perception in window.foreground] == [
        "anchor",
        "later-7",
        "later-8",
        "later-9",
    ]


def test_context_window_rebuild_from_event_log(tmp_path) -> None:
    log = _sqlite_log(tmp_path)
    projection = ContextWindowProjection(log, recent_limit=3)
    subject = Subject()
    thread_id = ThreadId.from_session("s1")

    first = _perceived_event(log, thread_id, "first", subject=subject)
    second = _perceived_event(log, thread_id, "second", subject=subject)
    projection.apply(first)
    projection.apply(second)
    projection.mark_anchor(thread_id, first.payload["perception"]["id"])
    before = projection.get(thread_id, subject)

    with log.store.immediate_transaction() as conn:
        conn.execute("DROP TABLE context_window_view")

    rebuilt = ContextWindowProjection(log, recent_limit=3)
    registry = ProjectionRegistry()
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()

    after = rebuilt.get(thread_id, subject)

    assert [item.raw for item in after.foreground] == [item.raw for item in before.foreground]
    assert after.counterpart == before.counterpart


def test_real_reactive_tick_populates_recalled_and_recent_judgments(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))
    thread_id = ThreadId.from_session("s1")
    belief_projection = BeliefProjection(store)
    belief_projection.apply(
        EventEmitter(log).emit(
            CognitiveEventKind.BELIEF_FORMED,
            payload={
                "belief": _belief(
                    "belief:user-a",
                    "User prefers Python.",
                    counterpart,
                ).to_record()
            },
        )
    )
    context_projection = ContextWindowProjection(log)
    captured_windows = []

    registry = ProjectionRegistry()
    registry.register(SubjectProjection(log))
    registry.register(belief_projection)
    registry.register(ProcedureProjection())
    registry.register(context_projection)
    controller = CognitiveController(
        event_log=log,
        projections=registry,
        llm=_StaticProvider(),
        tools=ToolRegistry(),
        effector=Effector(
            llm_provider=_StaticProvider(),
            tool_registry=ToolRegistry(),
            completion_runner=lambda _decision, view, _rendered: captured_windows.append(
                view.window
            )
            or Outcome(
                text="ok",
                tool_calls=[],
                tool_results=[],
                raw_llm_response=LLMResponse(content="ok", model="test", provider="static"),
            ),
        ),
    )

    controller.reactive_tick(
        Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=counterpart,
            payload="I prefer Python",
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=thread_id,
    )

    subject = registry.get_typed(SubjectProjection).current()
    window_after_tick = context_projection.get(thread_id, subject)

    assert [ref.id for ref in captured_windows[-1].recalled] == ["belief:user-a"]
    assert [ref.kind for ref in window_after_tick.recent_judgments] == ["judgment"]
    assert window_after_tick.recent_judgments[0].id


def _sqlite_log(tmp_path) -> SQLiteEventLog:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return SQLiteEventLog(store)


def _perceived_event(
    log: SQLiteEventLog,
    thread_id: ThreadId,
    payload: str,
    *,
    subject: Subject,
    source=None,
):
    return Perceiver().perceive(
        Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=source,
            payload=payload,
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        subject,
        emitter=EventEmitter(log),
        tick_id=f"tick-{payload}",
    ).event


def _belief(belief_id: str, content: str, about) -> Belief:
    return Belief(
        id=BeliefId(belief_id),
        subject=subject_ref(SubjectId("subject:self")),
        about=[about],
        object="python",
        content=NLStatement(content),
        cognitive_type=CognitiveType.PREFERENCE,
        structure=None,
        sources=[],
        confidence=0.8,
        applicability=Applicability("{}"),
        value_profile=ValueProfile(),
        relations=[],
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("holder"),
        action_orientation=[],
        update_policy=UpdatePolicy("{}"),
        status=Lifecycle("active"),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


class _StaticProvider:
    name = "static"

    def complete(self, messages: list[ChatMessage], **_kwargs) -> LLMResponse:
        return LLMResponse(content="ok", model="test", provider=self.name)
