from __future__ import annotations

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    Applicability,
    Belief,
    BeliefId,
    CognitiveEventKind,
    CognitiveType,
    CounterpartId,
    DerivationTrace,
    FeedbackEntry,
    Instant,
    Lifecycle,
    NLStatement,
    ReflectionId,
    Role,
    SituationId,
    SubjectId,
    UpdatePolicy,
    ValueProfile,
    counterpart_ref,
    entity_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory


def belief(
    belief_id: str,
    content: str,
    *,
    about: list[object] | None = None,
    object_: str = "python",
    confidence: float = 0.6,
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> Belief:
    return Belief(
        id=BeliefId(belief_id),
        subject=subject_ref(SubjectId("subject:self")),
        about=list(about or []),
        object=object_,
        content=NLStatement(content),
        cognitive_type=CognitiveType.PREFERENCE,
        structure=None,
        sources=[],
        confidence=confidence,
        applicability=Applicability("{}"),
        value_profile=ValueProfile(),
        relations=[],
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("holder"),
        action_orientation=[],
        update_policy=UpdatePolicy("{}"),
        status=Lifecycle("active"),
        held_since=Instant(held_since),
    )


def test_apply_belief_formed_materializes_view(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event = emit(
        log,
        CognitiveEventKind.BELIEF_FORMED,
        payload={"belief": belief("belief:python", "User prefers Python.").to_record()},
    )

    projection.apply(event)

    materialized = projection.get_by_id(BeliefId("belief:python"))
    assert materialized is not None
    assert materialized.content == "User prefers Python."
    assert [item.id for item in projection.list_active()] == ["belief:python"]


def test_belief_formed_round_trips_full_record_fields(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    original = Belief.from_record(
        {
            **belief("belief:full", "User prefers explicit plans.").to_record(),
            "derivation": DerivationTrace("derived-from-user-statement"),
            "feedback_history": [
                FeedbackEntry("confirmed-on-followup"),
                FeedbackEntry("used-successfully"),
            ],
            "self_audit": [
                {"kind": "reflection", "id": ReflectionId("reflection:phase03-audit")},
            ],
        }
    )

    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": original.to_record()},
        )
    )

    materialized = projection.get_by_id(BeliefId("belief:full"))
    assert materialized is not None
    assert materialized.derivation == original.derivation
    assert materialized.feedback_history == original.feedback_history
    assert materialized.self_audit == original.self_audit


def test_apply_belief_superseded_marks_old_and_keeps_new_active(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    old_belief = belief("belief:old", "User prefers Python.")
    new_belief = belief("belief:new", "User prefers Rust.")

    for item in [old_belief, new_belief]:
        projection.apply(
            emit(
                log,
                CognitiveEventKind.BELIEF_FORMED,
                payload={"belief": item.to_record()},
                event_ids=event_ids,
                clock=clock,
            )
        )
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            payload={"old_belief_id": "belief:old", "new_belief_id": "belief:new"},
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert projection.get_by_id(BeliefId("belief:old")).status == "superseded"
    assert projection.get_by_id(BeliefId("belief:old")).superseded_by.id == "belief:new"
    assert projection.get_by_id(BeliefId("belief:new")).status == "active"
    assert projection.get_by_id(BeliefId("belief:new")).supersedes.id == "belief:old"
    assert [item.id for item in projection.list_active()] == ["belief:new"]


def test_apply_belief_retracted_removes_from_active_scope(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": belief("belief:python", "User prefers Python.").to_record()},
            event_ids=event_ids,
            clock=clock,
        )
    )

    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_RETRACTED,
            payload={"belief_id": "belief:python", "reason": "user corrected preference"},
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert projection.get_by_id(BeliefId("belief:python")).status == "retracted"
    assert projection.list_active() == []


def test_apply_belief_archived_removes_from_active_scope(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": belief("belief:python", "User prefers Python.").to_record()},
            event_ids=event_ids,
            clock=clock,
        )
    )

    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_ARCHIVED,
            payload={"belief_id": "belief:python", "reason": "obsolete"},
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert projection.get_by_id(BeliefId("belief:python")).status == "archived"
    assert projection.list_active() == []


def counterpart_a():
    return counterpart_ref(CounterpartId("counterpart:user-a"))


def counterpart_b():
    return counterpart_ref(CounterpartId("counterpart:user-b"))


def python_entity():
    return entity_ref("python")
