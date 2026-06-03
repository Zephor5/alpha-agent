from __future__ import annotations

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    BeliefId,
    CognitiveEventKind,
    ContextWindow,
    CounterpartId,
    Instant,
    Situation,
    SituationId,
    belief_ref,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.render.build_view import build_view
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, counterpart_payload, emit, id_factory
from tests.cognition.test_belief_projection_apply import belief


def test_build_view_resolves_subject_and_counterpart_without_prompt_recall(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))
    event_ids = id_factory()
    clock = clock_factory()
    emit(
        log,
        CognitiveEventKind.COUNTERPART_FIRST_OBSERVED,
        payload=counterpart_payload(),
        event_ids=event_ids,
        clock=clock,
    )
    for projection in projections.all():
        for event in log.iter():
            projection.apply(event)
    belief_projection = projections.get_typed(BeliefProjection)
    belief_projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={
                "belief": belief(
                    "belief:1",
                    "User prefers Python.",
                    about=[counterpart],
                ).to_record()
            },
            event_ids=event_ids,
            clock=clock,
        )
    )
    subject = projections.get_typed(SubjectProjection).current()
    situation = Situation(id=SituationId("situation:test"))
    window = ContextWindow(
        session_id="s1",
        counterpart=counterpart,
        foreground=[],
        background=None,
        recalled=[belief_ref(BeliefId("belief:1"))],
        matched_procedures=[],
        subject_at=subject_ref(subject.id),
        situation_at=situation_ref(situation.id),
        assembled_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    view = build_view(
        session_id=window.session_id,
        situation=situation,
        projections=projections,
        window=window,
    )

    assert view.subject.id == subject.id
    assert view.counterpart is not None
    assert view.counterpart.id == "counterpart:user-a"
    assert not hasattr(view, "recalled_beliefs")


def test_build_view_uses_explicit_counterpart_profile(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    subject = projections.get_typed(SubjectProjection).current()
    situation = Situation(id=SituationId("situation:test"))
    window = ContextWindow(
        session_id="s1",
        counterpart=None,
        foreground=[],
        background=None,
        recalled=[],
        matched_procedures=[],
        subject_at=subject_ref(subject.id),
        situation_at=situation_ref(situation.id),
        assembled_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    view = build_view(
        session_id=window.session_id,
        situation=situation,
        projections=projections,
        window=window,
        counterpart_profile="User prefers concise answers.",
    )

    assert view.counterpart_profile == "User prefers concise answers."


def test_build_view_does_not_fallback_to_all_active_beliefs_without_recalled_refs(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    projection = projections.get_typed(BeliefProjection)
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": belief("belief:global", "Global preference.").to_record()},
        )
    )
    subject = projections.get_typed(SubjectProjection).current()
    situation = Situation(id=SituationId("situation:test"))
    window = ContextWindow(
        session_id="s1",
        counterpart=None,
        foreground=[],
        background=None,
        recalled=[],
        matched_procedures=[],
        subject_at=subject_ref(subject.id),
        situation_at=situation_ref(situation.id),
        assembled_at=Instant("2026-01-01T00:00:00+00:00"),
    )

    view = build_view(
        session_id=window.session_id,
        situation=situation,
        projections=projections,
        window=window,
    )

    assert not hasattr(view, "recalled_beliefs")
