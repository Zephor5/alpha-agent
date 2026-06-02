from __future__ import annotations

import pytest

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
    Reference,
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
from alpha_agent.cognition.projection_runner import ProjectionRunner
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    build_term_fts_query,
    build_trigram_fts_query,
)
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory


def belief(
    belief_id: str,
    content: str,
    *,
    about: list[Reference] | None = None,
    object_: str = "python",
    cognitive_type: CognitiveType = CognitiveType.PREFERENCE,
    confidence: float = 0.6,
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> Belief:
    return Belief(
        id=BeliefId(belief_id),
        subject=subject_ref(SubjectId("subject:self")),
        about=list(about or []),
        object=object_,
        content=NLStatement(content),
        cognitive_type=cognitive_type,
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


def _fts_ids(store: StateStore, table: str) -> list[str]:
    with store.connect() as conn:
        rows = conn.execute(
            f"SELECT belief_id FROM {table} ORDER BY belief_id"
        ).fetchall()
    return [str(row["belief_id"]) for row in rows]


def _fts_count(store: StateStore, table: str, belief_id: str | None = None) -> int:
    with store.connect() as conn:
        if belief_id is None:
            row = conn.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
        else:
            row = conn.execute(
                f"SELECT COUNT(*) AS count FROM {table} WHERE belief_id = ?",
                (belief_id,),
            ).fetchone()
    return int(row["count"])


def _search_terms_for(store: StateStore, belief_id: str) -> str:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT search_terms
            FROM belief_search_terms_fts
            WHERE belief_id = ?
            """,
            (belief_id,),
        ).fetchone()
    assert row is not None
    return str(row["search_terms"])


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


def test_apply_belief_formed_indexes_active_belief_in_fts_without_duplicates(tmp_path) -> None:
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
            payload={
                "belief": belief(
                    "belief:examples",
                    "User prefers Python examples.",
                    about=[python_entity()],
                    object_="Python",
                ).to_record()
            },
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:examples"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:examples"]
    terms = _search_terms_for(store, "belief:examples")
    assert "python" in terms
    assert "examples" in terms
    assert "user prefers python examples." in terms

    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={
                "belief": belief(
                    "belief:examples",
                    "User prefers Rust snippets.",
                    about=[entity_ref("rust")],
                    object_="Rust",
                ).to_record()
            },
            event_ids=event_ids,
            clock=clock,
        )
    )

    assert _fts_count(store, "belief_search_terms_fts", "belief:examples") == 1
    assert _fts_count(store, "belief_search_trigram_fts", "belief:examples") == 1
    terms = _search_terms_for(store, "belief:examples")
    assert "rust" in terms
    assert "snippets" in terms
    assert "python" not in terms


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

    materialized_old = projection.get_by_id(BeliefId("belief:old"))
    materialized_new = projection.get_by_id(BeliefId("belief:new"))
    assert materialized_old is not None
    assert materialized_new is not None
    assert materialized_old.superseded_by is not None
    assert materialized_new.supersedes is not None
    assert materialized_old.status == "superseded"
    assert materialized_old.superseded_by.id == "belief:new"
    assert materialized_new.status == "active"
    assert materialized_new.supersedes.id == "belief:old"
    assert [item.id for item in projection.list_active()] == ["belief:new"]
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:new"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:new"]


def test_apply_belief_superseded_payload_new_belief_updates_fts_in_one_lifecycle(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    old_belief = belief("belief:old", "User prefers Python.")
    new_belief = belief("belief:new", "User prefers Rust.", object_="rust")
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": old_belief.to_record()},
            event_ids=event_ids,
            clock=clock,
        )
    )

    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_SUPERSEDED,
            payload={"old_belief_id": "belief:old", "belief": new_belief.to_record()},
            event_ids=event_ids,
            clock=clock,
        )
    )

    materialized_old = projection.get_by_id(BeliefId("belief:old"))
    materialized_new = projection.get_by_id(BeliefId("belief:new"))
    assert materialized_old is not None
    assert materialized_new is not None
    assert materialized_old.status == "superseded"
    assert materialized_old.superseded_by is not None
    assert materialized_old.superseded_by.id == "belief:new"
    assert materialized_new.status == "active"
    assert materialized_new.supersedes is not None
    assert materialized_new.supersedes.id == "belief:old"
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:new"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:new"]
    assert _fts_count(store, "belief_search_terms_fts", "belief:new") == 1
    assert _fts_count(store, "belief_search_trigram_fts", "belief:new") == 1


def test_apply_belief_superseded_payload_new_belief_rolls_back_as_one_transaction(
    tmp_path,
    monkeypatch,
) -> None:
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
            payload={"belief": belief("belief:old", "User prefers Python.").to_record()},
            event_ids=event_ids,
            clock=clock,
        )
    )
    original_delete_fts = projection._delete_belief_fts

    def fail_on_old_fts_delete(conn, belief_id: str) -> None:
        if belief_id == "belief:old":
            raise RuntimeError("simulated supersede failure")
        original_delete_fts(conn, belief_id)

    monkeypatch.setattr(projection, "_delete_belief_fts", fail_on_old_fts_delete)

    with pytest.raises(RuntimeError, match="simulated supersede failure"):
        projection.apply(
            emit(
                log,
                CognitiveEventKind.BELIEF_SUPERSEDED,
                payload={
                    "old_belief_id": "belief:old",
                    "belief": belief(
                        "belief:new",
                        "User prefers Rust.",
                        object_="rust",
                    ).to_record(),
                },
                event_ids=event_ids,
                clock=clock,
            )
        )

    materialized_old = projection.get_by_id(BeliefId("belief:old"))
    assert materialized_old is not None
    assert materialized_old.status == "active"
    assert projection.get_by_id(BeliefId("belief:new")) is None
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:old"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:old"]


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

    materialized = projection.get_by_id(BeliefId("belief:python"))
    assert materialized is not None
    assert materialized.status == "retracted"
    assert projection.list_active() == []
    assert _fts_ids(store, "belief_search_terms_fts") == []
    assert _fts_ids(store, "belief_search_trigram_fts") == []


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

    materialized = projection.get_by_id(BeliefId("belief:python"))
    assert materialized is not None
    assert materialized.status == "archived"
    assert projection.list_active() == []
    assert _fts_ids(store, "belief_search_terms_fts") == []
    assert _fts_ids(store, "belief_search_trigram_fts") == []


def test_reset_clears_belief_fts_tables(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={"belief": belief("belief:python", "User prefers Python.").to_record()},
        )
    )
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:python"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:python"]

    projection.reset()

    assert _fts_ids(store, "belief_search_terms_fts") == []
    assert _fts_ids(store, "belief_search_trigram_fts") == []


def test_replay_and_auto_rebuild_restore_belief_fts_consistently(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()

    for item in [
        belief("belief:old", "User prefers Python."),
        belief("belief:new", "User prefers Rust.", object_="rust"),
    ]:
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
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:new"]

    registry = ProjectionRegistry()
    rebuilt = BeliefProjection(store)
    registry.register(rebuilt)
    ProjectionRunner(log, registry).replay_all()
    ProjectionRunner(log, registry).replay_all()

    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:new"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:new"]
    assert _fts_count(store, "belief_search_terms_fts", "belief:new") == 1
    assert _fts_count(store, "belief_search_trigram_fts", "belief:new") == 1

    rebuilt.reset()
    auto_rebuilt = BeliefProjection(store, event_log=log, auto_rebuild=True)

    assert [item.id for item in auto_rebuilt.list_active()] == ["belief:new"]
    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:new"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:new"]


def test_fts_query_builders_escape_special_characters_and_skip_short_trigrams(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    BeliefProjection(store)
    term_query = build_term_fts_query(
        [
            "C++",
            "C#",
            "v3.0.1",
            "src/alpha_agent/runtime:agent.py",
            '"quoted"',
            "OpenAI API",
        ]
    )
    trigram_query = build_trigram_fts_query(
        [
            "C++",
            "C#",
            "ab",
            "v3.0.1",
            "src/alpha_agent/runtime:agent.py",
            '"quoted"',
        ]
    )

    assert term_query
    assert trigram_query
    assert build_term_fts_query(["", "   "]) == ""
    assert build_trigram_fts_query(["C#", "ab", "  "]) == ""

    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO belief_search_terms_fts (belief_id, search_terms, object)
            VALUES (?, ?, ?)
            """,
            (
                "belief:special",
                'C++ C# v3.0.1 src/alpha_agent/runtime:agent.py "quoted" OpenAI API',
                "OpenAI API",
            ),
        )
        conn.execute(
            """
            INSERT INTO belief_search_trigram_fts
                (belief_id, content, object, normalized_content)
            VALUES (?, ?, ?, ?)
            """,
            (
                "belief:special",
                'C++ C# v3.0.1 src/alpha_agent/runtime:agent.py "quoted"',
                "OpenAI API",
                'c++ c# v3.0.1 src/alpha_agent/runtime:agent.py "quoted"',
            ),
        )

    with store.connect() as conn:
        term_rows = conn.execute(
            """
            SELECT belief_id
            FROM belief_search_terms_fts
            WHERE belief_search_terms_fts MATCH ?
            """,
            (term_query,),
        ).fetchall()
        trigram_rows = conn.execute(
            """
            SELECT belief_id
            FROM belief_search_trigram_fts
            WHERE belief_search_trigram_fts MATCH ?
            """,
            (trigram_query,),
        ).fetchall()

    assert [row["belief_id"] for row in term_rows] == ["belief:special"]
    assert [row["belief_id"] for row in trigram_rows] == ["belief:special"]



def counterpart_a():
    return counterpart_ref(CounterpartId("counterpart:user-a"))


def counterpart_b():
    return counterpart_ref(CounterpartId("counterpart:user-b"))


def python_entity():
    return entity_ref("python")
