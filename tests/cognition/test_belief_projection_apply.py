from __future__ import annotations

import pytest

from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    CounterpartId,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    Role,
    SituationId,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
    counterpart_ref,
    entity_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    BeliefRecallParams,
    BeliefSearchParams,
    build_term_fts_query,
    build_trigram_fts_query,
)
from alpha_agent.state.store import StateStore


def belief(
    belief_id: str,
    content: str,
    *,
    about: list[Reference] | None = None,
    object_: str = "python",
    memory_kind: MemoryKind = MemoryKind.PREFERENCE,
    lifecycle: BeliefLifecycle = BeliefLifecycle.ACTIVE,
    held_since: str = "2026-01-01T00:00:00+00:00",
    scope: BeliefScope | None = None,
) -> AtomicBelief:
    about_refs = list(about or [])
    return AtomicBelief(
        id=BeliefId(belief_id),
        subject=subject_ref(SUBJECT_SELF),
        about=about_refs,
        object=object_,
        content=NLStatement(content),
        memory_kind=memory_kind,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=scope or _scope_for_about(about_refs),
        authority=Authority.USER_ASSERTED,
        structure=None,
        sources=[],
        validity=ValidityWindow(observed_at=Instant(held_since)),
        relations=[],
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        action_orientation=[],
        update_policy={"updates": "operation_driven"},
        lifecycle=lifecycle,
        held_since=Instant(held_since),
    )


def summary_belief(
    belief_id: str,
    content: str,
    *,
    about: list[Reference],
    object_: str = "profile",
    summary_kind: SummaryKind = SummaryKind.COUNTERPART_PROFILE,
    scope: BeliefScope = BeliefScope.COUNTERPART,
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId(belief_id),
        subject=subject_ref(SUBJECT_SELF),
        about=list(about),
        object=object_,
        content=NLStatement(content),
        summary_kind=summary_kind,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=scope,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        validity=ValidityWindow(observed_at=Instant(held_since)),
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        lifecycle=BeliefLifecycle.ACTIVE,
        held_since=Instant(held_since),
    )


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _fts_ids(store: StateStore, table: str) -> list[str]:
    with store.connect() as conn:
        rows = conn.execute(f"SELECT belief_id FROM {table} ORDER BY belief_id").fetchall()
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


def test_belief_models_enforce_atomic_and_summary_split() -> None:
    atomic_record = belief("belief:atomic", "User prefers Python.").to_record()
    with pytest.raises(ValueError, match="summary_kind"):
        AtomicBelief.from_record({**atomic_record, "summary_kind": "counterpart_profile"})

    summary_record = summary_belief(
        "belief:summary",
        "User profile summary.",
        about=[counterpart_a()],
    ).to_record()
    with pytest.raises(ValueError, match="memory_kind"):
        SummaryBelief.from_record({**summary_record, "memory_kind": "preference"})


def test_scope_requires_matching_about_reference() -> None:
    with pytest.raises(ValueError, match="counterpart-scoped"):
        belief(
            "belief:bad-scope",
            "User prefers Python.",
            about=[],
            scope=BeliefScope.COUNTERPART,
        )


def test_upsert_atomic_belief_stores_current_entity_state(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)

    projection.upsert_atomic(belief("belief:python", "User prefers Python."))

    materialized = projection.get_by_id(BeliefId("belief:python"))
    assert isinstance(materialized, AtomicBelief)
    assert materialized.content == "User prefers Python."
    assert materialized.memory_kind == MemoryKind.PREFERENCE
    assert materialized.lifecycle == BeliefLifecycle.ACTIVE
    assert [item.id for item in projection.list_active()] == ["belief:python"]


def test_upsert_atomic_belief_indexes_fts_without_duplicates(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)

    projection.upsert_atomic(
        belief(
            "belief:examples",
            "User prefers Python examples.",
            about=[counterpart_a(), python_entity()],
            object_="Python",
        )
    )

    assert _fts_ids(store, "belief_search_terms_fts") == ["belief:examples"]
    assert _fts_ids(store, "belief_search_trigram_fts") == ["belief:examples"]
    terms = _search_terms_for(store, "belief:examples")
    assert "python" in terms
    assert "examples" in terms
    assert "counterpart:user-a" in terms

    projection.upsert_atomic(
        belief(
            "belief:examples",
            "User prefers Rust snippets.",
            about=[counterpart_a(), entity_ref("rust")],
            object_="Rust",
        )
    )

    assert _fts_count(store, "belief_search_terms_fts", "belief:examples") == 1
    assert _fts_count(store, "belief_search_trigram_fts", "belief:examples") == 1
    terms = _search_terms_for(store, "belief:examples")
    assert "rust" in terms
    assert "snippets" in terms


def test_supersede_and_retract_mutate_lifecycle_directly(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    old_belief = belief("belief:old", "User prefers Python.")
    new_belief = belief("belief:new", "User prefers Rust.", object_="rust")
    projection.upsert_atomic(old_belief)

    projection.supersede_many([old_belief.id], new_belief, at="2026-01-02T00:00:00+00:00")

    materialized_old = projection.get_by_id(BeliefId("belief:old"))
    materialized_new = projection.get_by_id(BeliefId("belief:new"))
    assert isinstance(materialized_old, AtomicBelief)
    assert isinstance(materialized_new, AtomicBelief)
    assert materialized_old.lifecycle == BeliefLifecycle.SUPERSEDED
    assert materialized_old.superseded_by is not None
    assert materialized_old.superseded_by.id == "belief:new"
    assert materialized_new.lifecycle == BeliefLifecycle.ACTIVE
    assert [item.id for item in projection.list_active()] == ["belief:new"]

    projection.mark_lifecycle(
        materialized_new.id,
        BeliefLifecycle.RETRACTED,
        at="2026-01-03T00:00:00+00:00",
    )

    assert projection.list_active() == []
    assert _fts_ids(store, "belief_search_terms_fts") == []
    assert _fts_ids(store, "belief_search_trigram_fts") == []


def test_recall_filters_by_memory_kind_scope_and_lifecycle(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    for item in [
        belief("belief:a-python", "User A prefers Python.", about=[counterpart_a()]),
        belief("belief:global-fact", "Python uses indentation.", memory_kind=MemoryKind.FACT),
        belief(
            "belief:retracted",
            "User A used to prefer Go.",
            about=[counterpart_a()],
            lifecycle=BeliefLifecycle.RETRACTED,
        ),
    ]:
        projection.upsert_atomic(item)

    recalled = projection.recall(
        BeliefRecallParams(
            counterpart=counterpart_a(),
            memory_kinds=frozenset({MemoryKind.PREFERENCE}),
            scopes=frozenset({BeliefScope.COUNTERPART}),
        )
    )

    assert [item.id for item in recalled] == ["belief:a-python"]


def test_recall_explicit_non_counterpart_scope_without_counterpart(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    self_memory = belief(
        "belief:self-python",
        "The agent should prefer concise Python snippets.",
        about=[subject_ref(SUBJECT_SELF)],
        scope=BeliefScope.SELF,
    )
    projection.upsert_atomic(self_memory)
    projection.upsert_atomic(belief("belief:global-python", "Python uses indentation."))

    recalled = projection.recall(
        BeliefRecallParams(
            counterpart=None,
            include_global=False,
            scopes=frozenset({BeliefScope.SELF}),
        )
    )
    searched = projection.recall_candidates(
        BeliefSearchParams(
            query="concise Python",
            counterpart=None,
            include_global=False,
            scopes=frozenset({BeliefScope.SELF}),
        )
    )

    assert [item.id for item in recalled] == ["belief:self-python"]
    assert [candidate.belief.id for candidate in searched] == ["belief:self-python"]


@pytest.mark.parametrize(
    ("scope", "about", "belief_id", "content", "query"),
    [
        (
            BeliefScope.PROJECT,
            [Reference("project", "alpha-agent")],
            "belief:project-deploy",
            "The alpha-agent project deploy task uses the gateway daemon.",
            "deploy gateway",
        ),
        (
            BeliefScope.SELF,
            [subject_ref(SUBJECT_SELF)],
            "belief:self-style",
            "The agent should keep implementation notes concise.",
            "concise implementation",
        ),
    ],
)
def test_recall_explicit_non_counterpart_scope_with_counterpart_context(
    tmp_path,
    scope: BeliefScope,
    about: list[Reference],
    belief_id: str,
    content: str,
    query: str,
) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    projection.upsert_atomic(
        belief(
            belief_id,
            content,
            about=about,
            object_=scope.value,
            scope=scope,
        )
    )
    projection.upsert_atomic(
        belief(
            "belief:a-python",
            "User A prefers Python examples.",
            about=[counterpart_a()],
            object_="python",
        )
    )
    projection.upsert_atomic(belief("belief:global-python", "Python uses indentation."))

    recalled = projection.recall(
        BeliefRecallParams(
            counterpart=counterpart_a(),
            scopes=frozenset({scope}),
        )
    )
    searched = projection.recall_candidates(
        BeliefSearchParams(
            query=query,
            counterpart=counterpart_a(),
            scopes=frozenset({scope}),
        )
    )

    assert [item.id for item in recalled] == [belief_id]
    assert [candidate.belief.id for candidate in searched] == [belief_id]


def test_summary_beliefs_are_stored_separately_and_not_recalled_by_default(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    projection.upsert_atomic(belief("belief:preference", "User A prefers Python."))
    profile = summary_belief(
        "belief:profile",
        "User A likes concise Python examples.",
        about=[counterpart_a()],
    )
    projection.upsert_summary(profile)

    ordinary = projection.recall_candidates(BeliefSearchParams(query="Python"))
    explicit_summary = projection.recall_candidates(
        BeliefSearchParams(
            query="Python",
            counterpart=counterpart_a(),
            summary_kinds=frozenset({SummaryKind.COUNTERPART_PROFILE}),
        )
    )

    assert [candidate.belief.id for candidate in ordinary] == ["belief:preference"]
    assert [candidate.belief.id for candidate in explicit_summary] == ["belief:profile"]
    assert projection.latest_summary(
        summary_kind=SummaryKind.COUNTERPART_PROFILE,
        about=counterpart_a(),
    ) == profile


def test_fts_query_builders_escape_special_characters_and_skip_short_trigrams(
    tmp_path,
) -> None:
    store = _store(tmp_path)
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
            INSERT INTO belief_search_terms_fts
                (belief_table, belief_id, search_terms, object, about)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "atomic",
                "belief:special",
                'C++ C# v3.0.1 src/alpha_agent/runtime:agent.py "quoted" OpenAI API',
                "OpenAI API",
                "",
            ),
        )
        conn.execute(
            """
            INSERT INTO belief_search_trigram_fts
                (belief_table, belief_id, content, object, normalized_content)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "atomic",
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


def _scope_for_about(about: list[Reference]) -> BeliefScope:
    if any(ref.kind == "counterpart" for ref in about):
        return BeliefScope.COUNTERPART
    return BeliefScope.GLOBAL
