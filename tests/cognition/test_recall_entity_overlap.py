from __future__ import annotations

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind, CognitiveType, entity_ref
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    BeliefRecallParams,
    BeliefSearchParams,
)
from alpha_agent.cognition.stages.types import AttentionFocus
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.test_belief_projection_apply import (
    belief,
    counterpart_a,
    counterpart_b,
    python_entity,
)


def test_recall_with_focus_entities_requires_entity_overlap(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in [
        belief("belief:python", "User A prefers Python.", about=[counterpart_a()]),
        belief("belief:rust", "User A prefers Rust.", about=[counterpart_a()], object_="rust"),
        belief(
            "belief:global-python",
            "Python uses indentation.",
            about=[],
            object_="python",
        ),
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

    recalled = projection.recall(
        BeliefRecallParams(
            focus=AttentionFocus(
                entities=[python_entity(), entity_ref("unrelated")],
                salient_claims=[],
                value_signals={},
            ),
            counterpart=counterpart_a(),
        )
    )

    assert [item.id for item in recalled] == ["belief:python", "belief:global-python"]


def test_recall_candidates_requires_actual_match_signal_not_scope_only(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in [
        belief("belief:python", "User A prefers Python.", about=[counterpart_a()]),
        belief("belief:rust", "User A prefers Rust.", about=[counterpart_a()], object_="rust"),
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

    candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="unmatched zeppelin",
            counterpart=counterpart_a(),
            include_global=False,
        )
    )

    assert candidates == []


def test_recall_candidates_retrieves_natural_language_query_through_terms_fts(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in [
        belief(
            "belief:examples",
            "User prefers Python examples.",
            about=[counterpart_a()],
            object_="Python examples",
        ),
        belief(
            "belief:other-user",
            "User prefers Python examples.",
            about=[counterpart_b()],
            object_="Python examples",
        ),
        belief(
            "belief:factual",
            "Python examples use indentation.",
            about=[counterpart_a()],
            object_="Python examples",
            cognitive_type=CognitiveType.FACTUAL,
        ),
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

    candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="what examples do I prefer?",
            counterpart=counterpart_a(),
            include_global=False,
            types=frozenset({CognitiveType.PREFERENCE}),
        )
    )

    assert [candidate.belief.id for candidate in candidates] == ["belief:examples"]
    assert "term_fts" in candidates[0].reasons
    assert candidates[0].term_rank is not None


def test_recall_candidates_entity_exact_uses_query_and_keyword_probes(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in [
        belief(
            "belief:python",
            "User A prefers Python.",
            about=[counterpart_a(), entity_ref("python")],
            object_="language preference",
        ),
        belief(
            "belief:openai",
            "User A uses OpenAI API.",
            about=[counterpart_a(), entity_ref("OpenAI API")],
            object_="api preference",
        ),
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

    query_candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="Python",
            counterpart=counterpart_a(),
            include_global=False,
        )
    )
    keyword_candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="unmatched",
            keywords=("OpenAI API",),
            counterpart=counterpart_a(),
            include_global=False,
        )
    )

    query_match = next(
        candidate for candidate in query_candidates if candidate.belief.id == "belief:python"
    )
    keyword_match = next(
        candidate for candidate in keyword_candidates if candidate.belief.id == "belief:openai"
    )
    assert "entity_exact" in query_match.reasons
    assert "entity_exact" in keyword_match.reasons


def test_recall_candidates_merges_entity_object_fts_and_substring_reasons(
    tmp_path,
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
            payload={
                "belief": belief(
                    "belief:tech",
                    "User uses OpenAI API v3.0.1 at src/alpha_agent/runtime/agent.py "
                    "for C++ examples.",
                    about=[counterpart_a(), entity_ref("OpenAI API")],
                    object_="OpenAI API client",
                ).to_record()
            },
            event_ids=event_ids,
            clock=clock,
        )
    )
    projection.apply(
        emit(
            log,
            CognitiveEventKind.BELIEF_FORMED,
            payload={
                "belief": belief(
                    "belief:global-tech",
                    "OpenAI API v3.0.1 has a migration guide.",
                    about=[],
                    object_="OpenAI API",
                ).to_record()
            },
            event_ids=event_ids,
            clock=clock,
        )
    )

    candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="Where is the OpenAI API v3.0.1 C++ path?",
            keywords=("src/alpha_agent/runtime/agent.py", "v3.0.1", "C++"),
            entities=("OpenAI API",),
            counterpart=counterpart_a(),
            include_global=False,
        )
    )

    assert [candidate.belief.id for candidate in candidates] == ["belief:tech"]
    assert set(candidates[0].reasons) >= {
        "entity_exact",
        "object_partial",
        "term_fts",
        "trigram_fts",
        "substring",
    }
    assert candidates[0].term_rank is not None
    assert candidates[0].trigram_rank is not None
