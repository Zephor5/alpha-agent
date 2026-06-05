from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from alpha_agent.cognition.models import (
    AtomicBelief,
    BeliefLifecycle,
    MemoryKind,
    Reference,
    SummaryKind,
)
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    BeliefSearchCandidate,
    BeliefSearchParams,
)
from alpha_agent.runtime.tools import ToolExecutor
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import ToolCall, ToolExecutionContext
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_recall import (
    MEMORY_RECALL_CONTEXT_KEY,
    MEMORY_RECALL_TOOL_NAME,
    MemoryRecallTool,
    score_belief_candidates,
)
from alpha_agent.tools.registry import ToolRegistry
from tests.cognition.test_belief_projection_apply import (
    belief,
    counterpart_a,
    counterpart_b,
    summary_belief,
)


def test_memory_recall_schema_is_strict_and_exposes_new_memory_kinds_only() -> None:
    definition = next(
        tool
        for tool in build_tool_registry().to_llm_tool_definitions()
        if tool.name == MEMORY_RECALL_TOOL_NAME
    )

    assert definition.strict is True
    assert "memory_kind" in definition.description
    assert definition.parameters["properties"]["types"]["items"]["enum"] == [
        "fact",
        "preference",
        "constraint",
        "procedure",
        "value",
        "relationship",
    ]


def test_memory_recall_queries_counterpart_and_global_atomic_beliefs(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:a-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
            belief(
                "belief:b-python",
                "User B prefers Python jokes.",
                about=[counterpart_b()],
                object_="python",
            ),
            belief(
                "belief:global-python",
                "Python uses indentation.",
                about=[],
                object_="python",
                memory_kind=MemoryKind.FACT,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "max_results": 8},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert result.output == {
        "results": [
            {
                "id": "belief:a-python",
                "content": "User A prefers Python examples.",
                "memory_kind": "preference",
                "scope": "counterpart",
                "lifecycle": "active",
                "held_since": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "belief:global-python",
                "content": "Python uses indentation.",
                "memory_kind": "fact",
                "scope": "global",
                "lifecycle": "active",
                "held_since": "2026-01-01T00:00:00+00:00",
            },
        ]
    }


@pytest.mark.parametrize(
    ("scope", "expected_contents"),
    [
        ("counterpart", ["User A prefers Python examples."]),
        ("global", ["Python uses indentation."]),
        ("both", ["User A prefers Python examples.", "Python uses indentation."]),
    ],
)
def test_memory_recall_supports_scope(
    tmp_path: Path,
    scope: str,
    expected_contents: list[str],
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:a-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
            belief(
                "belief:global-python",
                "Python uses indentation.",
                about=[],
                object_="python",
                memory_kind=MemoryKind.FACT,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "scope": scope, "max_results": 8},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert [item["content"] for item in _results(result.output)] == expected_contents


def test_memory_recall_filters_memory_kinds_and_bounds_results(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:preference",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
            belief(
                "belief:fact",
                "Python uses indentation.",
                about=[],
                object_="python",
                memory_kind=MemoryKind.FACT,
            ),
            belief(
                "belief:value",
                "Correctness matters more than speed.",
                about=[],
                object_="python correctness",
                memory_kind=MemoryKind.VALUE,
            ),
        ],
    )

    fact_result = MemoryRecallTool().run(
        {
            "query": "Python",
            "types": ["fact"],
            "max_results": 8,
        },
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )
    bounded = MemoryRecallTool().run(
        {
            "query": "Python",
            "scope": "global",
            "max_results": 1,
        },
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert _results(fact_result.output) == [
        {
            "id": "belief:fact",
            "content": "Python uses indentation.",
            "memory_kind": "fact",
            "scope": "global",
            "lifecycle": "active",
            "held_since": "2026-01-01T00:00:00+00:00",
        }
    ]
    assert len(_results(bounded.output)) == 1


def test_memory_recall_outputs_constraint_from_memory_kind_not_object_prefix(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:constraint",
                "Do not write local machine-specific absolute paths into the repo.",
                about=[],
                object_="repository path rule",
                memory_kind=MemoryKind.CONSTRAINT,
            ),
            belief(
                "belief:procedure",
                "When editing repository paths, use project-root-relative paths.",
                about=[],
                object_="repository path workflow",
                memory_kind=MemoryKind.PROCEDURE,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {
            "query": "repository paths",
            "types": ["constraint"],
            "scope": "global",
            "max_results": 8,
        },
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert _results(result.output) == [
        {
            "id": "belief:constraint",
            "content": "Do not write local machine-specific absolute paths into the repo.",
            "memory_kind": "constraint",
            "scope": "global",
            "lifecycle": "active",
            "held_since": "2026-01-01T00:00:00+00:00",
        }
    ]


def test_memory_recall_excludes_summary_beliefs_by_default(tmp_path: Path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    projection.upsert_summary(
        summary_belief(
            "belief:profile",
            "Python appears in the stable profile.",
            about=[counterpart_a()],
        )
    )
    projection.upsert_atomic(
        belief(
            "belief:preference",
            "User A prefers Python examples.",
            about=[counterpart_a()],
            object_="python",
        )
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "max_results": 8},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert [item["id"] for item in _results(result.output)] == ["belief:preference"]
    explicit_summary = projection.recall_candidates(
        BeliefSearchParams(
            query="Python",
            counterpart=counterpart_a(),
            summary_kinds=frozenset({SummaryKind.COUNTERPART_PROFILE}),
        )
    )
    assert [candidate.belief.id for candidate in explicit_summary] == ["belief:profile"]


def test_memory_recall_returns_active_belief_handles_only(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:active-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
            belief(
                "belief:retracted-python",
                "User A used to prefer Python jokes.",
                about=[counterpart_a()],
                object_="python",
                lifecycle=BeliefLifecycle.RETRACTED,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "max_results": 8},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert [item["id"] for item in _results(result.output)] == ["belief:active-python"]


def test_memory_recall_counterpart_scope_without_context_returns_empty(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:preference",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "scope": "counterpart"},
        _tool_context(projection=projection, counterpart=None),
    )

    assert result.output == {"results": []}


def test_memory_recall_output_does_not_expose_internal_scoring(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:preference",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python"},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    [item] = _results(result.output)
    assert set(item) == {"id", "content", "memory_kind", "scope", "lifecycle", "held_since"}


def test_memory_recall_scored_candidates_are_explainable_and_ordered(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            belief(
                "belief:counterpart-exact",
                "User A prefers Python examples.",
                about=[counterpart_a(), Reference(kind="entity", id="python")],
                object_="python",
                memory_kind=MemoryKind.PREFERENCE,
                held_since="2026-01-01T00:00:00+00:00",
            ),
            belief(
                "belief:global-exact",
                "Python examples should be concise.",
                about=[],
                object_="python",
                memory_kind=MemoryKind.PREFERENCE,
                held_since="2026-01-01T00:00:01+00:00",
            ),
        ],
    )
    candidates = projection.recall_candidates(
        BeliefSearchParams(
            query="Python",
            entities=("python",),
            counterpart=counterpart_a(),
            include_global=True,
            memory_kinds=frozenset({MemoryKind.PREFERENCE}),
        )
    )

    scored = score_belief_candidates(
        candidates,
        counterpart=counterpart_a(),
        requested_memory_kinds=frozenset({MemoryKind.PREFERENCE}),
        query_scope="both",
    )

    assert [item.belief.id for item in scored] == [
        "belief:counterpart-exact",
        "belief:global-exact",
    ]
    assert set(scored[0].reasons) >= {
        "entity_exact",
        "object_exact",
        "term_fts",
        "substring",
        "scope:counterpart",
        "memory_kind:preference",
    }
    assert set(scored[1].reasons) >= {"scope:global", "memory_kind:preference"}
    assert scored[0].score > scored[1].score


def test_memory_recall_exact_match_tier_beats_loose_fts_score() -> None:
    exact_belief = belief(
        "belief:old-exact-entity",
        "User A prefers Python examples.",
        about=[counterpart_a(), Reference(kind="entity", id="python")],
        object_="examples",
        memory_kind=MemoryKind.PREFERENCE,
        held_since="2026-01-01T00:00:00+00:00",
    )
    loose_belief = belief(
        "belief:new-loose-fts",
        "Python examples should include pytest fixtures.",
        about=[counterpart_a()],
        object_="examples",
        memory_kind=MemoryKind.PREFERENCE,
        held_since="2026-01-02T00:00:00+00:00",
    )
    candidates = [
        BeliefSearchCandidate(
            belief=loose_belief,
            reasons=("term_fts", "trigram_fts", "substring"),
            term_rank=-10.0,
            trigram_rank=-10.0,
        ),
        BeliefSearchCandidate(
            belief=exact_belief,
            reasons=("entity_exact",),
            term_rank=None,
            trigram_rank=None,
        ),
    ]

    scored = score_belief_candidates(
        candidates,
        counterpart=counterpart_a(),
        requested_memory_kinds=frozenset({MemoryKind.PREFERENCE}),
        query_scope="counterpart",
    )

    assert scored[0].belief.id == "belief:old-exact-entity"
    assert scored[1].belief.id == "belief:new-loose-fts"
    assert scored[1].score > scored[0].score


def test_memory_recall_empty_results_succeed(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(tmp_path, [])

    result = MemoryRecallTool().run(
        {"query": "missing"},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert result.output == {"results": []}


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "query"),
        ({"query": ""}, "query"),
        ({"query": "x" * 301}, "query"),
        ({"query": "Python", "scope": "local"}, "scope"),
        ({"query": "Python", "types": "fact"}, "types"),
        ({"query": "Python", "types": ["fact"] * 9}, "types"),
        ({"query": "Python", "types": ["unknown"]}, "types"),
        ({"query": "Python", "keywords": "examples"}, "keywords"),
        ({"query": "Python", "keywords": ["examples"] * 13}, "keywords"),
        ({"query": "Python", "keywords": ["x" * 81]}, "keywords"),
        ({"query": "Python", "keywords": [("x" * 80) + " "]}, "keywords"),
        ({"query": "Python", "keywords": [42]}, "keywords"),
        ({"query": "Python", "entities": "Python"}, "entities"),
        ({"query": "Python", "entities": ["Python"] * 9}, "entities"),
        ({"query": "Python", "entities": ["x" * 121]}, "entities"),
        ({"query": "Python", "entities": [("x" * 120) + " "]}, "entities"),
        ({"query": "Python", "entities": [42]}, "entities"),
        ({"query": "Python", "max_results": 0}, "max_results"),
        ({"query": "Python", "max_results": 9}, "max_results"),
        ({"query": "Python", "unexpected": True}, "unexpected"),
    ],
)
def test_memory_recall_invalid_arguments_raise_value_error(
    tmp_path: Path,
    arguments: dict[str, Any],
    match: str,
) -> None:
    projection = _projection_with_beliefs(tmp_path, [])

    with pytest.raises(ValueError, match=match):
        MemoryRecallTool().run(
            arguments,
            _tool_context(projection=projection, counterpart=counterpart_a()),
        )


def test_memory_recall_invalid_arguments_use_recoverable_tool_failure(
    tmp_path: Path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    registry = ToolRegistry()
    registry.register(MemoryRecallTool())
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[
            ToolCall(
                id="call_recall",
                name=MEMORY_RECALL_TOOL_NAME,
                arguments={"query": "Python", "keywords": "examples"},
            )
        ],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=True,
    )

    assert executed[0].result.metadata["failed"] is True
    assert executed[0].result.metadata["error_type"] == "ValueError"
    assert executed[0].trace.event_type == "tool.failed"
    assert "keywords" in executed[0].trace.content
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "tool.started",
        "tool.failed",
    ]


def _store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _projection_with_beliefs(
    tmp_path: Path,
    beliefs: list[AtomicBelief],
) -> BeliefProjection:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    for item in beliefs:
        projection.upsert_atomic(item)
    return projection


def _tool_context(
    *,
    projection: BeliefProjection,
    counterpart: Reference | None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_recall",
        output_dir=Path(".alpha-agent/tool-results"),
        check_canceled=lambda _stage: None,
        extensions={
            MEMORY_RECALL_CONTEXT_KEY: {
                "session_id": "s1",
                "counterpart": counterpart,
                "belief_projection": projection,
            }
        },
    )


def _results(output: object) -> list[dict[str, Any]]:
    assert isinstance(output, Mapping)
    results = output["results"]
    assert isinstance(results, list)
    return results
