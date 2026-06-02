from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    Belief,
    CognitiveEventKind,
    CognitiveType,
    Reference,
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
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.test_belief_projection_apply import (
    belief,
    counterpart_a,
    counterpart_b,
)


def test_memory_recall_schema_is_strict_and_compact() -> None:
    definition = next(
        tool
        for tool in build_tool_registry().to_llm_tool_definitions()
        if tool.name == MEMORY_RECALL_TOOL_NAME
    )

    assert definition.strict is True
    assert definition.parameters == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "maxLength": 300},
            "keywords": {
                "type": "array",
                "maxItems": 12,
                "items": {"type": "string", "maxLength": 80},
            },
            "entities": {
                "type": "array",
                "maxItems": 8,
                "items": {"type": "string", "maxLength": 120},
            },
            "intent": {"type": "string", "maxLength": 120},
            "scope": {
                "type": "string",
                "enum": ["counterpart", "global", "both"],
            },
            "types": {
                "type": "array",
                "maxItems": 8,
                "items": {
                    "type": "string",
                    "enum": [
                        "factual",
                        "procedural",
                        "preference",
                        "value",
                        "causal",
                        "social",
                        "temporal",
                        "meta",
                        "concept",
                    ],
                },
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
            },
        },
        "required": ["query"],
    }


def test_memory_recall_queries_counterpart_and_global_beliefs(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:a-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
            _belief(
                "belief:b-python",
                "User B prefers Python jokes.",
                about=[counterpart_b()],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
            _belief(
                "belief:global-python",
                "Python uses indentation.",
                about=[],
                object_="python",
                cognitive_type=CognitiveType.FACTUAL,
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
                "content": "User A prefers Python examples.",
                "type": "preference",
                "scope": "counterpart",
            },
            {
                "content": "Python uses indentation.",
                "type": "factual",
                "scope": "global",
            },
        ]
    }


def test_memory_recall_accepts_structured_retrieval_intent_without_changing_output(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:a-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {
            "query": "Python",
            "keywords": ["examples", "Python"],
            "entities": ["Python", "coding examples"],
            "intent": "preference lookup",
        },
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert result.output == {
        "results": [
            {
                "content": "User A prefers Python examples.",
                "type": "preference",
                "scope": "counterpart",
            }
        ]
    }


def test_memory_recall_query_only_answers_natural_language(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:examples",
                "User prefers Python examples.",
                about=[counterpart_a()],
                object_="Python examples",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "what examples do I prefer?"},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert result.output == {
        "results": [
            {
                "content": "User prefers Python examples.",
                "type": "preference",
                "scope": "counterpart",
            }
        ]
    }


def test_memory_recall_uses_keywords_and_entities_when_query_is_loose(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:openai",
                "User A prefers OpenAI API examples.",
                about=[counterpart_a(), Reference(kind="entity", id="openai api")],
                object_="OpenAI API",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {
            "query": "which examples should I use?",
            "keywords": ["examples"],
            "entities": ["OpenAI API"],
            "types": ["preference"],
        },
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert result.output == {
        "results": [
            {
                "content": "User A prefers OpenAI API examples.",
                "type": "preference",
                "scope": "counterpart",
            }
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
            _belief(
                "belief:a-python",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
            ),
            _belief(
                "belief:global-python",
                "Python uses indentation.",
                about=[],
                object_="python",
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "scope": scope, "max_results": 8},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert [item["content"] for item in _results(result.output)] == expected_contents


def test_memory_recall_filters_types_and_bounds_results(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:preference",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
            ),
            _belief(
                "belief:factual",
                "Python uses indentation.",
                about=[],
                object_="python",
                cognitive_type=CognitiveType.FACTUAL,
            ),
            _belief(
                "belief:concept",
                "Python is a high-level language.",
                about=[],
                object_="python",
                cognitive_type=CognitiveType.CONCEPT,
            ),
        ],
    )

    factual = MemoryRecallTool().run(
        {
            "query": "Python",
            "types": ["factual"],
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

    assert _results(factual.output) == [
        {
            "content": "Python uses indentation.",
            "type": "factual",
            "scope": "global",
        }
    ]
    assert len(_results(bounded.output)) == 1


def test_memory_recall_excludes_counterpart_digest_beliefs(tmp_path: Path) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:digest",
                "Python appears in the stable profile.",
                about=[counterpart_a()],
                object_="counterpart_digest:counterpart:user-a",
                held_since="2026-01-01T00:00:00+00:00",
            ),
            _belief(
                "belief:preference",
                "User A prefers Python examples.",
                about=[counterpart_a()],
                object_="python",
                held_since="2026-01-01T00:00:01+00:00",
            ),
        ],
    )

    result = MemoryRecallTool().run(
        {"query": "Python", "max_results": 1},
        _tool_context(projection=projection, counterpart=counterpart_a()),
    )

    assert _results(result.output) == [
        {
            "content": "User A prefers Python examples.",
            "type": "preference",
            "scope": "counterpart",
        }
    ]


def test_memory_recall_counterpart_scope_without_context_returns_empty(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
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
            _belief(
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
    assert set(item) == {"content", "type", "scope"}


def test_memory_recall_scored_candidates_are_explainable_and_ordered(
    tmp_path: Path,
) -> None:
    projection = _projection_with_beliefs(
        tmp_path,
        [
            _belief(
                "belief:counterpart-exact",
                "User A prefers Python examples.",
                about=[counterpart_a(), Reference(kind="entity", id="python")],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
                confidence=0.6,
                held_since="2026-01-01T00:00:00+00:00",
            ),
            _belief(
                "belief:global-exact",
                "Python examples should be concise.",
                about=[],
                object_="python",
                cognitive_type=CognitiveType.PREFERENCE,
                confidence=0.9,
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
            types=frozenset({CognitiveType.PREFERENCE}),
        )
    )

    scored = score_belief_candidates(
        candidates,
        counterpart=counterpart_a(),
        requested_types=frozenset({CognitiveType.PREFERENCE}),
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
        "type:preference",
    }
    assert set(scored[1].reasons) >= {"scope:global", "type:preference"}
    assert scored[0].score > scored[1].score


def test_memory_recall_exact_entity_match_stably_beats_loose_fts_score(
    tmp_path: Path,
) -> None:
    exact_belief = _belief(
        "belief:old-exact-entity",
        "User A prefers Python examples.",
        about=[counterpart_a(), Reference(kind="entity", id="python")],
        object_="examples",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=0.1,
        held_since="2026-01-01T00:00:00+00:00",
    )
    loose_belief = _belief(
        "belief:new-loose-fts",
        "Python examples should include pytest fixtures.",
        about=[counterpart_a()],
        object_="examples",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=1.0,
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
        requested_types=frozenset({CognitiveType.PREFERENCE}),
        query_scope="counterpart",
    )

    assert scored[0].belief.id == "belief:old-exact-entity"
    assert scored[1].belief.id == "belief:new-loose-fts"
    assert scored[1].score > scored[0].score
    assert set(scored[0].reasons) >= {
        "entity_exact",
        "scope:counterpart",
        "type:preference",
    }
    assert set(scored[1].reasons) >= {
        "term_fts",
        "trigram_fts",
        "substring",
        "scope:counterpart",
        "type:preference",
    }


def test_memory_recall_exact_object_match_stably_beats_loose_fts_score(
    tmp_path: Path,
) -> None:
    exact_belief = _belief(
        "belief:old-exact-object",
        "User A prefers Python examples.",
        about=[counterpart_a()],
        object_="python",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=0.1,
        held_since="2026-01-01T00:00:00+00:00",
    )
    loose_belief = _belief(
        "belief:new-loose-fts",
        "Python examples should include pytest fixtures.",
        about=[counterpart_a()],
        object_="examples",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=1.0,
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
            reasons=("object_exact",),
            term_rank=None,
            trigram_rank=None,
        ),
    ]

    scored = score_belief_candidates(
        candidates,
        counterpart=counterpart_a(),
        requested_types=frozenset({CognitiveType.PREFERENCE}),
        query_scope="counterpart",
    )

    assert scored[0].belief.id == "belief:old-exact-object"
    assert scored[1].belief.id == "belief:new-loose-fts"
    assert scored[1].score > scored[0].score
    assert set(scored[0].reasons) >= {
        "object_exact",
        "scope:counterpart",
        "type:preference",
    }
    assert set(scored[1].reasons) >= {
        "term_fts",
        "trigram_fts",
        "substring",
        "scope:counterpart",
        "type:preference",
    }


def test_memory_recall_exact_matches_are_ranked_by_utility_within_exact_tier(
    tmp_path: Path,
) -> None:
    global_entity = _belief(
        "belief:global-entity-exact",
        "Python examples should be short.",
        about=[],
        object_="examples",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=0.1,
        held_since="2026-01-01T00:00:00+00:00",
    )
    counterpart_object = _belief(
        "belief:counterpart-object-exact",
        "User A prefers Python examples.",
        about=[counterpart_a()],
        object_="python",
        cognitive_type=CognitiveType.PREFERENCE,
        confidence=0.9,
        held_since="2026-01-02T00:00:00+00:00",
    )
    candidates = [
        BeliefSearchCandidate(
            belief=global_entity,
            reasons=("entity_exact",),
            term_rank=None,
            trigram_rank=None,
        ),
        BeliefSearchCandidate(
            belief=counterpart_object,
            reasons=("object_exact",),
            term_rank=None,
            trigram_rank=None,
        ),
    ]

    scored = score_belief_candidates(
        candidates,
        counterpart=counterpart_a(),
        requested_types=frozenset({CognitiveType.PREFERENCE}),
        query_scope="both",
    )

    assert [item.belief.id for item in scored] == [
        "belief:counterpart-object-exact",
        "belief:global-entity-exact",
    ]
    assert scored[0].score > scored[1].score
    assert set(scored[0].reasons) >= {
        "object_exact",
        "scope:counterpart",
        "type:preference",
    }
    assert set(scored[1].reasons) >= {
        "entity_exact",
        "scope:global",
        "type:preference",
    }


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
        ({"query": "Python", "types": "factual"}, "types"),
        ({"query": "Python", "types": ["factual"] * 9}, "types"),
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
        ({"query": "Python", "intent": 42}, "intent"),
        ({"query": "Python", "intent": "x" * 121}, "intent"),
        ({"query": "Python", "intent": ("x" * 120) + " "}, "intent"),
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
    beliefs: list[Belief],
) -> BeliefProjection:
    store = _store(tmp_path)
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in beliefs:
        projection.apply(
            emit(
                log,
                CognitiveEventKind.BELIEF_FORMED,
                payload={"belief": item.to_record()},
                event_ids=event_ids,
                clock=clock,
            )
        )
    return projection


def _belief(
    belief_id: str,
    content: str,
    *,
    about: list[Reference],
    object_: str,
    cognitive_type: CognitiveType = CognitiveType.PREFERENCE,
    confidence: float = 0.6,
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> Belief:
    return Belief.from_record(
        {
            **belief(
                belief_id,
                content,
                about=about,
                object_=object_,
                confidence=confidence,
                held_since=held_since,
            ).to_record(),
            "cognitive_type": cognitive_type.value,
        }
    )


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
