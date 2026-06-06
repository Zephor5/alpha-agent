"""Read-only long-term belief recall tool."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from alpha_agent.cognition.models import (
    AtomicBelief,
    BeliefLifecycle,
    BeliefScope,
    MemoryKind,
    Reference,
)
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    BeliefSearchCandidate,
    BeliefSearchParams,
)
from alpha_agent.tools.base import (
    JSONValue,
    ToolAvailability,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)

MEMORY_RECALL_TOOL_NAME = "memory_recall"
MEMORY_RECALL_CONTEXT_KEY = "memory_recall"

_DEFAULT_SCOPE = "both"
_DEFAULT_MAX_RESULTS = 4
_MAX_QUERY_LENGTH = 300
_MAX_KEYWORDS = 12
_MAX_KEYWORD_LENGTH = 80
_MAX_ENTITIES = 8
_MAX_ENTITY_LENGTH = 120
_MAX_TYPES = 8
_MAX_RESULTS = 8
_RECALL_SCAN_LIMIT = 32
_ALLOWED_ARGUMENTS = frozenset(
    {"query", "keywords", "entities", "scope", "types", "max_results"}
)
_PROTOCOL_MEMORY_TYPES = frozenset(kind.value for kind in MemoryKind)
_SCOPE_SCORE_COUNTERPART = 4.0
_SCOPE_SCORE_GLOBAL_IN_BOTH = 1.0
_SCOPE_SCORE_GLOBAL_ONLY = 3.0
_TYPE_SCORE = 2.0
_ENTITY_EXACT_SCORE = 4.0
_OBJECT_EXACT_SCORE = 3.0
_OBJECT_PARTIAL_SCORE = 1.0
_TERM_FTS_MAX_SCORE = 4.0
_TRIGRAM_FTS_MAX_SCORE = 2.0
_SUBSTRING_SCORE = 1.0
_RECENCY_TIEBREAK_MAX = 0.25
_EXACT_PRIORITY_EXACT = 0
_EXACT_PRIORITY_NONE = 1

type MemoryRecallScope = Literal["counterpart", "global", "both"]
type MemoryRecallResultScope = Literal["counterpart", "global"]


@dataclass(frozen=True)
class MemoryRecallContext:
    """Runtime read context injected for belief recall."""

    session_id: str
    counterpart: Reference | None
    belief_projection: BeliefProjection


@dataclass(frozen=True)
class _RecallArguments:
    query: str
    keywords: tuple[str, ...]
    entities: tuple[str, ...]
    scope: MemoryRecallScope
    memory_kinds: frozenset[MemoryKind] | None
    max_results: int


@dataclass(frozen=True)
class ScoredBeliefCandidate:
    """Internal scored recall candidate, exposed for deterministic tests/debugging."""

    belief: AtomicBelief
    scope: MemoryRecallResultScope
    exact_priority: int
    score: float
    reasons: tuple[str, ...]


class MemoryRecallTool:
    """Search active long-term atomic beliefs through the belief store."""

    spec = ToolSpec(
        name=MEMORY_RECALL_TOOL_NAME,
        description=(
            "Search stable long-term atomic beliefs when explicit memory lookup would help "
            "answer the current turn. Returns compact belief handles with id, content, "
            "memory_kind, scope, lifecycle, and held_since. Does not write memory; use "
            "memory_propose for explicit long-term memory write proposals."
        ),
        parameters={
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "type": "string",
                    "maxLength": 300,
                },
                "keywords": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "string",
                        "maxLength": 80,
                    },
                },
                "entities": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "string",
                        "maxLength": 120,
                    },
                },
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
                            "fact",
                            "preference",
                            "constraint",
                            "procedure",
                            "value",
                            "relationship",
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
        },
        toolset="memory",
        read_only=True,
        concurrency_safe=True,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        parsed = _parse_arguments(arguments)
        recall_context = _memory_recall_context(context.extensions, context.session_id)

        if parsed.scope == "counterpart" and recall_context.counterpart is None:
            return ToolResult(name=self.spec.name, output={"results": []})

        counterpart = (
            recall_context.counterpart if parsed.scope in {"counterpart", "both"} else None
        )
        include_global = parsed.scope in {"global", "both"}
        candidates = recall_context.belief_projection.recall_candidates(
            BeliefSearchParams(
                query=parsed.query,
                keywords=parsed.keywords,
                entities=parsed.entities,
                counterpart=counterpart,
                include_global=include_global,
                memory_kinds=parsed.memory_kinds,
                lifecycles=frozenset({BeliefLifecycle.ACTIVE}),
                limit=max(_RECALL_SCAN_LIMIT, parsed.max_results),
            )
        )
        atomic_candidates = [
            candidate
            for candidate in candidates
            if isinstance(candidate.belief, AtomicBelief)
            and candidate.belief.lifecycle == BeliefLifecycle.ACTIVE
        ]
        scored = score_belief_candidates(
            atomic_candidates,
            counterpart=recall_context.counterpart,
            requested_memory_kinds=parsed.memory_kinds,
            query_scope=parsed.scope,
        )
        results: list[JSONValue] = []
        for item in scored:
            belief = item.belief
            result: dict[str, JSONValue] = {
                "id": str(belief.id),
                "content": str(belief.content),
                "memory_kind": belief.memory_kind.value,
                "scope": item.scope,
                "lifecycle": belief.lifecycle.value,
                "held_since": str(belief.held_since),
            }
            results.append(result)
            if len(results) >= parsed.max_results:
                break
        return ToolResult(name=self.spec.name, output={"results": results})


def score_belief_candidates(
    candidates: Sequence[BeliefSearchCandidate],
    *,
    counterpart: Reference | None,
    requested_memory_kinds: frozenset[MemoryKind] | None,
    query_scope: MemoryRecallScope,
) -> list[ScoredBeliefCandidate]:
    """Score and sort merged projection candidates deterministically."""

    term_scores = _rank_score_by_belief_id(
        candidates,
        rank_kind="term",
        max_score=_TERM_FTS_MAX_SCORE,
    )
    trigram_scores = _rank_score_by_belief_id(
        candidates,
        rank_kind="trigram",
        max_score=_TRIGRAM_FTS_MAX_SCORE,
    )
    recency_scores = _recency_score_by_belief_id(candidates)

    scored: list[ScoredBeliefCandidate] = []
    for candidate in candidates:
        belief = candidate.belief
        if not isinstance(belief, AtomicBelief):
            continue
        belief_id = str(belief.id)
        projection_reasons = tuple(candidate.reasons)
        reason_set = set(projection_reasons)
        scope = _belief_scope(belief, counterpart)
        scorer_reasons = [f"scope:{scope}"]
        score = _scope_score(scope, query_scope)

        if requested_memory_kinds and belief.memory_kind in requested_memory_kinds:
            score += _TYPE_SCORE
            scorer_reasons.append(f"memory_kind:{belief.memory_kind.value}")
        if "entity_exact" in reason_set:
            score += _ENTITY_EXACT_SCORE
        if "object_exact" in reason_set:
            score += _OBJECT_EXACT_SCORE
        if "object_partial" in reason_set:
            score += _OBJECT_PARTIAL_SCORE
        if "substring" in reason_set:
            score += _SUBSTRING_SCORE

        score += term_scores.get(belief_id, 0.0)
        score += trigram_scores.get(belief_id, 0.0)
        score += recency_scores.get(belief_id, 0.0)

        scored.append(
            ScoredBeliefCandidate(
                belief=belief,
                scope=scope,
                exact_priority=_exact_priority(reason_set),
                score=score,
                reasons=_merge_reasons(projection_reasons, scorer_reasons),
            )
        )

    scored.sort(key=_scored_candidate_sort_key)
    return scored


def _parse_arguments(arguments: Mapping[str, Any]) -> _RecallArguments:
    unexpected = sorted(set(arguments) - _ALLOWED_ARGUMENTS)
    if unexpected:
        raise ValueError(f"unexpected memory_recall argument: {unexpected[0]}")

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("memory_recall query must be a non-empty string")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError("memory_recall query exceeds 300 characters")

    keywords = _parse_string_array(
        arguments,
        name="keywords",
        max_items=_MAX_KEYWORDS,
        max_item_length=_MAX_KEYWORD_LENGTH,
    )
    entities = _parse_string_array(
        arguments,
        name="entities",
        max_items=_MAX_ENTITIES,
        max_item_length=_MAX_ENTITY_LENGTH,
    )
    raw_scope = arguments.get("scope", _DEFAULT_SCOPE)
    if not isinstance(raw_scope, str) or raw_scope not in {"counterpart", "global", "both"}:
        raise ValueError("memory_recall scope must be one of counterpart, global, both")
    scope = cast(MemoryRecallScope, raw_scope)

    raw_types = arguments.get("types")
    memory_kinds: frozenset[MemoryKind] | None = None
    if raw_types is not None:
        if not isinstance(raw_types, list):
            raise ValueError("memory_recall types must be an array")
        if len(raw_types) > _MAX_TYPES:
            raise ValueError("memory_recall types must contain at most 8 items")
        parsed_types: set[MemoryKind] = set()
        for raw_type in raw_types:
            if not isinstance(raw_type, str):
                raise ValueError("memory_recall types must contain string values")
            if raw_type not in _PROTOCOL_MEMORY_TYPES:
                raise ValueError(f"memory_recall types contains unsupported value: {raw_type}")
            parsed_types.add(MemoryKind(raw_type))
        memory_kinds = frozenset(parsed_types)

    raw_max_results = arguments.get("max_results", _DEFAULT_MAX_RESULTS)
    if type(raw_max_results) is not int:
        raise ValueError("memory_recall max_results must be an integer")
    if raw_max_results < 1 or raw_max_results > _MAX_RESULTS:
        raise ValueError("memory_recall max_results must be between 1 and 8")

    return _RecallArguments(
        query=query.strip(),
        keywords=keywords,
        entities=entities,
        scope=scope,
        memory_kinds=memory_kinds,
        max_results=raw_max_results,
    )


def _parse_string_array(
    arguments: Mapping[str, Any],
    *,
    name: str,
    max_items: int,
    max_item_length: int,
) -> tuple[str, ...]:
    raw_values = arguments.get(name)
    if raw_values is None:
        return ()
    if not isinstance(raw_values, list):
        raise ValueError(f"memory_recall {name} must be an array")
    if len(raw_values) > max_items:
        raise ValueError(f"memory_recall {name} must contain at most {max_items} items")

    values: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise ValueError(f"memory_recall {name} must contain string values")
        if len(raw_value) > max_item_length:
            raise ValueError(
                f"memory_recall {name} items must be at most {max_item_length} characters"
            )
        value = raw_value.strip()
        if not value:
            raise ValueError(f"memory_recall {name} must contain non-empty string values")
        values.append(value)
    return tuple(values)


def _memory_recall_context(
    extensions: Mapping[str, Any],
    fallback_session_id: str,
) -> MemoryRecallContext:
    raw = extensions.get(MEMORY_RECALL_CONTEXT_KEY)
    if not isinstance(raw, Mapping):
        raise ValueError("memory_recall context is missing")
    projection = raw.get("belief_projection")
    if not isinstance(projection, BeliefProjection):
        raise ValueError("memory_recall context is missing belief_projection")
    counterpart = raw.get("counterpart")
    session_id = _non_empty_str(raw.get("session_id")) or fallback_session_id
    return MemoryRecallContext(
        session_id=session_id,
        counterpart=counterpart if isinstance(counterpart, Reference) else None,
        belief_projection=projection,
    )


def _belief_scope(
    belief: AtomicBelief,
    counterpart: Reference | None,
) -> MemoryRecallResultScope:
    if belief.scope == BeliefScope.COUNTERPART and counterpart is not None:
        if any(ref.kind == counterpart.kind and ref.id == counterpart.id for ref in belief.about):
            return "counterpart"
    return "global"


def _scope_score(scope: MemoryRecallResultScope, query_scope: MemoryRecallScope) -> float:
    if scope == "counterpart":
        return _SCOPE_SCORE_COUNTERPART
    if query_scope == "global":
        return _SCOPE_SCORE_GLOBAL_ONLY
    return _SCOPE_SCORE_GLOBAL_IN_BOTH


def _rank_score_by_belief_id(
    candidates: Sequence[BeliefSearchCandidate],
    *,
    rank_kind: Literal["term", "trigram"],
    max_score: float,
) -> dict[str, float]:
    ranks: list[tuple[str, float]] = []
    for candidate in candidates:
        rank = candidate.term_rank if rank_kind == "term" else candidate.trigram_rank
        if rank is None or not math.isfinite(rank):
            continue
        ranks.append((str(candidate.belief.id), rank))
    if not ranks:
        return {}

    min_rank = min(rank for _, rank in ranks)
    return {
        belief_id: max_score / (1.0 + max(0.0, rank - min_rank))
        for belief_id, rank in ranks
    }


def _recency_score_by_belief_id(
    candidates: Sequence[BeliefSearchCandidate],
) -> dict[str, float]:
    timestamps = [
        (str(candidate.belief.id), _held_since_timestamp(candidate.belief))
        for candidate in candidates
        if isinstance(candidate.belief, AtomicBelief)
    ]
    finite_timestamps = [
        (belief_id, timestamp)
        for belief_id, timestamp in timestamps
        if math.isfinite(timestamp)
    ]
    if not finite_timestamps:
        return {}

    min_timestamp = min(timestamp for _, timestamp in finite_timestamps)
    max_timestamp = max(timestamp for _, timestamp in finite_timestamps)
    if math.isclose(min_timestamp, max_timestamp):
        return {belief_id: 0.0 for belief_id, _ in finite_timestamps}

    spread = max_timestamp - min_timestamp
    return {
        belief_id: _RECENCY_TIEBREAK_MAX * (timestamp - min_timestamp) / spread
        for belief_id, timestamp in finite_timestamps
    }


def _scored_candidate_sort_key(candidate: ScoredBeliefCandidate) -> tuple[Any, ...]:
    return (
        candidate.exact_priority,
        -candidate.score,
        0 if candidate.scope == "counterpart" else 1,
        -_held_since_timestamp(candidate.belief),
        str(candidate.belief.id),
    )


def _exact_priority(reasons: set[str]) -> int:
    if "entity_exact" in reasons or "object_exact" in reasons:
        return _EXACT_PRIORITY_EXACT
    return _EXACT_PRIORITY_NONE


def _held_since_timestamp(belief: AtomicBelief) -> float:
    value = str(belief.held_since).strip()
    if not value:
        return float("-inf")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return float("-inf")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _merge_reasons(
    projection_reasons: Sequence[str],
    scorer_reasons: Sequence[str],
) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for reason in (*projection_reasons, *scorer_reasons):
        if reason in seen:
            continue
        seen.add(reason)
        merged.append(reason)
    return tuple(merged)


def _non_empty_str(value: object) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""
