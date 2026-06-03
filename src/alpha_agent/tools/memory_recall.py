"""Read-only long-term belief recall tool."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from alpha_agent.cognition.counterpart_profile import COUNTERPART_DIGEST_OBJECT_PREFIX
from alpha_agent.cognition.models import Belief, CognitiveType, Reference
from alpha_agent.cognition.projections.belief import (
    BeliefProjection,
    BeliefSearchCandidate,
    BeliefSearchParams,
)
from alpha_agent.tools.base import JSONValue, ToolExecutionContext, ToolResult

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
_PROTOCOL_MEMORY_TYPES = frozenset({"preference", "constraint", "procedure", "factual"})
_EXCLUDED_MEMORY_OBJECT_PREFIXES = (
    COUNTERPART_DIGEST_OBJECT_PREFIX,
    "counterpart_profile:",
)
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
    types: frozenset[CognitiveType] | None
    protocol_types: frozenset[str]
    procedural_broader_requested: bool
    max_results: int


@dataclass(frozen=True)
class ScoredBeliefCandidate:
    """Internal scored recall candidate, exposed for deterministic tests/debugging."""

    belief: Belief
    scope: MemoryRecallResultScope
    exact_priority: int
    score: float
    reasons: tuple[str, ...]


class MemoryRecallTool:
    """Search active long-term beliefs through the belief projection."""

    name = MEMORY_RECALL_TOOL_NAME
    description = (
        "Search stable long-term beliefs when explicit memory lookup would help answer "
        "the current turn. Returns compact belief handles with id, content, type, scope, "
        "status, and held_since. Does not write memory; use memory_propose for explicit "
        "long-term memory write proposals."
    )
    strict = True
    parameters = {
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
                        "factual",
                        "constraint",
                        "procedure",
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

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        parsed = _parse_arguments(arguments)
        recall_context = _memory_recall_context(context.extensions, context.session_id)

        if parsed.scope == "counterpart" and recall_context.counterpart is None:
            return ToolResult(name=self.name, output={"results": []})

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
                types=parsed.types,
                limit=max(_RECALL_SCAN_LIMIT, parsed.max_results),
            )
        )
        candidates = [
            candidate
            for candidate in candidates
            if _is_active_belief(candidate.belief)
            and not _is_excluded_memory_belief(candidate.belief)
            and _matches_protocol_types(candidate.belief, parsed)
        ]
        scored = score_belief_candidates(
            candidates,
            counterpart=recall_context.counterpart,
            requested_types=parsed.types,
            query_scope=parsed.scope,
        )
        results: list[JSONValue] = []
        for item in scored:
            belief = item.belief
            result: dict[str, JSONValue] = {
                "id": str(belief.id),
                "content": str(belief.content),
                "type": _memory_type_for_belief(belief),
                "scope": item.scope,
                "status": str(belief.status),
                "held_since": str(belief.held_since),
            }
            results.append(result)
            if len(results) >= parsed.max_results:
                break
        return ToolResult(name=self.name, output={"results": results})


def score_belief_candidates(
    candidates: Sequence[BeliefSearchCandidate],
    *,
    counterpart: Reference | None,
    requested_types: frozenset[CognitiveType] | None,
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
        belief_id = str(belief.id)
        projection_reasons = tuple(candidate.reasons)
        reason_set = set(projection_reasons)
        scope = _belief_scope(belief.about, counterpart)
        scorer_reasons = [f"scope:{scope}"]
        score = _scope_score(scope, query_scope)

        if requested_types and belief.cognitive_type in requested_types:
            score += _TYPE_SCORE
            scorer_reasons.append(f"type:{belief.cognitive_type.value}")
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
        score += _confidence_score(belief)
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
    types: frozenset[CognitiveType] | None = None
    protocol_types: set[str] = set()
    procedural_broader_requested = False
    if raw_types is not None:
        if not isinstance(raw_types, list):
            raise ValueError("memory_recall types must be an array")
        if len(raw_types) > _MAX_TYPES:
            raise ValueError("memory_recall types must contain at most 8 items")
        parsed_types: set[CognitiveType] = set()
        for raw_type in raw_types:
            if not isinstance(raw_type, str):
                raise ValueError("memory_recall types must contain string values")
            if raw_type in _PROTOCOL_MEMORY_TYPES:
                protocol_types.add(raw_type)
                parsed_types.add(_cognitive_type_for_memory_type(raw_type))
                continue
            if raw_type == CognitiveType.PROCEDURAL.value:
                procedural_broader_requested = True
            try:
                parsed_types.add(CognitiveType(raw_type))
            except ValueError as exc:
                raise ValueError(
                    f"memory_recall types contains unsupported value: {raw_type}"
                ) from exc
        types = frozenset(parsed_types)

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
        types=types,
        protocol_types=frozenset(protocol_types),
        procedural_broader_requested=procedural_broader_requested,
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
    belief_about: list[Reference],
    counterpart: Reference | None,
) -> MemoryRecallResultScope:
    if counterpart is None:
        return "global"
    if any(ref.kind == counterpart.kind and ref.id == counterpart.id for ref in belief_about):
        return "counterpart"
    return "global"


def _scope_score(scope: MemoryRecallResultScope, query_scope: MemoryRecallScope) -> float:
    if scope == "counterpart":
        return _SCOPE_SCORE_COUNTERPART
    if query_scope == "global":
        return _SCOPE_SCORE_GLOBAL_ONLY
    return _SCOPE_SCORE_GLOBAL_IN_BOTH


def _matches_protocol_types(belief: Belief, parsed: _RecallArguments) -> bool:
    if not parsed.protocol_types:
        return True
    if belief.cognitive_type != CognitiveType.PROCEDURAL:
        return True
    if parsed.procedural_broader_requested:
        return True
    return _memory_type_for_belief(belief) in parsed.protocol_types


def _memory_type_for_belief(belief: Belief) -> str:
    prefix = str(belief.object).split(":", 1)[0]
    if prefix in _PROTOCOL_MEMORY_TYPES:
        return prefix
    if belief.cognitive_type == CognitiveType.PREFERENCE:
        return "preference"
    if belief.cognitive_type == CognitiveType.FACTUAL:
        return "factual"
    if belief.cognitive_type == CognitiveType.PROCEDURAL:
        return "procedure"
    return belief.cognitive_type.value


def _cognitive_type_for_memory_type(memory_type: str) -> CognitiveType:
    if memory_type == "preference":
        return CognitiveType.PREFERENCE
    if memory_type == "factual":
        return CognitiveType.FACTUAL
    return CognitiveType.PROCEDURAL


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
        -_confidence_score(candidate.belief),
        -_held_since_timestamp(candidate.belief),
        str(candidate.belief.id),
    )


def _exact_priority(reasons: set[str]) -> int:
    if "entity_exact" in reasons or "object_exact" in reasons:
        return _EXACT_PRIORITY_EXACT
    return _EXACT_PRIORITY_NONE


def _confidence_score(belief: Belief) -> float:
    try:
        confidence = float(belief.confidence)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _held_since_timestamp(belief: Belief) -> float:
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


def _is_excluded_memory_belief(belief: Belief) -> bool:
    belief_object = str(belief.object)
    return belief_object.startswith(_EXCLUDED_MEMORY_OBJECT_PREFIXES)


def _is_active_belief(belief: Belief) -> bool:
    return str(belief.status) == "active"


def _non_empty_str(value: object) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""
