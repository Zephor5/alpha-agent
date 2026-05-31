"""Read-only long-term belief recall tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from alpha_agent.cognition.counterpart_profile import COUNTERPART_DIGEST_OBJECT_PREFIX
from alpha_agent.cognition.models import CognitiveType, NLStatement, Reference, entity_ref
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.cognition.stages.types import AttentionFocus
from alpha_agent.tools.base import JSONValue, ToolExecutionContext, ToolResult

MEMORY_RECALL_TOOL_NAME = "memory_recall"
MEMORY_RECALL_CONTEXT_KEY = "memory_recall"

_DEFAULT_SCOPE = "both"
_DEFAULT_MAX_RESULTS = 4
_MAX_QUERY_LENGTH = 300
_MAX_TYPES = 8
_MAX_RESULTS = 8
_RECALL_SCAN_LIMIT = 32
_ALLOWED_ARGUMENTS = frozenset({"query", "scope", "types", "max_results"})

type MemoryRecallScope = Literal["counterpart", "global", "both"]


@dataclass(frozen=True)
class MemoryRecallContext:
    """Runtime read context injected for belief recall."""

    session_id: str
    counterpart: Reference | None
    belief_projection: BeliefProjection


@dataclass(frozen=True)
class _RecallArguments:
    query: str
    scope: MemoryRecallScope
    types: frozenset[CognitiveType] | None
    max_results: int


class MemoryRecallTool:
    """Search active long-term beliefs through the belief projection."""

    name = MEMORY_RECALL_TOOL_NAME
    description = (
        "Search stable long-term beliefs when explicit memory lookup would help answer "
        "the current turn. Returns compact belief content only. Does not write memory; "
        "use memory_propose for explicit long-term memory write proposals."
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

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        parsed = _parse_arguments(arguments)
        recall_context = _memory_recall_context(context.extensions, context.session_id)

        if parsed.scope == "counterpart" and recall_context.counterpart is None:
            return ToolResult(name=self.name, output={"results": []})

        counterpart = (
            recall_context.counterpart if parsed.scope in {"counterpart", "both"} else None
        )
        include_global = parsed.scope in {"global", "both"}
        beliefs = recall_context.belief_projection.recall(
            BeliefRecallParams(
                focus=AttentionFocus(
                    entities=[entity_ref(parsed.query)],
                    salient_claims=[NLStatement(parsed.query)],
                    value_signals={},
                ),
                counterpart=counterpart,
                include_global=include_global,
                types=parsed.types,
                limit=max(_RECALL_SCAN_LIMIT, parsed.max_results),
            )
        )
        results: list[JSONValue] = []
        for item in beliefs:
            if str(item.object).startswith(COUNTERPART_DIGEST_OBJECT_PREFIX):
                continue
            result: dict[str, JSONValue] = {
                "content": str(item.content),
                "type": item.cognitive_type.value,
                "scope": _belief_scope(item.about, recall_context.counterpart),
            }
            results.append(result)
            if len(results) >= parsed.max_results:
                break
        return ToolResult(name=self.name, output={"results": results})


def _parse_arguments(arguments: Mapping[str, Any]) -> _RecallArguments:
    unexpected = sorted(set(arguments) - _ALLOWED_ARGUMENTS)
    if unexpected:
        raise ValueError(f"unexpected memory_recall argument: {unexpected[0]}")

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("memory_recall query must be a non-empty string")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError("memory_recall query exceeds 300 characters")

    raw_scope = arguments.get("scope", _DEFAULT_SCOPE)
    if raw_scope not in {"counterpart", "global", "both"}:
        raise ValueError("memory_recall scope must be one of counterpart, global, both")
    scope = raw_scope

    raw_types = arguments.get("types")
    types: frozenset[CognitiveType] | None = None
    if raw_types is not None:
        if not isinstance(raw_types, list):
            raise ValueError("memory_recall types must be an array")
        if len(raw_types) > _MAX_TYPES:
            raise ValueError("memory_recall types must contain at most 8 items")
        parsed_types: set[CognitiveType] = set()
        for raw_type in raw_types:
            if not isinstance(raw_type, str):
                raise ValueError("memory_recall types must contain string values")
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
        scope=scope,
        types=types,
        max_results=raw_max_results,
    )


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


def _belief_scope(belief_about: list[Reference], counterpart: Reference | None) -> str:
    if counterpart is None:
        return "global"
    if any(ref.kind == counterpart.kind and ref.id == counterpart.id for ref in belief_about):
        return "counterpart"
    return "global"


def _non_empty_str(value: object) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""
