"""Intermediate values passed through one reactive cognition tick."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from alpha_agent.cognition.models import (
    BeliefId,
    BeliefRef,
    CognitiveEvent,
    EntityRef,
    JudgmentRef,
    NLStatement,
    ProcedureRef,
    Reflection,
    ValueKind,
)
from alpha_agent.tools.base import ToolCall, ToolResult


@dataclass(frozen=True)
class Emitted[T]:
    """Stage output paired with the event that recorded it."""

    value: T
    event: CognitiveEvent


@dataclass(frozen=True)
class AttentionFocus:
    entities: list[EntityRef]
    salient_claims: list[NLStatement]
    value_signals: dict[ValueKind, float]


@dataclass(frozen=True)
class Interpretation:
    stance: Literal["consistent", "contradicting", "novel", "ambiguous"]
    supporting_beliefs: list[BeliefRef]
    contradicting_beliefs: list[BeliefRef]
    novel_claims: list[NLStatement]
    ambiguity_notes: list[str]
    source_text: str = ""


@dataclass(frozen=True)
class Outcome:
    text: str | None
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    raw_llm_response: Any
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Feedback:
    matched_expected: bool
    surprises: list[str] = field(default_factory=list)
    affected_belief_ids: list[BeliefId] = field(default_factory=list)
    formed_belief_ids: list[BeliefId] = field(default_factory=list)


@dataclass(frozen=True)
class Revision:
    kind: str
    content: str
    based_on: list[JudgmentRef]
    procedures: list[ProcedureRef] = field(default_factory=list)
    reflections: list[Reflection] = field(default_factory=list)
