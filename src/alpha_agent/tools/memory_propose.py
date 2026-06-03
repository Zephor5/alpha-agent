"""Cognition memory update proposal tool."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    Applicability,
    Belief,
    BeliefId,
    CognitiveEvent,
    CognitiveEventKind,
    CognitiveType,
    DerivationTrace,
    EventId,
    Instant,
    Lifecycle,
    NLStatement,
    Reference,
    Role,
    SituationId,
    SituationRef,
    Subject,
    UpdatePolicy,
    ValueProfile,
    situation_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefSearchParams
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.tools.base import ToolExecutionContext, ToolResult
from alpha_agent.utils.ids import new_id

MEMORY_PROPOSE_TOOL_NAME = "memory_propose"
MEMORY_PROPOSE_CONTEXT_KEY = "memory_propose"

_ALLOWED_OPERATIONS = frozenset(
    {"append", "reinforce", "replace", "merge", "correct", "retract"}
)
_ALLOWED_MEMORY_TYPES = frozenset({"preference", "constraint", "procedure", "factual"})
_ALLOWED_SCOPES = frozenset({"counterpart", "global"})

Decision = Literal[
    "accepted",
    "pending_confirmation",
    "needs_target_selection",
    "rejected",
]
MemoryStatus = Literal[
    "accepted",
    "pending_confirmation",
    "needs_target_selection",
    "rejected",
    "mixed",
]
NextAction = Literal[
    "none",
    "retry_with_target",
    "ask_user_confirmation",
    "explain_rejection",
]


@dataclass(frozen=True)
class MemoryProposalContext:
    """Runtime turn write context injected by the tool loop."""

    turn_id: str
    session_id: str
    user_message_id: str
    turn_received_event_id: str
    emitter: EventEmitter
    apply_cognitive_event: Callable[[CognitiveEvent], None]
    subject: Subject
    situation: SituationRef
    counterpart: Reference | None
    llm_call_id: str
    llm_trace_ids: list[str]
    belief_projection: BeliefProjection | None = None


@dataclass(frozen=True)
class _MemoryRecord:
    type: str
    content: str
    evidence: str
    scope: str

    def to_payload(self) -> dict[str, str]:
        return {
            "type": self.type,
            "content": self.content,
            "evidence": self.evidence,
            "scope": self.scope,
        }


@dataclass(frozen=True)
class _ParsedUpdate:
    index: int
    operation: str
    targets: list[str]
    target_hint: str
    memory: _MemoryRecord | None
    reason: str
    errors: list[str] = field(default_factory=list)

    @property
    def evidence(self) -> str:
        return self.memory.evidence if self.memory is not None else ""

    def update_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "operation": self.operation,
            "targets": list(self.targets),
            "target_hint": self.target_hint,
            "reason": self.reason,
        }
        if self.memory is not None:
            payload["memory"] = self.memory.to_payload()
        return payload


@dataclass(frozen=True)
class _Candidate:
    id: str
    content: str
    type: str
    scope: str
    status: str
    relation_hint: str = "possibly_related"

    def to_output(self) -> dict[str, str]:
        return {
            "id": self.id,
            "content": self.content,
            "type": self.type,
            "scope": self.scope,
            "status": self.status,
            "relation_hint": self.relation_hint,
        }


@dataclass(frozen=True)
class _OperationPlan:
    decision: Decision
    operation: str
    reason: str
    target_beliefs: list[Belief] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    memory: _MemoryRecord | None = None
    emit_memory_proposed: bool = True

    @property
    def target_belief_ids(self) -> list[str]:
        return [str(belief.id) for belief in self.target_beliefs]


@dataclass
class _UpdateResult:
    proposal_id: str
    update_index: int
    operation: str
    decision: Decision
    reason: str
    target_belief_ids: list[str] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    new_belief_id: str | None = None

    def to_output(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "proposal_id": self.proposal_id,
            "update_index": self.update_index,
            "operation": self.operation,
            "decision": self.decision,
            "reason": self.reason,
            "target_belief_ids": list(self.target_belief_ids),
        }
        if self.new_belief_id is not None:
            output["new_belief_id"] = self.new_belief_id
        if self.candidates:
            output["candidates"] = [candidate.to_output() for candidate in self.candidates]
        return output


class MemoryProposeTool:
    """Accept model-proposed long-term memory updates."""

    name = MEMORY_PROPOSE_TOOL_NAME
    description = (
        "Propose explicit long-term memory updates. Use updates with operation append, "
        "reinforce, replace, merge, correct, or retract, and memory.type preference, "
        "constraint, procedure, or factual. Do not use for ordinary facts, transient "
        "context, guesses, or tool summaries. Returns accepted, pending_confirmation, "
        "needs_target_selection, rejected, or mixed with a next_action for the model."
    )
    strict = True
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "updates": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": [
                                "append",
                                "reinforce",
                                "replace",
                                "merge",
                                "correct",
                                "retract",
                            ],
                        },
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 5,
                        },
                        "target_hint": {"type": "string", "maxLength": 300},
                        "memory": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": [
                                        "preference",
                                        "constraint",
                                        "procedure",
                                        "factual",
                                    ],
                                },
                                "content": {"type": "string", "maxLength": 500},
                                "evidence": {"type": "string", "maxLength": 500},
                                "scope": {
                                    "type": "string",
                                    "enum": ["counterpart", "global"],
                                },
                            },
                            "required": ["type", "content", "evidence", "scope"],
                        },
                        "reason": {"type": "string", "maxLength": 500},
                    },
                    "required": ["operation", "targets", "target_hint", "reason"],
                },
            }
        },
        "required": ["updates"],
    }

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw_updates = arguments.get("updates")
        update_items = raw_updates if isinstance(raw_updates, list) else []
        memory_context = _memory_proposal_context(context.extensions)

        if memory_context is None:
            return ToolResult(
                name=self.name,
                output=_memory_output(
                    status="rejected",
                    next_action="explain_rejection",
                    results=[],
                ),
                metadata=_tool_metadata(cognitive_event_ids=[]),
            )

        if not isinstance(raw_updates, list) or not update_items:
            return ToolResult(
                name=self.name,
                output=_memory_output(
                    status="rejected",
                    next_action="explain_rejection",
                    results=[],
                ),
                metadata=_tool_metadata(cognitive_event_ids=[]),
            )

        cognitive_event_ids: list[str] = []
        results: list[_UpdateResult] = []
        too_many_updates = len(update_items) > 5
        for index, raw in enumerate(update_items):
            proposal_id = new_id("proposal")
            parsed = _parse_update(raw, index)
            if too_many_updates:
                parsed = _ParsedUpdate(
                    index=parsed.index,
                    operation=parsed.operation,
                    targets=parsed.targets,
                    target_hint=parsed.target_hint,
                    memory=parsed.memory,
                    reason=parsed.reason,
                    errors=[*parsed.errors, "too_many_updates"],
                )
            plan = _plan_update(parsed, memory_context)
            result = _UpdateResult(
                proposal_id=proposal_id,
                update_index=index,
                operation=plan.operation,
                decision=plan.decision,
                reason=plan.reason,
                target_belief_ids=plan.target_belief_ids,
                candidates=plan.candidates,
            )
            proposed_event: CognitiveEvent | None = None
            if plan.emit_memory_proposed:
                proposed_event = _emit_memory_proposed(
                    context=memory_context,
                    proposal_id=proposal_id,
                    parsed=parsed,
                    plan=plan,
                    tool_call_id=context.tool_call_id,
                )
                cognitive_event_ids.append(str(proposed_event.id))

            if plan.decision == "accepted":
                emitted = _apply_accepted_update(
                    context=memory_context,
                    proposal_id=proposal_id,
                    parsed=parsed,
                    plan=plan,
                    proposed_event=proposed_event,
                    tool_call_id=context.tool_call_id,
                )
                result.new_belief_id = emitted.new_belief_id
                cognitive_event_ids.extend(emitted.event_ids)
            elif plan.decision == "pending_confirmation" and proposed_event is not None:
                pending = _emit_pending_confirmation(
                    context=memory_context,
                    proposal_id=proposal_id,
                    parsed=parsed,
                    plan=plan,
                    proposed_event=proposed_event,
                    tool_call_id=context.tool_call_id,
                )
                cognitive_event_ids.append(str(pending.id))
            results.append(result)

        status = _aggregate_status(results)
        return ToolResult(
            name=self.name,
            output=_memory_output(
                status=status,
                next_action=_aggregate_next_action(results),
                results=results,
            ),
            metadata=_tool_metadata(cognitive_event_ids=cognitive_event_ids),
        )


@dataclass(frozen=True)
class _AcceptedEmission:
    event_ids: list[str]
    new_belief_id: str | None = None


def _apply_accepted_update(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    proposed_event: CognitiveEvent | None,
    tool_call_id: str | None,
) -> _AcceptedEmission:
    parent_ids = [proposed_event.id] if proposed_event is not None else []
    if plan.operation == "reinforce":
        strengthened = _emit_belief_strengthened(
            context=context,
            proposal_id=proposal_id,
            parsed=parsed,
            plan=plan,
            causal_parents=parent_ids,
            tool_call_id=tool_call_id,
        )
        return _AcceptedEmission(event_ids=[str(strengthened.id)])
    if plan.operation == "retract":
        retracted = _emit_belief_retracted(
            context=context,
            proposal_id=proposal_id,
            parsed=parsed,
            plan=plan,
            causal_parents=parent_ids,
            tool_call_id=tool_call_id,
        )
        return _AcceptedEmission(event_ids=[str(retracted.id)])
    if plan.memory is None:
        return _AcceptedEmission(event_ids=[])

    extra_sources: list[Reference] = []
    if plan.operation == "merge":
        extra_sources = [Reference("belief", str(belief.id)) for belief in plan.target_beliefs]
    belief = build_belief_from_memory_update(
        memory=plan.memory,
        proposal_id=proposal_id,
        proposed_event_id=str(proposed_event.id) if proposed_event is not None else "",
        operation=plan.operation,
        reason=parsed.reason,
        context=context,
        extra_sources=extra_sources,
    )
    if plan.operation == "append":
        formed = _emit_belief_formed(
            context=context,
            proposal_id=proposal_id,
            parsed=parsed,
            plan=plan,
            belief=belief,
            causal_parents=parent_ids,
            tool_call_id=tool_call_id,
        )
        return _AcceptedEmission(event_ids=[str(formed.id)], new_belief_id=str(belief.id))
    if plan.operation == "replace":
        superseded = _emit_belief_superseded(
            context=context,
            proposal_id=proposal_id,
            parsed=parsed,
            plan=plan,
            old_belief=plan.target_beliefs[0],
            new_belief=belief,
            causal_parents=parent_ids,
            tool_call_id=tool_call_id,
        )
        return _AcceptedEmission(
            event_ids=[str(superseded.id)],
            new_belief_id=str(belief.id),
        )
    if plan.operation == "merge":
        formed = _emit_belief_formed(
            context=context,
            proposal_id=proposal_id,
            parsed=parsed,
            plan=plan,
            belief=belief,
            causal_parents=parent_ids,
            tool_call_id=tool_call_id,
        )
        event_ids = [str(formed.id)]
        for target in plan.target_beliefs:
            superseded = _emit_belief_superseded(
                context=context,
                proposal_id=proposal_id,
                parsed=parsed,
                plan=plan,
                old_belief=target,
                new_belief=belief,
                causal_parents=[formed.id],
                tool_call_id=tool_call_id,
            )
            event_ids.append(str(superseded.id))
        return _AcceptedEmission(event_ids=event_ids, new_belief_id=str(belief.id))
    return _AcceptedEmission(event_ids=[])


def build_belief_from_memory_update(
    *,
    memory: _MemoryRecord,
    proposal_id: str,
    proposed_event_id: str,
    operation: str,
    reason: str,
    context: MemoryProposalContext,
    extra_sources: list[Reference] | None = None,
) -> Belief:
    """Map an accepted memory update onto the existing Belief model."""

    about = _derived_about(memory.scope, context)
    sources = [*(extra_sources or []), Reference("session_message", context.user_message_id)]
    return Belief(
        id=BeliefId(new_id("belief")),
        subject=Reference("subject", str(context.subject.id)),
        about=about,
        object=_belief_object(memory_type=memory.type, scope=memory.scope, about=about),
        content=NLStatement(memory.content),
        cognitive_type=_belief_type(memory.type),
        structure=None,
        sources=sources,
        confidence=0.72,
        applicability=Applicability(
            deterministic_json(
                {
                    "source": MEMORY_PROPOSE_TOOL_NAME,
                    "scope": memory.scope,
                    "about": [item.to_record() for item in about],
                }
            )
        ),
        value_profile=ValueProfile(),
        relations=[],
        formed_in=context.situation,
        holder_role=Role(str(context.subject.role or "agent")),
        action_orientation=[],
        update_policy=UpdatePolicy(
            deterministic_json(
                {
                    "conflict": "model_target_required",
                    "updates": "operation_driven",
                }
            )
        ),
        status=Lifecycle("active"),
        held_since=Instant(context.emitter.clock()),
        derivation=DerivationTrace(
            deterministic_json(
                {
                    "source": MEMORY_PROPOSE_TOOL_NAME,
                    "proposal_id": proposal_id,
                    "memory_proposed_event_id": proposed_event_id,
                    "operation": operation,
                    "reason": reason,
                }
            )
        ),
    )


def _memory_proposal_context(extensions: Mapping[str, Any]) -> MemoryProposalContext | None:
    raw = extensions.get(MEMORY_PROPOSE_CONTEXT_KEY)
    if not isinstance(raw, Mapping):
        return None
    emitter = raw.get("emitter")
    apply_cognitive_event = raw.get("apply_cognitive_event")
    subject = raw.get("subject")
    situation = raw.get("situation")
    if not isinstance(emitter, EventEmitter):
        return None
    if not callable(apply_cognitive_event):
        return None
    if not isinstance(subject, Subject):
        return None
    if not isinstance(situation, Reference):
        situation = situation_ref(SituationId("situation:memory-propose"))
    turn_id = _non_empty_str(raw.get("turn_id"))
    session_id = _non_empty_str(raw.get("session_id"))
    user_message_id = _non_empty_str(raw.get("user_message_id"))
    if not all([turn_id, session_id, user_message_id]):
        return None
    counterpart = raw.get("counterpart")
    return MemoryProposalContext(
        turn_id=turn_id,
        session_id=session_id,
        user_message_id=user_message_id,
        turn_received_event_id=_non_empty_str(raw.get("turn_received_event_id")),
        emitter=emitter,
        apply_cognitive_event=apply_cognitive_event,
        subject=subject,
        situation=situation,
        counterpart=counterpart if isinstance(counterpart, Reference) else None,
        llm_call_id=_non_empty_str(raw.get("llm_call_id")),
        llm_trace_ids=_string_list(raw.get("llm_trace_ids")),
        belief_projection=raw.get("belief_projection")
        if isinstance(raw.get("belief_projection"), BeliefProjection)
        else None,
    )


def _parse_update(raw: object, index: int) -> _ParsedUpdate:
    if not isinstance(raw, Mapping):
        return _ParsedUpdate(
            index=index,
            operation="",
            targets=[],
            target_hint="",
            memory=None,
            reason="",
            errors=["update_not_object"],
        )
    operation = _string_field(raw.get("operation"), max_length=64)
    targets = _target_list(raw.get("targets"))
    target_hint = _string_field(raw.get("target_hint"), max_length=300)
    reason = _string_field(raw.get("reason"), max_length=500)
    memory, memory_errors = _parse_memory(raw.get("memory"))
    errors = list(memory_errors)
    if operation not in _ALLOWED_OPERATIONS:
        errors.append("invalid_operation")
    if not reason:
        errors.append("missing_reason")
    if raw.get("targets") is not None and not isinstance(raw.get("targets"), list):
        errors.append("invalid_targets")
    if operation != "retract" and memory is None:
        errors.append("missing_memory")
    return _ParsedUpdate(
        index=index,
        operation=operation,
        targets=targets,
        target_hint=target_hint,
        memory=memory,
        reason=reason,
        errors=errors,
    )


def _parse_memory(raw: object) -> tuple[_MemoryRecord | None, list[str]]:
    if raw is None:
        return None, []
    if not isinstance(raw, Mapping):
        return None, ["invalid_memory"]
    memory_type = _string_field(raw.get("type"), max_length=64)
    content = _string_field(raw.get("content"), max_length=500)
    evidence = _string_field(raw.get("evidence"), max_length=500)
    scope = _string_field(raw.get("scope"), max_length=64)
    errors: list[str] = []
    if memory_type not in _ALLOWED_MEMORY_TYPES:
        errors.append("invalid_memory_type")
    if not content:
        errors.append("missing_memory_content")
    if not evidence:
        errors.append("missing_memory_evidence")
    if scope not in _ALLOWED_SCOPES:
        errors.append("invalid_scope")
    if errors:
        return None, errors
    return _MemoryRecord(type=memory_type, content=content, evidence=evidence, scope=scope), []


def _plan_update(parsed: _ParsedUpdate, context: MemoryProposalContext) -> _OperationPlan:
    if parsed.errors:
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation or "invalid",
            reason="invalid_schema:" + ",".join(parsed.errors),
            memory=parsed.memory,
            emit_memory_proposed=False,
        )
    if (
        parsed.memory is not None
        and parsed.memory.scope == "counterpart"
        and context.counterpart is None
    ):
        return _OperationPlan(
            decision="pending_confirmation",
            operation=parsed.operation,
            reason="missing_counterpart_scope",
            memory=parsed.memory,
        )

    target_check = _validate_targets(parsed, context)
    if target_check.decision != "accepted":
        return target_check

    if parsed.operation == "append":
        return _plan_append(parsed, context, target_check.target_beliefs)
    if parsed.operation == "reinforce":
        if not target_check.target_beliefs:
            return _needs_targets(parsed, context, "reinforce_requires_target")
        return _OperationPlan(
            decision="accepted",
            operation="reinforce",
            reason="accepted_reinforce",
            target_beliefs=target_check.target_beliefs,
            memory=parsed.memory,
        )
    if parsed.operation == "replace":
        if len(target_check.target_beliefs) != 1:
            return _OperationPlan(
                decision="rejected",
                operation="replace",
                reason="replace_requires_exactly_one_target",
                memory=parsed.memory,
            )
        return _OperationPlan(
            decision="accepted",
            operation="replace",
            reason="accepted_replace",
            target_beliefs=target_check.target_beliefs,
            memory=parsed.memory,
        )
    if parsed.operation == "merge":
        if len(target_check.target_beliefs) < 2:
            return _OperationPlan(
                decision="rejected",
                operation="merge",
                reason="merge_requires_at_least_two_targets",
                target_beliefs=target_check.target_beliefs,
                memory=parsed.memory,
            )
        return _OperationPlan(
            decision="accepted",
            operation="merge",
            reason="accepted_merge",
            target_beliefs=target_check.target_beliefs,
            memory=parsed.memory,
        )
    if parsed.operation == "correct":
        if target_check.target_beliefs:
            return _OperationPlan(
                decision="pending_confirmation",
                operation="correct",
                reason="correct_requires_confirmation",
                target_beliefs=target_check.target_beliefs,
                memory=parsed.memory,
            )
        candidates = _candidate_outputs(parsed, context)
        if candidates:
            return _OperationPlan(
                decision="needs_target_selection",
                operation="correct",
                reason="correct_requires_target_selection",
                candidates=candidates,
                memory=parsed.memory,
            )
        return _OperationPlan(
            decision="pending_confirmation",
            operation="correct",
            reason="correct_requires_confirmed_target",
            memory=parsed.memory,
        )
    if parsed.operation == "retract":
        if not target_check.target_beliefs:
            return _needs_targets(
                parsed,
                context,
                "retract_requires_target",
                empty_decision="needs_target_selection",
            )
        if not parsed.evidence:
            return _OperationPlan(
                decision="pending_confirmation",
                operation="retract",
                reason="retract_requires_evidence",
                target_beliefs=target_check.target_beliefs,
                memory=parsed.memory,
            )
        return _OperationPlan(
            decision="accepted",
            operation="retract",
            reason="accepted_retract",
            target_beliefs=target_check.target_beliefs,
            memory=parsed.memory,
        )
    return _OperationPlan(
        decision="rejected",
        operation=parsed.operation,
        reason="invalid_operation",
        memory=parsed.memory,
    )


def _plan_append(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
    target_beliefs: list[Belief],
) -> _OperationPlan:
    if target_beliefs:
        return _OperationPlan(
            decision="accepted",
            operation="append",
            reason="accepted_append",
            target_beliefs=target_beliefs,
            memory=parsed.memory,
        )
    exact_duplicates = _exact_duplicate_beliefs(parsed, context)
    if exact_duplicates:
        return _OperationPlan(
            decision="accepted",
            operation="reinforce",
            reason="accepted_duplicate_reinforced",
            target_beliefs=[exact_duplicates[0]],
            memory=parsed.memory,
        )
    candidates = _candidate_outputs(parsed, context)
    if candidates:
        return _OperationPlan(
            decision="needs_target_selection",
            operation="append",
            reason="related_active_beliefs_require_target_selection",
            candidates=candidates,
            memory=parsed.memory,
        )
    return _OperationPlan(
        decision="accepted",
        operation="append",
        reason="accepted_append",
        memory=parsed.memory,
    )


def _needs_targets(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
    fallback_reason: str,
    *,
    empty_decision: Decision = "rejected",
) -> _OperationPlan:
    candidates = _candidate_outputs(parsed, context, include_exact=True)
    if candidates:
        return _OperationPlan(
            decision="needs_target_selection",
            operation=parsed.operation,
            reason=fallback_reason,
            candidates=candidates,
            memory=parsed.memory,
        )
    return _OperationPlan(
        decision=empty_decision,
        operation=parsed.operation,
        reason=fallback_reason,
        memory=parsed.memory,
    )


def _validate_targets(parsed: _ParsedUpdate, context: MemoryProposalContext) -> _OperationPlan:
    if not parsed.targets:
        return _OperationPlan(
            decision="accepted",
            operation=parsed.operation,
            reason="targets_valid",
            memory=parsed.memory,
        )
    if len(set(parsed.targets)) != len(parsed.targets):
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="duplicate_targets",
            memory=parsed.memory,
        )
    projection = context.belief_projection
    if projection is None:
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="missing_belief_projection",
            memory=parsed.memory,
        )
    target_beliefs: list[Belief] = []
    for target_id in parsed.targets:
        belief = projection.get_by_id(target_id)
        if belief is None:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_not_found",
                memory=parsed.memory,
            )
        if str(belief.status) != "active":
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_not_active",
                memory=parsed.memory,
            )
        if not _target_scope_matches(belief, parsed.memory, context):
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_scope_mismatch",
                memory=parsed.memory,
            )
        if parsed.memory is not None and _memory_type_for_belief(belief) != parsed.memory.type:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_type_mismatch",
                memory=parsed.memory,
            )
        target_beliefs.append(belief)
    return _OperationPlan(
        decision="accepted",
        operation=parsed.operation,
        reason="targets_valid",
        target_beliefs=target_beliefs,
        memory=parsed.memory,
    )


def _target_scope_matches(
    belief: Belief,
    memory: _MemoryRecord | None,
    context: MemoryProposalContext,
) -> bool:
    scope = _scope_for_belief(belief)
    if memory is not None and scope != memory.scope:
        return False
    if scope == "counterpart":
        if context.counterpart is None:
            return False
        return any(
            ref.kind == context.counterpart.kind and ref.id == context.counterpart.id
            for ref in belief.about
        )
    return True


def _exact_duplicate_beliefs(parsed: _ParsedUpdate, context: MemoryProposalContext) -> list[Belief]:
    if parsed.memory is None or context.belief_projection is None:
        return []
    normalized = _normalized_content(parsed.memory.content)
    return [
        belief
        for belief in _same_scope_type_active_beliefs(parsed, context)
        if _normalized_content(belief.content) == normalized
    ]


def _candidate_outputs(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
    *,
    include_exact: bool = False,
) -> list[_Candidate]:
    if context.belief_projection is None:
        return []
    query = _candidate_query(parsed)
    if not query:
        return []
    counterpart = (
        context.counterpart
        if parsed.memory is None or parsed.memory.scope == "counterpart"
        else None
    )
    types = (
        frozenset({_belief_type(parsed.memory.type)}) if parsed.memory is not None else None
    )
    candidates = context.belief_projection.recall_candidates(
        BeliefSearchParams(
            query=query,
            counterpart=counterpart,
            include_global=parsed.memory is None or parsed.memory.scope == "global",
            types=types,
            limit=8,
        )
    )
    outputs: list[_Candidate] = []
    for candidate in candidates:
        belief = candidate.belief
        if str(belief.status) != "active":
            continue
        if parsed.memory is not None:
            if (
                not include_exact
                and _normalized_content(belief.content)
                == _normalized_content(parsed.memory.content)
            ):
                continue
            if _scope_for_belief(belief) != parsed.memory.scope:
                continue
            if _memory_type_for_belief(belief) != parsed.memory.type:
                continue
        outputs.append(_candidate_from_belief(belief))
        if len(outputs) == 3:
            break
    return outputs


def _candidate_query(parsed: _ParsedUpdate) -> str:
    if parsed.memory is None:
        return " ".join(item for item in [parsed.target_hint, parsed.reason] if item)
    return " ".join(
        item for item in [parsed.memory.content, parsed.target_hint, parsed.memory.evidence] if item
    )


def _same_scope_type_active_beliefs(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
) -> list[Belief]:
    if parsed.memory is None or context.belief_projection is None:
        return []
    if parsed.memory.scope == "counterpart" and context.counterpart is not None:
        candidates = context.belief_projection.recall_about(context.counterpart)
    else:
        candidates = context.belief_projection.list_active()
    return [
        belief
        for belief in candidates
        if str(belief.status) == "active"
        and _scope_for_belief(belief) == parsed.memory.scope
        and _memory_type_for_belief(belief) == parsed.memory.type
    ]


def _candidate_from_belief(belief: Belief) -> _Candidate:
    return _Candidate(
        id=str(belief.id),
        content=str(belief.content),
        type=_memory_type_for_belief(belief),
        scope=_scope_for_belief(belief),
        status=str(belief.status),
    )


def _emit_belief_formed(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    belief: Belief,
    causal_parents: list[EventId],
    tool_call_id: str | None,
) -> CognitiveEvent:
    formed = context.emitter.emit(
        CognitiveEventKind.BELIEF_FORMED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(belief.id))],
        rationale="Promoted accepted memory update.",
        causal_parents=causal_parents,
        payload={
            **_change_payload(
                context=context,
                parsed=parsed,
                plan=plan,
                tool_call_id=tool_call_id,
            ),
            "proposal_id": proposal_id,
            "origin": MEMORY_PROPOSE_TOOL_NAME,
            "new_belief_id": str(belief.id),
            "belief": belief.to_record(),
        },
    )
    context.apply_cognitive_event(formed)
    return formed


def _emit_belief_strengthened(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    causal_parents: list[EventId],
    tool_call_id: str | None,
) -> CognitiveEvent:
    target = plan.target_beliefs[0]
    strengthened = context.emitter.emit(
        CognitiveEventKind.BELIEF_STRENGTHENED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("belief", str(target.id)),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(target.id))],
        rationale="Reinforced an active belief through a memory update.",
        causal_parents=causal_parents,
        payload={
            **_change_payload(
                context=context,
                parsed=parsed,
                plan=plan,
                tool_call_id=tool_call_id,
            ),
            "proposal_id": proposal_id,
            "belief_id": str(target.id),
            "delta": 0.05,
        },
    )
    context.apply_cognitive_event(strengthened)
    return strengthened


def _emit_belief_superseded(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    old_belief: Belief,
    new_belief: Belief,
    causal_parents: list[EventId],
    tool_call_id: str | None,
    include_belief: bool = True,
) -> CognitiveEvent:
    payload: dict[str, Any] = {
        **_change_payload(
            context=context,
            parsed=parsed,
            plan=plan,
            tool_call_id=tool_call_id,
        ),
        "proposal_id": proposal_id,
        "origin": MEMORY_PROPOSE_TOOL_NAME,
        "old_belief_id": str(old_belief.id),
        "new_belief_id": str(new_belief.id),
    }
    if include_belief:
        payload["belief"] = new_belief.to_record()
    superseded = context.emitter.emit(
        CognitiveEventKind.BELIEF_SUPERSEDED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("belief", str(old_belief.id)),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(new_belief.id))],
        rationale="Superseded an active belief through an accepted memory update.",
        causal_parents=causal_parents,
        payload=payload,
    )
    context.apply_cognitive_event(superseded)
    return superseded


def _emit_belief_retracted(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    causal_parents: list[EventId],
    tool_call_id: str | None,
) -> CognitiveEvent:
    target = plan.target_beliefs[0]
    retracted = context.emitter.emit(
        CognitiveEventKind.BELIEF_RETRACTED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("belief", str(target.id)),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(target.id))],
        rationale="Retracted an active belief through a memory update.",
        causal_parents=causal_parents,
        payload={
            **_change_payload(
                context=context,
                parsed=parsed,
                plan=plan,
                tool_call_id=tool_call_id,
            ),
            "proposal_id": proposal_id,
            "belief_id": str(target.id),
        },
    )
    context.apply_cognitive_event(retracted)
    return retracted


def _emit_memory_proposed(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    tool_call_id: str | None,
) -> CognitiveEvent:
    source_refs = [
        Reference("session", context.session_id),
        Reference("session_message", context.user_message_id),
    ]
    audit_refs = _audit_refs(
        tool_call_id=tool_call_id,
        llm_call_id=context.llm_call_id,
        llm_trace_ids=context.llm_trace_ids,
    )
    scope = parsed.memory.scope if parsed.memory is not None else "global"
    return context.emitter.emit(
        CognitiveEventKind.MEMORY_PROPOSED,
        situation=context.situation,
        inputs=[Reference("session_message", context.user_message_id)],
        outputs=[Reference("memory_proposal", proposal_id)],
        rationale="Recorded foreground memory update proposal.",
        causal_parents=_event_ids([context.turn_received_event_id]),
        payload={
            **_change_payload(
                context=context,
                parsed=parsed,
                plan=plan,
                tool_call_id=tool_call_id,
            ),
            "proposal_id": proposal_id,
            "proposal": parsed.update_payload(),
            "derived_about": [
                item.to_record() for item in _derived_about(scope, context)
            ],
            "source_refs": [item.to_record() for item in source_refs],
            "audit_refs": [item.to_record() for item in audit_refs],
            "gate": {"decision": plan.decision, "reason": plan.reason},
        },
    )


def _emit_pending_confirmation(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    proposed_event: CognitiveEvent,
    tool_call_id: str | None,
) -> CognitiveEvent:
    return context.emitter.emit(
        CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION,
        situation=context.situation,
        inputs=[Reference("memory_proposal", proposal_id)],
        outputs=[],
        rationale="Memory update requires user confirmation before mutation.",
        causal_parents=[proposed_event.id],
        payload={
            **_change_payload(
                context=context,
                parsed=parsed,
                plan=plan,
                tool_call_id=tool_call_id,
            ),
            "proposal_id": proposal_id,
            "required_user_action": "confirm_memory_change",
            "candidate_change": {
                "operation": plan.operation,
                "memory": parsed.memory.to_payload() if parsed.memory is not None else None,
            },
            "conflict_belief_ids": plan.target_belief_ids,
        },
    )


def _change_payload(
    *,
    context: MemoryProposalContext,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
    tool_call_id: str | None,
) -> dict[str, Any]:
    return {
        "turn_id": context.turn_id,
        "session_id": context.session_id,
        "tool_call_id": tool_call_id or "",
        "operation": plan.operation,
        "target_belief_ids": plan.target_belief_ids,
        "reason": parsed.reason or plan.reason,
        "evidence": parsed.evidence or parsed.reason,
    }


def _audit_refs(
    *,
    tool_call_id: str | None,
    llm_call_id: str,
    llm_trace_ids: list[str],
) -> list[Reference]:
    refs: list[Reference] = []
    if tool_call_id:
        refs.append(Reference("tool_call", tool_call_id))
    if llm_call_id:
        refs.append(Reference("llm_call", llm_call_id))
    refs.extend(Reference("llm_trace", item) for item in llm_trace_ids)
    return refs


def _event_ids(values: list[str]) -> list[EventId]:
    return [EventId(value) for value in values if value]


def _derived_about(scope: str, context: MemoryProposalContext) -> list[Reference]:
    if scope == "counterpart" and context.counterpart is not None:
        return [context.counterpart]
    return []


def _belief_object(*, memory_type: str, scope: str, about: list[Reference]) -> str:
    if scope == "counterpart" and about:
        return f"{memory_type}:{about[0].id}"
    return f"{memory_type}:{scope}"


def _belief_type(memory_type: str) -> CognitiveType:
    if memory_type == "preference":
        return CognitiveType.PREFERENCE
    if memory_type == "factual":
        return CognitiveType.FACTUAL
    return CognitiveType.PROCEDURAL


def _memory_type_for_belief(belief: Belief) -> str:
    prefix = str(belief.object).split(":", 1)[0]
    if prefix in _ALLOWED_MEMORY_TYPES:
        return prefix
    if belief.cognitive_type == CognitiveType.PREFERENCE:
        return "preference"
    if belief.cognitive_type == CognitiveType.FACTUAL:
        return "factual"
    return "procedure"


def _scope_for_belief(belief: Belief) -> str:
    return "counterpart" if belief.about else "global"


def _target_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    targets: list[str] = []
    for item in value[:5]:
        target = _non_empty_str(item)
        if target:
            targets.append(target)
    return targets


def _aggregate_status(results: list[_UpdateResult]) -> MemoryStatus:
    decisions = {result.decision for result in results}
    if not decisions:
        return "rejected"
    if len(decisions) == 1:
        return results[0].decision
    return "mixed"


def _aggregate_next_action(results: list[_UpdateResult]) -> NextAction:
    decisions = {result.decision for result in results}
    if "pending_confirmation" in decisions:
        return "ask_user_confirmation"
    if "needs_target_selection" in decisions:
        return "retry_with_target"
    if "rejected" in decisions:
        return "explain_rejection"
    return "none"


def _memory_output(
    *,
    status: MemoryStatus,
    next_action: NextAction,
    results: list[_UpdateResult],
) -> dict[str, Any]:
    return {
        "status": status,
        "next_action": next_action,
        "results": [result.to_output() for result in results],
    }


def _tool_metadata(*, cognitive_event_ids: list[str]) -> dict[str, Any]:
    return {
        "cognitive_event_ids": list(cognitive_event_ids),
    }


def _non_empty_str(value: object) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""


def _string_field(value: object, *, max_length: int) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:max_length]


def _normalized_content(value: object) -> str:
    return " ".join(str(value).casefold().split())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if item is not None and str(item)]
