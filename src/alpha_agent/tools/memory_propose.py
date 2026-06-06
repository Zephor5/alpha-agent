"""Cognition memory update proposal tool."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from alpha_agent.cognition.domain_guidance import memory_propose_requires_confirmation
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    CognitiveEvent,
    CognitiveEventKind,
    DerivationStage,
    DerivationTrace,
    EventId,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    Role,
    SituationId,
    SituationRef,
    Subject,
    SummaryBelief,
    ValidityWindow,
    situation_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefSearchParams
from alpha_agent.cognition.state_service import CognitionSourceKind, CognitionStateStore
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.tools.base import (
    ToolAvailability,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from alpha_agent.utils.ids import new_id

MEMORY_PROPOSE_TOOL_NAME = "memory_propose"
MEMORY_PROPOSE_CONTEXT_KEY = "memory_propose"

_ALLOWED_OPERATIONS = frozenset(
    {"append_distinct", "reinforce", "replace", "merge", "correct", "retract"}
)
_ALLOWED_MEMORY_TYPES = frozenset(
    {"fact", "preference", "constraint", "procedure", "value", "relationship"}
)
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
    "review_candidates",
    "ask_user_confirmation",
    "explain_rejection",
]
_RESOLUTION_OPTIONS = (
    "append_distinct",
    "reinforce",
    "replace",
    "merge",
    "correct",
    "retract",
)


@dataclass(frozen=True)
class MemoryProposalContext:
    """Runtime turn write context injected by the tool loop."""

    turn_id: str
    session_id: str
    user_message_id: str
    turn_received_event_id: str
    emitter: EventEmitter
    subject: Subject
    situation: SituationRef
    counterpart: Reference | None
    llm_call_id: str
    llm_trace_ids: list[str]
    memory_state: CognitionStateStore | None = None
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
    target_belief_ids: list[str]
    reviewed_candidate_ids: list[str]
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
            "target_belief_ids": list(self.target_belief_ids),
            "reviewed_candidate_ids": list(self.reviewed_candidate_ids),
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
    target_beliefs: list[AtomicBelief] = field(default_factory=list)
    reviewed_candidate_beliefs: list[AtomicBelief] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    memory: _MemoryRecord | None = None
    emit_memory_proposed: bool = True

    @property
    def target_belief_ids(self) -> list[str]:
        return [str(belief.id) for belief in self.target_beliefs]

    @property
    def reviewed_candidate_ids(self) -> list[str]:
        return [str(belief.id) for belief in self.reviewed_candidate_beliefs]


@dataclass
class _UpdateResult:
    proposal_id: str
    update_index: int
    operation: str
    decision: Decision
    reason: str
    target_belief_ids: list[str] = field(default_factory=list)
    reviewed_candidate_ids: list[str] = field(default_factory=list)
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
        if self.reviewed_candidate_ids:
            output["reviewed_candidate_ids"] = list(self.reviewed_candidate_ids)
        if self.decision == "needs_target_selection":
            output["resolution_options"] = list(_RESOLUTION_OPTIONS)
        if self.new_belief_id is not None:
            output["new_belief_id"] = self.new_belief_id
        if self.candidates:
            output["candidates"] = [candidate.to_output() for candidate in self.candidates]
        return output


class MemoryProposeTool:
    """Accept model-proposed long-term memory updates."""

    spec = ToolSpec(
        name=MEMORY_PROPOSE_TOOL_NAME,
        description=(
            "Propose explicit long-term memories. append_distinct adds a new memory "
            "after reviewed_candidate_ids; reinforce, replace, merge, and retract use "
            "target_belief_ids; correct waits for confirmation. Not for transient facts, "
            "guesses, or tool summaries. Returns status and next_action."
        ),
        parameters={
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
                                    "append_distinct",
                                    "reinforce",
                                    "replace",
                                    "merge",
                                    "correct",
                                    "retract",
                                ],
                            },
                            "target_belief_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "maxItems": 5,
                            },
                            "reviewed_candidate_ids": {
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
                                            "fact",
                                            "preference",
                                            "constraint",
                                            "procedure",
                                            "value",
                                            "relationship",
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
                        "required": [
                            "operation",
                            "target_belief_ids",
                            "reviewed_candidate_ids",
                            "target_hint",
                            "reason",
                        ],
                    },
                }
            },
            "required": ["updates"],
        },
        toolset="memory",
        read_only=False,
        concurrency_safe=False,
        destructive=True,
        requires_user_interaction=False,
    )

    def check_available(self) -> ToolAvailability:
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw_updates = arguments.get("updates")
        update_items = raw_updates if isinstance(raw_updates, list) else []
        memory_context = _memory_proposal_context(context.extensions)

        if memory_context is None:
            return ToolResult(
                name=self.spec.name,
                output=_memory_output(
                    status="rejected",
                    next_action="explain_rejection",
                    results=[],
                ),
                metadata=_result_metadata(cognitive_event_ids=[]),
            )

        if not isinstance(raw_updates, list) or not update_items:
            return ToolResult(
                name=self.spec.name,
                output=_memory_output(
                    status="rejected",
                    next_action="explain_rejection",
                    results=[],
                ),
                metadata=_result_metadata(cognitive_event_ids=[]),
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
                    target_belief_ids=parsed.target_belief_ids,
                    reviewed_candidate_ids=parsed.reviewed_candidate_ids,
                    target_hint=parsed.target_hint,
                    memory=parsed.memory,
                    reason=parsed.reason,
                    errors=[*parsed.errors, "too_many_updates"],
                )
            plan = _plan_update(parsed, memory_context)
            plan = _apply_domain_guidance(parsed, memory_context, plan)
            result = _UpdateResult(
                proposal_id=proposal_id,
                update_index=index,
                operation=plan.operation,
                decision=plan.decision,
                reason=plan.reason,
                target_belief_ids=plan.target_belief_ids,
                reviewed_candidate_ids=plan.reviewed_candidate_ids,
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
            elif plan.decision == "pending_confirmation":
                pending = _apply_pending_update(
                    context=memory_context,
                    proposal_id=proposal_id,
                    parsed=parsed,
                    plan=plan,
                )
                result.new_belief_id = pending.new_belief_id
            results.append(result)

        status = _aggregate_status(results)
        return ToolResult(
            name=self.spec.name,
            output=_memory_output(
                status=status,
                next_action=_aggregate_next_action(results),
                results=results,
            ),
            metadata=_result_metadata(cognitive_event_ids=cognitive_event_ids),
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
    del proposed_event, tool_call_id
    memory_state = context.memory_state
    if memory_state is None:
        return _AcceptedEmission(event_ids=[])
    if plan.operation == "reinforce":
        target = plan.target_beliefs[0]
        memory_state.reaffirm_atomic_belief(
            target.id,
            source=Reference("session_message", context.user_message_id),
            observed_at=context.emitter.clock(),
            audit=_state_audit("memory_propose_reaffirm", proposal_id, parsed, plan),
        )
        return _AcceptedEmission(event_ids=[])
    if plan.operation == "retract":
        for target in plan.target_beliefs:
            memory_state.mark_belief_lifecycle(
                target.id,
                BeliefLifecycle.RETRACTED,
                at=context.emitter.clock(),
                audit=_state_audit("memory_propose_retract", proposal_id, parsed, plan),
            )
        return _AcceptedEmission(event_ids=[])
    if plan.memory is None:
        return _AcceptedEmission(event_ids=[])

    extra_sources: list[Reference] = []
    if plan.operation == "merge":
        extra_sources = [Reference("belief", str(belief.id)) for belief in plan.target_beliefs]
    belief = build_belief_from_memory_update(
        memory=plan.memory,
        proposal_id=proposal_id,
        proposed_event_id="",
        operation=plan.operation,
        reason=parsed.reason,
        target_hint=parsed.target_hint,
        context=context,
        extra_sources=extra_sources,
    )
    if plan.operation == "append_distinct":
        memory_state.write_atomic_belief(
            belief,
            source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
            audit=_state_audit("memory_propose_write", proposal_id, parsed, plan),
        )
        return _AcceptedEmission(event_ids=[], new_belief_id=str(belief.id))
    if plan.operation == "replace":
        memory_state.supersede_atomic_beliefs(
            [plan.target_beliefs[0].id],
            belief,
            source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
            at=context.emitter.clock(),
            audit=_state_audit("memory_propose_replace", proposal_id, parsed, plan),
        )
        return _AcceptedEmission(event_ids=[], new_belief_id=str(belief.id))
    if plan.operation == "merge":
        memory_state.supersede_atomic_beliefs(
            [target.id for target in plan.target_beliefs],
            belief,
            source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
            at=context.emitter.clock(),
            audit=_state_audit("memory_propose_merge", proposal_id, parsed, plan),
        )
        return _AcceptedEmission(event_ids=[], new_belief_id=str(belief.id))
    return _AcceptedEmission(event_ids=[])


def _apply_pending_update(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
) -> _AcceptedEmission:
    memory_state = context.memory_state
    if memory_state is None or plan.memory is None:
        return _AcceptedEmission(event_ids=[])
    if plan.memory.scope == "counterpart" and context.counterpart is None:
        return _AcceptedEmission(event_ids=[])
    extra_sources = [Reference("belief", str(belief.id)) for belief in plan.target_beliefs]
    belief = build_belief_from_memory_update(
        memory=plan.memory,
        proposal_id=proposal_id,
        proposed_event_id="",
        operation=plan.operation,
        reason=parsed.reason,
        target_hint=parsed.target_hint,
        context=context,
        extra_sources=extra_sources,
        lifecycle=BeliefLifecycle.PENDING_CONFIRMATION,
    )
    memory_state.write_atomic_belief(
        belief,
        source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
        audit=_state_audit("memory_propose_pending", proposal_id, parsed, plan),
    )
    return _AcceptedEmission(event_ids=[], new_belief_id=str(belief.id))


def build_belief_from_memory_update(
    *,
    memory: _MemoryRecord,
    proposal_id: str,
    proposed_event_id: str,
    operation: str,
    reason: str,
    target_hint: str,
    context: MemoryProposalContext,
    extra_sources: list[Reference] | None = None,
    lifecycle: BeliefLifecycle = BeliefLifecycle.ACTIVE,
) -> AtomicBelief:
    """Map an accepted memory update onto an atomic belief entity."""

    del proposed_event_id
    about = _derived_about(memory.scope, context)
    sources = [*(extra_sources or []), Reference("session_message", context.user_message_id)]
    now = context.emitter.clock()
    return AtomicBelief(
        id=BeliefId(new_id("belief")),
        subject=Reference("subject", str(context.subject.id)),
        about=about,
        object=_belief_object(memory=memory, target_hint=target_hint, about=about),
        content=NLStatement(memory.content),
        memory_kind=MemoryKind(memory.type),
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope(memory.scope),
        authority=Authority.USER_ASSERTED,
        structure=None,
        sources=sources,
        relations=[],
        formed_in=context.situation,
        holder_role=Role(str(context.subject.role or "agent")),
        action_orientation=[],
        update_policy={
            "conflict": "model_target_required",
            "updates": "operation_driven",
        },
        lifecycle=lifecycle,
        validity=ValidityWindow(observed_at=Instant(now)),
        held_since=Instant(now),
        derivation=DerivationTrace(
            deterministic_json(
                {
                    "source": MEMORY_PROPOSE_TOOL_NAME,
                    "proposal_id": proposal_id,
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
    subject = raw.get("subject")
    situation = raw.get("situation")
    if not isinstance(emitter, EventEmitter):
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
    memory_state = raw.get("memory_state")
    if not isinstance(memory_state, CognitionStateStore):
        memory_state = raw.get("memory_state_service")
    if not isinstance(memory_state, CognitionStateStore):
        memory_state = None
    raw_projection = raw.get("belief_projection")
    belief_projection = (
        raw_projection
        if isinstance(raw_projection, BeliefProjection)
        else memory_state.beliefs
        if memory_state is not None
        else None
    )
    return MemoryProposalContext(
        turn_id=turn_id,
        session_id=session_id,
        user_message_id=user_message_id,
        turn_received_event_id=_non_empty_str(raw.get("turn_received_event_id")),
        emitter=emitter,
        subject=subject,
        situation=situation,
        counterpart=counterpart if isinstance(counterpart, Reference) else None,
        llm_call_id=_non_empty_str(raw.get("llm_call_id")),
        llm_trace_ids=_string_list(raw.get("llm_trace_ids")),
        memory_state=memory_state,
        belief_projection=belief_projection,
    )


def _parse_update(raw: object, index: int) -> _ParsedUpdate:
    if not isinstance(raw, Mapping):
        return _ParsedUpdate(
            index=index,
            operation="",
            target_belief_ids=[],
            reviewed_candidate_ids=[],
            target_hint="",
            memory=None,
            reason="",
            errors=["update_not_object"],
        )
    operation = _string_field(raw.get("operation"), max_length=64)
    target_belief_ids = _id_list(raw.get("target_belief_ids"))
    reviewed_candidate_ids = _id_list(raw.get("reviewed_candidate_ids"))
    target_hint = _string_field(raw.get("target_hint"), max_length=300)
    reason = _string_field(raw.get("reason"), max_length=500)
    memory, memory_errors = _parse_memory(raw.get("memory"))
    errors = list(memory_errors)
    if operation not in _ALLOWED_OPERATIONS:
        errors.append("invalid_operation")
    if not reason:
        errors.append("missing_reason")
    if raw.get("target_belief_ids") is not None and not isinstance(
        raw.get("target_belief_ids"), list
    ):
        errors.append("invalid_target_belief_ids")
    if raw.get("reviewed_candidate_ids") is not None and not isinstance(
        raw.get("reviewed_candidate_ids"), list
    ):
        errors.append("invalid_reviewed_candidate_ids")
    if operation != "retract" and memory is None:
        errors.append("missing_memory")
    return _ParsedUpdate(
        index=index,
        operation=operation,
        target_belief_ids=target_belief_ids,
        reviewed_candidate_ids=reviewed_candidate_ids,
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
        )
    if (
        parsed.memory is not None
        and parsed.memory.scope == "counterpart"
        and context.counterpart is None
    ):
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="missing_counterpart_scope",
            memory=parsed.memory,
        )
    if context.memory_state is None:
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="missing_memory_state_service",
            memory=parsed.memory,
            emit_memory_proposed=False,
        )
    if context.belief_projection is None:
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="missing_belief_projection",
            memory=parsed.memory,
            emit_memory_proposed=False,
        )

    reviewed_check = _validate_reviewed_candidates(parsed, context)
    if reviewed_check.decision != "accepted":
        return reviewed_check

    target_check = _validate_targets(parsed, context)
    if target_check.decision != "accepted":
        return target_check

    if parsed.operation == "append_distinct":
        return _plan_append_distinct(
            parsed,
            context,
            reviewed_check.reviewed_candidate_beliefs,
        )
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


def _apply_domain_guidance(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
    plan: _OperationPlan,
) -> _OperationPlan:
    del parsed
    if plan.decision != "accepted" or context.belief_projection is None:
        return plan
    if not memory_propose_requires_confirmation(
        context.belief_projection,
        counterpart=context.counterpart,
    ):
        return plan
    return replace(
        plan,
        decision="pending_confirmation",
        reason="domain_guidance_requires_confirmation",
    )


def _plan_append_distinct(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
    reviewed_candidate_beliefs: list[AtomicBelief],
) -> _OperationPlan:
    exact_duplicates = _exact_duplicate_beliefs(parsed, context)
    if exact_duplicates:
        return _OperationPlan(
            decision="accepted",
            operation="reinforce",
            reason="accepted_duplicate_reinforced",
            target_beliefs=[exact_duplicates[0]],
            memory=parsed.memory,
        )
    if reviewed_candidate_beliefs:
        return _OperationPlan(
            decision="accepted",
            operation="append_distinct",
            reason="accepted_append_distinct",
            reviewed_candidate_beliefs=reviewed_candidate_beliefs,
            memory=parsed.memory,
        )
    candidates = _candidate_outputs(parsed, context)
    if candidates:
        return _OperationPlan(
            decision="needs_target_selection",
            operation="append_distinct",
            reason="related_active_beliefs_require_target_selection",
            candidates=candidates,
            memory=parsed.memory,
        )
    return _OperationPlan(
        decision="accepted",
        operation="append_distinct",
        reason="accepted_append_distinct",
        reviewed_candidate_beliefs=reviewed_candidate_beliefs,
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
    if parsed.operation == "append_distinct" and parsed.target_belief_ids:
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="append_distinct_uses_reviewed_candidate_ids",
            memory=parsed.memory,
        )
    if not parsed.target_belief_ids:
        return _OperationPlan(
            decision="accepted",
            operation=parsed.operation,
            reason="targets_valid",
            memory=parsed.memory,
        )
    if len(set(parsed.target_belief_ids)) != len(parsed.target_belief_ids):
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="duplicate_target_belief_ids",
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
    target_beliefs: list[AtomicBelief] = []
    for target_id in parsed.target_belief_ids:
        belief = projection.get_by_id(target_id)
        if belief is None:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_not_found",
                memory=parsed.memory,
            )
        if isinstance(belief, SummaryBelief):
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="target_not_atomic",
                memory=parsed.memory,
            )
        if belief.lifecycle != BeliefLifecycle.ACTIVE:
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


def _validate_reviewed_candidates(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
) -> _OperationPlan:
    if not parsed.reviewed_candidate_ids:
        return _OperationPlan(
            decision="accepted",
            operation=parsed.operation,
            reason="reviewed_candidates_valid",
            memory=parsed.memory,
        )
    if parsed.operation != "append_distinct":
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="reviewed_candidates_only_for_append_distinct",
            memory=parsed.memory,
        )
    if len(set(parsed.reviewed_candidate_ids)) != len(parsed.reviewed_candidate_ids):
        return _OperationPlan(
            decision="rejected",
            operation=parsed.operation,
            reason="duplicate_reviewed_candidate_ids",
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
    reviewed: list[AtomicBelief] = []
    for candidate_id in parsed.reviewed_candidate_ids:
        belief = projection.get_by_id(candidate_id)
        if belief is None:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="reviewed_candidate_not_found",
                memory=parsed.memory,
            )
        if isinstance(belief, SummaryBelief):
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="reviewed_candidate_not_atomic",
                memory=parsed.memory,
            )
        if belief.lifecycle != BeliefLifecycle.ACTIVE:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="reviewed_candidate_not_active",
                memory=parsed.memory,
            )
        if not _target_scope_matches(belief, parsed.memory, context):
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="reviewed_candidate_scope_mismatch",
                memory=parsed.memory,
            )
        if parsed.memory is not None and _memory_type_for_belief(belief) != parsed.memory.type:
            return _OperationPlan(
                decision="rejected",
                operation=parsed.operation,
                reason="reviewed_candidate_type_mismatch",
                memory=parsed.memory,
            )
        reviewed.append(belief)
    return _OperationPlan(
        decision="accepted",
        operation=parsed.operation,
        reason="reviewed_candidates_valid",
        reviewed_candidate_beliefs=reviewed,
        memory=parsed.memory,
    )


def _target_scope_matches(
    belief: AtomicBelief,
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


def _exact_duplicate_beliefs(
    parsed: _ParsedUpdate,
    context: MemoryProposalContext,
) -> list[AtomicBelief]:
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
    memory_kinds = (
        frozenset({MemoryKind(parsed.memory.type)}) if parsed.memory is not None else None
    )
    candidates = context.belief_projection.recall_candidates(
        BeliefSearchParams(
            query=query,
            counterpart=counterpart,
            include_global=parsed.memory is None or parsed.memory.scope == "global",
            memory_kinds=memory_kinds,
            limit=8,
        )
    )
    outputs: list[_Candidate] = []
    for candidate in candidates:
        belief = candidate.belief
        if not isinstance(belief, AtomicBelief):
            continue
        if belief.lifecycle != BeliefLifecycle.ACTIVE:
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
) -> list[AtomicBelief]:
    if parsed.memory is None or context.belief_projection is None:
        return []
    if parsed.memory.scope == "counterpart" and context.counterpart is not None:
        candidates = context.belief_projection.recall_about(context.counterpart)
    else:
        candidates = context.belief_projection.list_active()
    return [
        belief
        for belief in candidates
        if belief.lifecycle == BeliefLifecycle.ACTIVE
        and _scope_for_belief(belief) == parsed.memory.scope
        and _memory_type_for_belief(belief) == parsed.memory.type
    ]


def _candidate_from_belief(belief: AtomicBelief) -> _Candidate:
    return _Candidate(
        id=str(belief.id),
        content=str(belief.content),
        type=_memory_type_for_belief(belief),
        scope=_scope_for_belief(belief),
        status=belief.lifecycle.value,
    )


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
            "derived_about": [item.to_record() for item in _derived_about(scope, context)],
            "source_refs": [item.to_record() for item in source_refs],
            "audit_refs": [item.to_record() for item in audit_refs],
            "gate": {"decision": plan.decision, "reason": plan.reason},
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
        "reviewed_candidate_ids": plan.reviewed_candidate_ids,
        "reason": parsed.reason or plan.reason,
        "evidence": parsed.evidence or parsed.reason,
    }


def _state_audit(
    kind: str,
    proposal_id: str,
    parsed: _ParsedUpdate,
    plan: _OperationPlan,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "payload": {
            "source": MEMORY_PROPOSE_TOOL_NAME,
            "proposal_id": proposal_id,
            "operation": plan.operation,
            "decision": plan.decision,
            "target_belief_ids": plan.target_belief_ids,
            "reviewed_candidate_ids": plan.reviewed_candidate_ids,
            "reason": parsed.reason or plan.reason,
        },
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


def _belief_object(
    *,
    memory: _MemoryRecord,
    target_hint: str,
    about: list[Reference],
) -> str:
    del about
    return (target_hint or memory.evidence or memory.content).strip()[:240]


def _memory_type_for_belief(belief: AtomicBelief) -> str:
    return belief.memory_kind.value


def _scope_for_belief(belief: AtomicBelief) -> str:
    return belief.scope.value


def _id_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    ids: list[str] = []
    for item in value[:5]:
        raw_id = _non_empty_str(item)
        if raw_id:
            ids.append(raw_id)
    return ids


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
        return "review_candidates"
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


def _result_metadata(*, cognitive_event_ids: list[str]) -> dict[str, Any]:
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
