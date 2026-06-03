"""Cognition write-proposal tool."""

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
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.tools.base import ToolExecutionContext, ToolResult
from alpha_agent.utils.ids import new_id

MEMORY_PROPOSE_TOOL_NAME = "memory_propose"
MEMORY_PROPOSE_CONTEXT_KEY = "memory_propose"

_ALLOWED_KINDS = frozenset({"preference", "constraint", "correction", "procedure"})
_AUTO_ACCEPT_KINDS = frozenset({"preference", "constraint", "procedure"})
_ALLOWED_SCOPES = frozenset({"counterpart", "global"})

GateDecision = Literal["accepted", "pending_confirmation", "rejected"]
MemoryStatus = Literal["accepted", "pending_confirmation", "rejected", "mixed"]
UserAction = Literal["none", "ask_confirmation", "explain_rejection"]


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
class _ParsedProposal:
    record: dict[str, str]
    errors: list[str]


@dataclass(frozen=True)
class _GateResult:
    decision: GateDecision
    reason: str
    conflict_belief_ids: list[str] = field(default_factory=list)
    duplicate_belief_id: str | None = None
    replace_belief_id: str | None = None
    candidate_change_kind: str = "create"


@dataclass(frozen=True)
class _ProposalResult:
    proposal_id: str
    decision: GateDecision
    reason: str

    def to_output(self) -> dict[str, str]:
        return {
            "proposal_id": self.proposal_id,
            "decision": self.decision,
            "reason": self.reason,
        }


class MemoryProposeTool:
    """Accept model-proposed long-term cognition write candidates."""

    name = MEMORY_PROPOSE_TOOL_NAME
    description = (
        "Propose explicit long-term user cognition: preferences, stable constraints, "
        "reusable procedures, or direct corrections. Do not use for ordinary facts, "
        "transient context, guesses, or tool summaries. Returns a structured memory "
        "proposal result with status accepted, pending_confirmation, rejected, or mixed "
        "and tells the model whether user confirmation or rejection explanation is needed."
    )
    strict = True
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["preference", "constraint", "correction", "procedure"],
                        },
                        "content": {
                            "type": "string",
                            "maxLength": 500,
                        },
                        "evidence": {
                            "type": "string",
                            "maxLength": 300,
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["counterpart", "global"],
                        },
                    },
                    "required": ["kind", "content", "evidence", "scope"],
                },
            }
        },
        "required": ["proposals"],
    }

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        raw_proposals = arguments.get("proposals")
        proposal_items = raw_proposals if isinstance(raw_proposals, list) else []
        memory_context = _memory_proposal_context(context.extensions)

        if memory_context is None:
            return ToolResult(
                name=self.name,
                output=_memory_output(
                    status="rejected",
                    user_action="explain_rejection",
                    message_hint="Memory proposal rejected: missing_runtime_turn_context.",
                    proposal_results=[],
                ),
                metadata=_tool_metadata(cognitive_event_ids=[]),
            )

        cognitive_event_ids: list[str] = []
        proposal_results: list[_ProposalResult] = []
        if not isinstance(raw_proposals, list) or not proposal_items:
            return ToolResult(
                name=self.name,
                output=_memory_output(
                    status="rejected",
                    user_action="explain_rejection",
                    message_hint="Memory proposal rejected: missing_proposals.",
                    proposal_results=[],
                ),
                metadata=_tool_metadata(cognitive_event_ids=[]),
            )

        too_many_proposals = len(proposal_items) > 5
        for raw in proposal_items:
            proposal_id = new_id("proposal")
            parsed = _parse_proposal(raw)
            if too_many_proposals:
                parsed = _ParsedProposal(
                    record=parsed.record,
                    errors=[*parsed.errors, "too_many_proposals"],
                )
            gate = _gate(parsed, memory_context)
            proposed = _emit_memory_proposed(
                context=memory_context,
                proposal_id=proposal_id,
                proposal=parsed.record,
                gate_decision=gate.decision,
                gate_reason=gate.reason,
                tool_call_id=context.tool_call_id,
            )
            cognitive_event_ids.append(str(proposed.id))
            proposal_results.append(
                _ProposalResult(
                    proposal_id=proposal_id,
                    decision=gate.decision,
                    reason=gate.reason,
                )
            )
            if gate.decision == "pending_confirmation":
                pending = _emit_pending_confirmation(
                    context=memory_context,
                    proposal_id=proposal_id,
                    proposal=parsed.record,
                    gate=gate,
                    proposed_event=proposed,
                )
                cognitive_event_ids.append(str(pending.id))
            elif gate.decision == "accepted" and gate.duplicate_belief_id is None:
                belief = build_belief_from_memory_proposal(
                    proposal=parsed.record,
                    proposal_id=proposal_id,
                    proposed_event_id=str(proposed.id),
                    gate_reason=gate.reason,
                    context=memory_context,
                )
                if gate.replace_belief_id is not None:
                    superseded = _emit_belief_superseded(
                        context=memory_context,
                        proposal_id=proposal_id,
                        old_belief_id=gate.replace_belief_id,
                        belief=belief,
                        gate_reason=gate.reason,
                        proposed_event=proposed,
                    )
                    cognitive_event_ids.append(str(superseded.id))
                else:
                    formed = _emit_belief_formed(
                        context=memory_context,
                        proposal_id=proposal_id,
                        belief=belief,
                        gate_reason=gate.reason,
                        proposed_event=proposed,
                    )
                    cognitive_event_ids.append(str(formed.id))
        status = _aggregate_status(proposal_results)
        user_action = _aggregate_user_action(proposal_results)
        return ToolResult(
            name=self.name,
            output=_memory_output(
                status=status,
                user_action=user_action,
                message_hint=_message_hint(user_action, proposal_results),
                proposal_results=proposal_results,
            ),
            metadata=_tool_metadata(cognitive_event_ids=cognitive_event_ids),
        )


def build_belief_from_memory_proposal(
    *,
    proposal: Mapping[str, str],
    proposal_id: str,
    proposed_event_id: str,
    gate_reason: str,
    context: MemoryProposalContext,
) -> Belief:
    """Map an accepted foreground memory proposal onto the existing Belief model."""

    kind = proposal["kind"]
    scope = proposal["scope"]
    about = _derived_about(scope, context)
    return Belief(
        id=BeliefId(new_id("belief")),
        subject=Reference("subject", str(context.subject.id)),
        about=about,
        object=_belief_object(kind=kind, scope=scope, about=about),
        content=NLStatement(proposal["content"]),
        cognitive_type=_belief_type(kind),
        structure=None,
        sources=[Reference("session_message", context.user_message_id)],
        confidence=0.72,
        applicability=Applicability(
            deterministic_json(
                {
                    "source": MEMORY_PROPOSE_TOOL_NAME,
                    "scope": scope,
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
                    "conflict": "pending_review",
                    "updates": "do_not_auto_overwrite",
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
                    "gate_reason": gate_reason,
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


def _parse_proposal(raw: object) -> _ParsedProposal:
    if not isinstance(raw, Mapping):
        return _ParsedProposal(
            record={"kind": "", "content": "", "evidence": "", "scope": ""},
            errors=["proposal_not_object"],
        )
    kind = _string_field(raw.get("kind"), max_length=64)
    content = _string_field(raw.get("content"), max_length=500)
    evidence = _string_field(raw.get("evidence"), max_length=300)
    scope = _string_field(raw.get("scope"), max_length=64)
    errors: list[str] = []
    if kind not in _ALLOWED_KINDS:
        errors.append("invalid_kind")
    if not content:
        errors.append("missing_content")
    elif len(str(raw.get("content") or "")) > 500:
        errors.append("content_too_long")
    if not evidence:
        errors.append("missing_evidence")
    elif len(str(raw.get("evidence") or "")) > 300:
        errors.append("evidence_too_long")
    if scope not in _ALLOWED_SCOPES:
        errors.append("invalid_scope")
    return _ParsedProposal(
        record={"kind": kind, "content": content, "evidence": evidence, "scope": scope},
        errors=errors,
    )


def _gate(
    parsed: _ParsedProposal,
    context: MemoryProposalContext,
) -> _GateResult:
    if parsed.errors:
        return _GateResult("rejected", "invalid_schema:" + ",".join(parsed.errors))
    kind = parsed.record["kind"]
    scope = parsed.record["scope"]
    if scope == "counterpart" and context.counterpart is None:
        return _GateResult("pending_confirmation", "missing_counterpart_scope")
    targeted_beliefs = _targeted_active_beliefs(parsed.record, context)
    same_content = [
        belief
        for belief in targeted_beliefs
        if _normalized_content(belief.content) == _normalized_content(parsed.record["content"])
    ]
    if same_content:
        return _GateResult(
            "accepted",
            "accepted_duplicate_active_belief",
            duplicate_belief_id=str(same_content[0].id),
        )
    conflict_belief_ids = [
        str(belief.id)
        for belief in targeted_beliefs
        if _normalized_content(belief.content) != _normalized_content(parsed.record["content"])
    ]
    if kind == "correction":
        return _GateResult(
            "pending_confirmation",
            "correction_requires_review",
            conflict_belief_ids=conflict_belief_ids,
            candidate_change_kind="correct",
        )
    if len(conflict_belief_ids) == 1:
        return _GateResult(
            "accepted",
            "accepted_single_structured_replacement",
            conflict_belief_ids=conflict_belief_ids,
            replace_belief_id=conflict_belief_ids[0],
            candidate_change_kind="replace",
        )
    if conflict_belief_ids:
        return _GateResult(
            "pending_confirmation",
            "ambiguous_conflict",
            conflict_belief_ids=conflict_belief_ids,
            candidate_change_kind="replace",
        )
    if kind not in _AUTO_ACCEPT_KINDS:
        return _GateResult("rejected", "unsupported_auto_accept_kind")
    return _GateResult("accepted", f"accepted_foreground_{kind}")


def _emit_belief_formed(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    belief: Belief,
    gate_reason: str,
    proposed_event: CognitiveEvent,
) -> CognitiveEvent:
    formed = context.emitter.emit(
        CognitiveEventKind.BELIEF_FORMED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(belief.id))],
        rationale="Promoted accepted foreground memory proposal.",
        causal_parents=[proposed_event.id],
        payload={
            "turn_id": context.turn_id,
            "session_id": context.session_id,
            "proposal_id": proposal_id,
            "origin": MEMORY_PROPOSE_TOOL_NAME,
            "gate_reason": gate_reason,
            "belief": belief.to_record(),
        },
    )
    context.apply_cognitive_event(formed)
    return formed


def _emit_belief_superseded(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    old_belief_id: str,
    belief: Belief,
    gate_reason: str,
    proposed_event: CognitiveEvent,
) -> CognitiveEvent:
    superseded = context.emitter.emit(
        CognitiveEventKind.BELIEF_SUPERSEDED,
        situation=context.situation,
        inputs=[
            Reference("memory_proposal", proposal_id),
            Reference("belief", old_belief_id),
            Reference("session_message", context.user_message_id),
        ],
        outputs=[Reference("belief", str(belief.id))],
        rationale="Replaced one active belief through an accepted foreground memory proposal.",
        causal_parents=[proposed_event.id],
        payload={
            "turn_id": context.turn_id,
            "session_id": context.session_id,
            "proposal_id": proposal_id,
            "origin": MEMORY_PROPOSE_TOOL_NAME,
            "gate_reason": gate_reason,
            "old_belief_id": old_belief_id,
            "new_belief_id": str(belief.id),
            "reason": gate_reason,
            "belief": belief.to_record(),
        },
    )
    context.apply_cognitive_event(superseded)
    return superseded


def _emit_memory_proposed(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    proposal: dict[str, str],
    gate_decision: GateDecision,
    gate_reason: str,
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
    return context.emitter.emit(
        CognitiveEventKind.MEMORY_PROPOSED,
        situation=context.situation,
        inputs=[Reference("session_message", context.user_message_id)],
        outputs=[Reference("memory_proposal", proposal_id)],
        rationale="Recorded foreground memory write proposal.",
        causal_parents=_event_ids([context.turn_received_event_id]),
        payload={
            "turn_id": context.turn_id,
            "session_id": context.session_id,
            "tool_call_id": tool_call_id or "",
            "proposal_id": proposal_id,
            "proposal": dict(proposal),
            "derived_about": [
                item.to_record() for item in _derived_about(proposal["scope"], context)
            ],
            "source_refs": [item.to_record() for item in source_refs],
            "audit_refs": [item.to_record() for item in audit_refs],
            "gate": {"decision": gate_decision, "reason": gate_reason},
        },
    )


def _emit_pending_confirmation(
    *,
    context: MemoryProposalContext,
    proposal_id: str,
    proposal: dict[str, str],
    gate: _GateResult,
    proposed_event: CognitiveEvent,
) -> CognitiveEvent:
    return context.emitter.emit(
        CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION,
        situation=context.situation,
        inputs=[Reference("memory_proposal", proposal_id)],
        outputs=[],
        rationale="Memory proposal requires user confirmation before mutation.",
        causal_parents=[proposed_event.id],
        payload={
            "turn_id": context.turn_id,
            "session_id": context.session_id,
            "proposal_id": proposal_id,
            "reason": gate.reason,
            "required_user_action": "confirm_memory_change",
            "candidate_change": {
                "kind": gate.candidate_change_kind,
                "content": proposal["content"],
            },
            "conflict_belief_ids": list(gate.conflict_belief_ids),
        },
    )


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


def _belief_object(*, kind: str, scope: str, about: list[Reference]) -> str:
    if scope == "counterpart" and about:
        return f"{kind}:{about[0].id}"
    return f"{kind}:{scope}"


def _belief_type(kind: str) -> CognitiveType:
    if kind == "preference":
        return CognitiveType.PREFERENCE
    return CognitiveType.PROCEDURAL


def _targeted_active_beliefs(
    proposal: Mapping[str, str],
    context: MemoryProposalContext,
) -> list[Belief]:
    projection = context.belief_projection
    if projection is None:
        return []
    about = _derived_about(proposal["scope"], context)
    if about:
        candidates = projection.recall_about(about[0])
    else:
        candidates = projection.list_active()
    belief_object = _belief_object(
        kind=proposal["kind"],
        scope=proposal["scope"],
        about=about,
    )
    return [
        belief
        for belief in candidates
        if str(belief.status) == "active" and str(belief.object) == belief_object
    ]


def _aggregate_status(proposal_results: list[_ProposalResult]) -> MemoryStatus:
    decisions = {result.decision for result in proposal_results}
    if not decisions:
        return "rejected"
    if len(decisions) == 1:
        return proposal_results[0].decision
    return "mixed"


def _aggregate_user_action(proposal_results: list[_ProposalResult]) -> UserAction:
    actions = [_user_action_for(result) for result in proposal_results]
    if "ask_confirmation" in actions:
        return "ask_confirmation"
    if "explain_rejection" in actions:
        return "explain_rejection"
    return "none"


def _user_action_for(result: _ProposalResult) -> UserAction:
    if result.decision == "pending_confirmation":
        return "ask_confirmation"
    if result.decision == "rejected" and (
        result.reason.startswith("invalid_schema:")
        or result.reason == "unsupported_auto_accept_kind"
    ):
        return "explain_rejection"
    return "none"


def _message_hint(
    user_action: UserAction,
    proposal_results: list[_ProposalResult],
) -> str:
    if user_action == "ask_confirmation":
        return "Ask the user to confirm the pending memory change before applying it."
    if user_action == "explain_rejection":
        reason = next(
            (
                result.reason
                for result in proposal_results
                if _user_action_for(result) == "explain_rejection"
            ),
            "unknown",
        )
        return f"Memory proposal rejected: {reason}."
    return ""


def _memory_output(
    *,
    status: MemoryStatus,
    user_action: UserAction,
    message_hint: str,
    proposal_results: list[_ProposalResult],
) -> dict[str, Any]:
    return {
        "status": status,
        "user_action": user_action,
        "message_hint": message_hint,
        "proposal_results": [result.to_output() for result in proposal_results],
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
