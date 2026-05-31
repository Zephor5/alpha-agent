"""Cognition write-proposal tool."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.tools.base import ToolExecutionContext, ToolResult
from alpha_agent.utils.ids import new_id

MEMORY_PROPOSE_TOOL_NAME = "memory_propose"
MEMORY_PROPOSE_CONTEXT_KEY = "memory_propose"

_ALLOWED_KINDS = frozenset({"preference", "constraint", "correction", "procedure"})
_AUTO_ACCEPT_KINDS = frozenset({"preference", "constraint", "procedure"})
_ALLOWED_SCOPES = frozenset({"counterpart", "global"})
_SUCCESS_OUTPUT = "success"
_FAILED_OUTPUT = "failed"

GateDecision = Literal["accepted", "pending", "rejected"]


@dataclass(frozen=True)
class MemoryProposalContext:
    """Reactive-only write context injected by the runtime tool loop."""

    tick_id: str
    session_id: str
    user_message_id: str
    decision_event_id: str
    emitter: EventEmitter
    apply_cognitive_event: Callable[[CognitiveEvent], None]
    subject: Subject
    situation: SituationRef
    counterpart: Reference | None
    llm_call_id: str
    llm_trace_ids: list[str]


@dataclass(frozen=True)
class _ParsedProposal:
    record: dict[str, str]
    errors: list[str]


class MemoryProposeTool:
    """Accept model-proposed long-term cognition write candidates."""

    name = MEMORY_PROPOSE_TOOL_NAME
    description = (
        "Propose explicit long-term user cognition: preferences, stable constraints, "
        "reusable procedures, or direct corrections. Do not use for ordinary facts, "
        "transient context, guesses, or tool summaries. Returns \"success\" only if all "
        'proposals are accepted; otherwise "failed".'
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
        reactive_context = _memory_proposal_context(context.extensions)

        if reactive_context is None:
            return ToolResult(
                name=self.name,
                output=_FAILED_OUTPUT,
                metadata=_tool_metadata(cognitive_event_ids=[]),
            )

        cognitive_event_ids: list[str] = []
        all_accepted = bool(proposal_items)
        too_many_proposals = len(proposal_items) > 5
        for raw in proposal_items:
            proposal_id = new_id("proposal")
            parsed = _parse_proposal(raw)
            if too_many_proposals:
                parsed = _ParsedProposal(
                    record=parsed.record,
                    errors=[*parsed.errors, "too_many_proposals"],
                )
            gate_decision, gate_reason = _gate(parsed, reactive_context)
            proposed = _emit_memory_proposed(
                context=reactive_context,
                proposal_id=proposal_id,
                proposal=parsed.record,
                gate_decision=gate_decision,
                gate_reason=gate_reason,
                tool_call_id=context.tool_call_id,
            )
            cognitive_event_ids.append(str(proposed.id))
            if gate_decision != "accepted":
                all_accepted = False
            if gate_decision == "accepted":
                belief = build_belief_from_memory_proposal(
                    proposal=parsed.record,
                    proposal_id=proposal_id,
                    proposed_event_id=str(proposed.id),
                    gate_reason=gate_reason,
                    context=reactive_context,
                )
                formed = reactive_context.emitter.emit(
                    CognitiveEventKind.BELIEF_FORMED,
                    situation=reactive_context.situation,
                    inputs=[
                        Reference("memory_proposal", proposal_id),
                        Reference("session_message", reactive_context.user_message_id),
                    ],
                    outputs=[Reference("belief", str(belief.id))],
                    rationale="Promoted accepted foreground memory proposal.",
                    causal_parents=[proposed.id],
                    payload={
                        "tick_id": reactive_context.tick_id,
                        "proposal_id": proposal_id,
                        "origin": MEMORY_PROPOSE_TOOL_NAME,
                        "gate_reason": gate_reason,
                        "belief": belief.to_record(),
                    },
                )
                reactive_context.apply_cognitive_event(formed)
                cognitive_event_ids.append(str(formed.id))
        return ToolResult(
            name=self.name,
            output=_SUCCESS_OUTPUT if all_accepted else _FAILED_OUTPUT,
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
    tick_id = _non_empty_str(raw.get("tick_id"))
    session_id = _non_empty_str(raw.get("session_id"))
    user_message_id = _non_empty_str(raw.get("user_message_id"))
    decision_event_id = _non_empty_str(raw.get("decision_event_id"))
    if not all([tick_id, session_id, user_message_id, decision_event_id]):
        return None
    counterpart = raw.get("counterpart")
    return MemoryProposalContext(
        tick_id=tick_id,
        session_id=session_id,
        user_message_id=user_message_id,
        decision_event_id=decision_event_id,
        emitter=emitter,
        apply_cognitive_event=apply_cognitive_event,
        subject=subject,
        situation=situation,
        counterpart=counterpart if isinstance(counterpart, Reference) else None,
        llm_call_id=_non_empty_str(raw.get("llm_call_id")),
        llm_trace_ids=_string_list(raw.get("llm_trace_ids")),
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
) -> tuple[GateDecision, str]:
    if parsed.errors:
        return "rejected", "invalid_schema:" + ",".join(parsed.errors)
    kind = parsed.record["kind"]
    scope = parsed.record["scope"]
    if scope == "counterpart" and context.counterpart is None:
        return "pending", "missing_counterpart_scope"
    if kind == "correction":
        return "pending", "correction_requires_review"
    if kind not in _AUTO_ACCEPT_KINDS:
        return "rejected", "unsupported_auto_accept_kind"
    return "accepted", f"accepted_foreground_{kind}"


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
        causal_parents=[EventId(context.decision_event_id)],
        payload={
            "tick_id": context.tick_id,
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


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [str(item) for item in value if item is not None and str(item)]
