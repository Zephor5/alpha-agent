"""Lightweight payload contracts for cognition events consumed by workers.

The checks here intentionally cover only fields that current write-side
consumers read. They are fail-fast guards, not a general event-schema DSL.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from alpha_agent.cognition.models.enums import CognitiveEventKind


class EventPayloadValidationError(ValueError):
    """Raised when an emitted cognition event is missing consumed payload data."""


_MEMORY_PROPOSE_ORIGIN = "memory_propose"


def validate_event_payload(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    """Validate locally consumed payload fields for key cognition events."""

    validator = _VALIDATORS.get(kind)
    if validator is not None:
        validator(kind, payload)


def _validate_perceived(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "stimulus_kind")
    _require_dict(kind, payload, "source")
    _require_list(kind, payload, "source_refs")
    _require_non_empty_str(kind, payload, "content_digest")
    _require_non_negative_int(kind, payload, "content_length")


def _validate_acted(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "assistant_message_id")
    _require_non_empty_str(kind, payload, "response_text_digest")
    _require_non_negative_int(kind, payload, "response_text_length")
    _require_list(kind, payload, "llm_call_ids")
    _require_list(kind, payload, "llm_trace_ids")
    _require_list(kind, payload, "tool_call_ids")
    _require_list(kind, payload, "tool_names")
    _require_list(kind, payload, "tool_result_trace_ids")
    _require_list(kind, payload, "tool_cognitive_event_ids")


def _validate_pending_confirmation(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "proposal_id")
    _require_non_empty_str(kind, payload, "reason")
    _require_non_empty_str(kind, payload, "required_user_action")
    _require_dict(kind, payload, "candidate_change")
    _require_list(kind, payload, "conflict_belief_ids")
    _require_non_empty_str(kind, payload, "operation")
    _require_list(kind, payload, "target_belief_ids")
    _require_non_empty_str(kind, payload, "evidence")
    _require_non_empty_str(kind, payload, "tool_call_id")


def _validate_memory_proposed(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "proposal_id")
    _require_non_empty_str(kind, payload, "tool_call_id")
    _require_dict(kind, payload, "proposal")
    _require_list(kind, payload, "derived_about")
    _require_list(kind, payload, "source_refs")
    _require_list(kind, payload, "audit_refs")
    _require_dict(kind, payload, "gate")
    _require_non_empty_str(kind, payload, "operation")
    _require_list(kind, payload, "target_belief_ids")
    _require_non_empty_str(kind, payload, "reason")
    _require_non_empty_str(kind, payload, "evidence")


def _validate_turn_sources_recorded(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "user_message_id")
    _require_non_empty_str(kind, payload, "assistant_message_id")
    _require_list(kind, payload, "provider_tool_message_ids")
    _require_list(kind, payload, "provider_tool_trace_ids")
    _require_list(kind, payload, "llm_call_ids")
    _require_list(kind, payload, "llm_trace_ids")
    _require_list(kind, payload, "cognitive_event_ids")
    _require_list(kind, payload, "tool_cognitive_event_ids")


def _validate_reflected(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_negative_int(kind, payload, "reflection_count")
    _require_list(kind, payload, "reflection_ids")
    _require_list(kind, payload, "targets")


def _validate_received_feedback(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "feedback_kind")
    _require_bool(kind, payload, "matched_expected")


def _validate_context_compressed(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if "thread_id" in payload:
        raise EventPayloadValidationError(
            f"{kind.value} payload includes retired foreground field: thread_id"
        )
    _require_non_empty_str(kind, payload, "session_id")
    _require_list(kind, payload, "absorbed_perception_ids")
    if not (
        _non_empty_str(payload.get("produced_summary_id"))
        or _non_empty_str(payload.get("background_summary_id"))
    ):
        _missing(kind, "produced_summary_id or background_summary_id")
    _require_non_empty_str(kind, payload, "summary")
    _require_non_empty_str(kind, payload, "compression_policy")


def _validate_procedure_learned(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_dict(kind, payload, "procedure")


def _validate_belief_formed(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if _requires_memory_change_contract(payload):
        _validate_memory_change_common(kind, payload)
        _require_non_empty_str(kind, payload, "new_belief_id")
        _require_dict(kind, payload, "belief")
        return
    if isinstance(payload.get("belief"), dict):
        return
    if _non_empty_str(payload.get("origin")) or payload.get("auto_formed_novel") is True:
        return
    _missing(kind, "belief")


def _validate_belief_strengthened(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if not _requires_memory_change_contract(payload):
        return
    _validate_memory_change_common(kind, payload)
    _require_non_empty_str(kind, payload, "belief_id")


def _validate_belief_superseded(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if not _requires_memory_change_contract(payload):
        return
    _validate_memory_change_common(kind, payload)
    _require_non_empty_str(kind, payload, "old_belief_id")
    _require_non_empty_str(kind, payload, "new_belief_id")
    _require_dict(kind, payload, "belief")


def _validate_belief_retracted(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if not _requires_memory_change_contract(payload):
        return
    _validate_memory_change_common(kind, payload)
    _require_non_empty_str(kind, payload, "belief_id")


def _requires_memory_change_contract(payload: dict[str, Any]) -> bool:
    return payload.get("origin") == _MEMORY_PROPOSE_ORIGIN or "operation" in payload


def _validate_memory_change_common(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
) -> None:
    _validate_foreground_identity(kind, payload)
    _require_non_empty_str(kind, payload, "operation")
    _require_list(kind, payload, "target_belief_ids")
    _require_non_empty_str(kind, payload, "reason")
    _require_non_empty_str(kind, payload, "evidence")
    _require_non_empty_str(kind, payload, "tool_call_id")


def _require_present(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
    field: str,
) -> None:
    if field not in payload or payload[field] is None:
        _missing(kind, field)


def _require_non_empty_str(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
    field: str,
) -> None:
    if not _non_empty_str(payload.get(field)):
        _missing(kind, field)


def _require_bool(kind: CognitiveEventKind, payload: dict[str, Any], field: str) -> None:
    if not isinstance(payload.get(field), bool):
        _missing(kind, field)


def _require_non_negative_int(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
    field: str,
) -> None:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _missing(kind, field)


def _require_dict(kind: CognitiveEventKind, payload: dict[str, Any], field: str) -> None:
    if not isinstance(payload.get(field), dict):
        _missing(kind, field)


def _require_list(kind: CognitiveEventKind, payload: dict[str, Any], field: str) -> None:
    if not isinstance(payload.get(field), list):
        _missing(kind, field)


def _non_empty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _missing(kind: CognitiveEventKind, field: str) -> None:
    raise EventPayloadValidationError(
        f"{kind.value} payload missing consumed field: {field}"
    )


_FORBIDDEN_FOREGROUND_FIELDS = frozenset(
    {"tick_id", "thread_id", "decision_id", "judgment_ids", "schema_version"}
)


def _validate_foreground_identity(
    kind: CognitiveEventKind,
    payload: dict[str, Any],
) -> None:
    for field in _FORBIDDEN_FOREGROUND_FIELDS:
        if field in payload:
            raise EventPayloadValidationError(
                f"{kind.value} payload includes retired foreground field: {field}"
            )
    _require_non_empty_str(kind, payload, "turn_id")
    _require_non_empty_str(kind, payload, "session_id")


_VALIDATORS: dict[CognitiveEventKind, Callable[[CognitiveEventKind, dict[str, Any]], None]] = {
    CognitiveEventKind.PERCEIVED: _validate_perceived,
    CognitiveEventKind.ACTED: _validate_acted,
    CognitiveEventKind.MEMORY_PROPOSED: _validate_memory_proposed,
    CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION: _validate_pending_confirmation,
    CognitiveEventKind.TURN_SOURCES_RECORDED: _validate_turn_sources_recorded,
    CognitiveEventKind.REFLECTED: _validate_reflected,
    CognitiveEventKind.RECEIVED_FEEDBACK: _validate_received_feedback,
    CognitiveEventKind.CONTEXT_COMPRESSED: _validate_context_compressed,
    CognitiveEventKind.PROCEDURE_LEARNED: _validate_procedure_learned,
    CognitiveEventKind.BELIEF_FORMED: _validate_belief_formed,
    CognitiveEventKind.BELIEF_STRENGTHENED: _validate_belief_strengthened,
    CognitiveEventKind.BELIEF_SUPERSEDED: _validate_belief_superseded,
    CognitiveEventKind.BELIEF_RETRACTED: _validate_belief_retracted,
}
