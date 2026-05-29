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


def validate_event_payload(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    """Validate locally consumed payload fields for key cognition events."""

    validator = _VALIDATORS.get(kind)
    if validator is not None:
        validator(kind, payload)


def _validate_perceived(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_dict(kind, payload, "thread_id")
    _require_dict(kind, payload, "perception")


def _validate_judged(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    if _non_empty_str(payload.get("claim")):
        return
    raw = payload.get("judgments")
    if isinstance(raw, list) and any(
        isinstance(item, dict) and _non_empty_str(item.get("claim")) for item in raw
    ):
        return
    _missing(kind, "claim or judgments[].claim")


def _validate_decided(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_present(kind, payload, "tick_id")
    _require_non_empty_str(kind, payload, "action")
    _require_non_empty_str(kind, payload, "message")


def _validate_acted(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_present(kind, payload, "tick_id")
    _require_non_empty_str(kind, payload, "decision_id")
    _require_list(kind, payload, "tool_call_ids")
    _require_list(kind, payload, "provider_tool_message_ids")
    _require_list(kind, payload, "provider_tool_trace_ids")


def _validate_feedback(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_present(kind, payload, "tick_id")
    _require_bool(kind, payload, "matched_expected")


def _validate_revised(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_present(kind, payload, "tick_id")
    _require_list(kind, payload, "judgment_ids")
    _require_list(kind, payload, "reflection_ids")
    _require_non_empty_str(kind, payload, "feedback_event_id")


def _validate_pending_confirmation(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_present(kind, payload, "tick_id")
    _require_non_empty_str(kind, payload, "reason")
    _require_list(kind, payload, "contradict_ids")


def _validate_context_compressed(kind: CognitiveEventKind, payload: dict[str, Any]) -> None:
    _require_dict(kind, payload, "thread_id")
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
    if isinstance(payload.get("belief"), dict):
        return
    if _non_empty_str(payload.get("origin")) or payload.get("auto_formed_novel") is True:
        return
    _missing(kind, "belief")


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


_VALIDATORS: dict[CognitiveEventKind, Callable[[CognitiveEventKind, dict[str, Any]], None]] = {
    CognitiveEventKind.PERCEIVED: _validate_perceived,
    CognitiveEventKind.JUDGED: _validate_judged,
    CognitiveEventKind.DECIDED: _validate_decided,
    CognitiveEventKind.ACTED: _validate_acted,
    CognitiveEventKind.RECEIVED_FEEDBACK: _validate_feedback,
    CognitiveEventKind.REVISED: _validate_revised,
    CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION: _validate_pending_confirmation,
    CognitiveEventKind.CONTEXT_COMPRESSED: _validate_context_compressed,
    CognitiveEventKind.PROCEDURE_LEARNED: _validate_procedure_learned,
    CognitiveEventKind.BELIEF_FORMED: _validate_belief_formed,
}
