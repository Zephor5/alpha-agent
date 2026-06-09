"""Structured-output contract validation for background cognition LLM calls."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.cognition.authority import (
    AuthorityOverclaimError,
    CognitionSourceKind,
    require_authority_within_ceiling,
)
from alpha_agent.cognition.models import (
    Authority,
    BeliefScope,
    DerivationStage,
    MemoryKind,
    Reference,
    SummaryKind,
    ValidityWindow,
)
from alpha_agent.cognition.processing_ledger import BackgroundSourceRef, BackgroundStage

_SUPPORTED_OPERATIONS = frozenset(
    {
        "create_atomic_belief",
        "create_summary_belief",
        "update_belief",
        "profile_summary_candidate",
        "create",
        "strengthen",
        "supersede",
        "retract",
        "archive",
        "pending-confirmation",
        "pending_confirmation",
    }
)
_EXTRACTION_OPERATION = "create_atomic_belief"
_EXTRACTION_PAYLOAD_KEYS = frozenset({"atomic_belief_draft"})
_SEMANTIC_OPERATIONS = frozenset(
    {"create", "strengthen", "supersede", "retract", "archive", "pending-confirmation"}
)
_SEMANTIC_OPERATION_ALIASES = {"pending_confirmation": "pending-confirmation"}
_CONSOLIDATION_STAGES = frozenset(
    {BackgroundStage.CONSOLIDATION, BackgroundStage.CONFLICT_REVIEW}
)
_FORBIDDEN_PROVENANCE_KEYS = frozenset(
    {
        "checkpoint_id",
        "evidence_ref",
        "evidence_refs",
        "extraction_run_id",
        "idempotency",
        "idempotency_key",
        "audit_id",
        "provenance",
        "provenance_ref",
        "provenance_refs",
        "source_belief_id",
        "source_belief_ids",
        "source_id",
        "source_ids",
        "source_message_id",
        "source_message_ids",
        "source_ref",
        "source_refs",
        "source_trace_id",
        "source_trace_ids",
        "source_window_id",
        "sources",
        "summary_id",
    }
)
_NORMALIZED_FORBIDDEN_PROVENANCE_KEYS = frozenset(
    re.sub(r"[^a-z0-9]+", "", key.casefold()) for key in _FORBIDDEN_PROVENANCE_KEYS
)
_NORMALIZED_GENERATED_DRAFT_ID_KEYS = frozenset({"id", "beliefid"})
_NUMERIC_STRENGTH_KEY_PARTS = frozenset(
    {
        "confidence",
        "strength",
        "certainty",
        "probability",
        "score",
        "weight",
    }
)
_PROMPT_INJECTION_PATTERNS = (
    "ignore previous",
    "ignore all previous",
    "developer message",
    "system prompt",
    "<system",
    "</system",
    "follow these instructions",
    "forget the instructions",
    "treat audit logs as canonical",
    "audit logs are canonical",
)
_SCOPE_REFERENCE_KINDS: dict[BeliefScope, frozenset[str]] = {
    BeliefScope.COUNTERPART: frozenset({"counterpart"}),
    BeliefScope.SELF: frozenset({"subject", "self"}),
    BeliefScope.PROJECT: frozenset({"project"}),
    BeliefScope.SESSION: frozenset({"session"}),
}
_SOURCE_WINDOW_STOPWORDS = frozenset(
    {
        "about",
        "agent",
        "alpha",
        "that",
        "this",
        "uses",
        "user",
        "with",
        "project",
    }
)


def extraction_output_json_schema() -> dict[str, Any]:
    """Return the LLM-facing JSON schema for extraction-stage outputs."""

    return _background_output_schema(
        operation=_EXTRACTION_OPERATION,
        payload_schema=_payload_schema(
            {"atomic_belief_draft": _atomic_belief_draft_schema()}
        ),
    )


def consolidation_output_json_schema(
    *,
    allowed_target_belief_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the LLM-facing JSON schema for consolidation-stage outputs."""

    target_ids = tuple(sorted({item for item in allowed_target_belief_ids if item.strip()}))
    atomic_payload = _payload_schema({"atomic_belief_draft": _atomic_belief_draft_schema()})
    return {
        "oneOf": [
            _background_output_schema(operation="create", payload_schema=atomic_payload),
            _background_output_schema(
                operation="strengthen",
                payload_schema=_belief_update_payload_schema(
                    operation="strengthen",
                    allowed_target_belief_ids=target_ids,
                ),
            ),
            _background_output_schema(
                operation="supersede",
                payload_schema=_supersede_payload_schema(allowed_target_belief_ids=target_ids),
            ),
            _background_output_schema(
                operation="retract",
                payload_schema=_belief_update_payload_schema(
                    operation="retract",
                    allowed_target_belief_ids=target_ids,
                ),
            ),
            _background_output_schema(
                operation="archive",
                payload_schema=_belief_update_payload_schema(
                    operation="archive",
                    allowed_target_belief_ids=target_ids,
                ),
            ),
            _background_output_schema(
                operation="pending-confirmation",
                payload_schema=atomic_payload,
                requires_confirmation=True,
            ),
        ]
    }


def summary_output_json_schema(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about_refs: Iterable[tuple[str, str]],
    target_domain: str | None = None,
) -> dict[str, Any]:
    """Return the LLM-facing JSON schema for one selected summary target."""

    required = ["summary_kind", "scope", "about", "content"]
    structure_schema = _summary_structure_schema(target_domain)
    if target_domain is not None:
        required.append("structure")
    summary_draft = _summary_belief_draft_schema(
        summary_kind=summary_kind,
        scope=scope,
        about_refs=about_refs,
        structure_schema=structure_schema,
        required=required,
    )
    return _background_output_schema(
        operation="create_summary_belief",
        payload_schema=_payload_schema({"summary_belief_draft": summary_draft}),
    )


def _background_output_schema(
    *,
    operation: str,
    payload_schema: dict[str, Any],
    requires_confirmation: bool | None = None,
) -> dict[str, Any]:
    requires_confirmation_schema: dict[str, Any] = (
        {"type": "boolean"}
        if requires_confirmation is None
        else {"const": requires_confirmation}
    )
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "operation",
            "authority",
            "rationale",
            "requires_confirmation",
            "payload",
        ],
        "properties": {
            "operation": {"const": operation},
            "authority": {"const": Authority.BACKGROUND_SYNTHESIZED.value},
            "rationale": {"type": "string", "minLength": 1},
            "requires_confirmation": requires_confirmation_schema,
            "source_span_note": {"type": ["string", "null"]},
            "payload": payload_schema,
        },
    }


def _payload_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


def _atomic_belief_draft_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_kind", "scope", "about", "content"],
        "properties": {
            "memory_kind": {"enum": [item.value for item in MemoryKind]},
            "scope": {"enum": [item.value for item in BeliefScope]},
            "about": _reference_array_schema(),
            "object": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
            "structure": {"type": "object"},
            "validity": {"type": "object"},
            "update_policy": {"type": "object"},
            "project_descriptor": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "object"},
                ],
            },
        },
    }


def _summary_belief_draft_schema(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about_refs: Iterable[tuple[str, str]],
    structure_schema: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": {
            "summary_kind": {"const": summary_kind.value},
            "scope": {"const": scope.value},
            "about": _reference_array_schema(const_refs=about_refs),
            "object": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 1},
            "structure": structure_schema,
            "validity": {"type": "object"},
            "update_policy": {"type": "object"},
            "project_descriptor": {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "object"},
                ],
            },
        },
    }


def _belief_update_payload_schema(
    *,
    operation: str,
    allowed_target_belief_ids: Iterable[str],
) -> dict[str, Any]:
    return _payload_schema(
        {
            "belief_update": _belief_update_schema(
                operation=operation,
                allowed_target_belief_ids=allowed_target_belief_ids,
            )
        }
    )


def _supersede_payload_schema(
    *,
    allowed_target_belief_ids: Iterable[str],
) -> dict[str, Any]:
    return _payload_schema(
        {
            "belief_update": _belief_update_schema(
                operation="supersede",
                allowed_target_belief_ids=allowed_target_belief_ids,
            ),
            "atomic_belief_draft": _atomic_belief_draft_schema(),
        }
    )


def _belief_update_schema(
    *,
    operation: str,
    allowed_target_belief_ids: Iterable[str],
) -> dict[str, Any]:
    target_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    target_ids = tuple(allowed_target_belief_ids)
    if target_ids:
        target_id_schema["enum"] = list(target_ids)
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["target_belief_id", "rationale"],
        "properties": {
            "target_belief_id": target_id_schema,
            "rationale": {"type": "string", "minLength": 1},
            "update_kind": {"const": operation},
        },
    }


def _summary_structure_schema(target_domain: str | None) -> dict[str, Any]:
    if target_domain is None:
        return {"type": "object"}
    return {
        "type": "object",
        "required": ["target_domain"],
        "properties": {"target_domain": {"const": target_domain}},
    }


def _reference_array_schema(
    *,
    const_refs: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "id"],
            "properties": {
                "kind": {"type": "string", "minLength": 1},
                "id": {"type": "string", "minLength": 1},
            },
        },
    }
    if const_refs is not None:
        schema["const"] = [
            {"kind": kind, "id": ref_id} for kind, ref_id in sorted(const_refs)
        ]
    return schema


class BackgroundLLMValidationError(ValueError):
    """Raised when background LLM output cannot be accepted."""


@dataclass(frozen=True)
class SourceWindowValidationContext:
    """Program-selected source window used to validate one LLM output."""

    window_id: str
    source_refs: tuple[BackgroundSourceRef, ...]
    stage: BackgroundStage = BackgroundStage.EXTRACTION
    target_unit: str | None = None
    session_id: str | None = None
    ordinal_start: int | None = None
    ordinal_end: int | None = None
    source_text: str | None = None


@dataclass(frozen=True)
class BackgroundLLMValidationContext:
    """Program-owned validation inputs that the LLM must not invent."""

    source_kind: CognitionSourceKind
    source_window: SourceWindowValidationContext
    allowed_target_belief_ids: frozenset[str] = frozenset()
    input_belief_ids: frozenset[str] = frozenset()
    allowed_about_refs: frozenset[tuple[str, str]] | None = None
    allowed_summary_kinds: frozenset[SummaryKind] | None = None
    required_summary_scope: BeliefScope | None = None
    required_summary_about_refs: frozenset[tuple[str, str]] | None = None
    required_summary_target_domain: str | None = None
    derivation_stage: DerivationStage = DerivationStage.BACKGROUND_EXTRACTED


@dataclass(frozen=True)
class ValidatedAtomicBeliefDraft:
    """Id-less atomic belief draft accepted from a background LLM output."""

    memory_kind: MemoryKind
    scope: BeliefScope
    about: tuple[Reference, ...]
    content: str
    object: str
    structure: dict[str, Any] | None = None
    validity: ValidityWindow | None = None
    update_policy: dict[str, Any] = field(default_factory=dict)
    project_descriptor: str | Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedSummaryBeliefDraft:
    """Id-less summary belief draft accepted from a background LLM output."""

    summary_kind: SummaryKind
    scope: BeliefScope
    about: tuple[Reference, ...]
    content: str
    object: str
    structure: dict[str, Any] | None = None
    validity: ValidityWindow | None = None
    update_policy: dict[str, Any] = field(default_factory=dict)
    project_descriptor: str | Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedBeliefUpdate:
    """Update operation against an input belief id."""

    update_kind: str
    target_belief_id: str
    rationale: str


ValidatedPayload = (
    ValidatedAtomicBeliefDraft | ValidatedSummaryBeliefDraft | ValidatedBeliefUpdate
)


@dataclass(frozen=True)
class ValidatedBackgroundLLMOutput:
    """Validated common envelope plus one stage-specific payload."""

    operation: str
    authority: Authority
    rationale: str
    requires_confirmation: bool
    source_span_note: str | None
    payloads: tuple[ValidatedPayload, ...]


def validate_background_llm_json(
    raw_output: str,
    context: BackgroundLLMValidationContext,
) -> ValidatedBackgroundLLMOutput:
    """Parse and validate a fixture or provider JSON string."""

    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise BackgroundLLMValidationError(f"malformed background LLM JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise BackgroundLLMValidationError("malformed background LLM output must be an object")
    return validate_background_llm_output(decoded, context)


def validate_background_llm_output(
    output: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> ValidatedBackgroundLLMOutput:
    """Validate decoded background LLM structured output."""

    _reject_numeric_strength_fields(output)
    _reject_forbidden_provenance_keys(output)
    _reject_prompt_injection(output)

    operation = _canonical_operation(_required_str(output, "operation"))
    if operation not in _SUPPORTED_OPERATIONS:
        raise BackgroundLLMValidationError(f"unsupported operation: {operation}")
    try:
        authority = require_authority_within_ceiling(
            _required_str(output, "authority"),
            source_kind=context.source_kind,
        )
    except (ValueError, AuthorityOverclaimError) as exc:
        raise BackgroundLLMValidationError(f"authority overclaim: {exc}") from exc

    rationale = _required_str(output, "rationale")
    requires_confirmation = output.get("requires_confirmation")
    if not isinstance(requires_confirmation, bool):
        raise BackgroundLLMValidationError("requires_confirmation must be a boolean")
    source_span_note = output.get("source_span_note")
    if source_span_note is not None and not isinstance(source_span_note, str):
        raise BackgroundLLMValidationError("source_span_note must be a string when provided")

    payload = output.get("payload")
    if not isinstance(payload, Mapping):
        raise BackgroundLLMValidationError("payload must be an object")
    _validate_stage_output_shape(operation=operation, payload=payload, context=context)

    payloads = _validate_payloads(operation=operation, payload=payload, context=context)
    return ValidatedBackgroundLLMOutput(
        operation=operation,
        authority=authority,
        rationale=rationale,
        requires_confirmation=requires_confirmation,
        source_span_note=source_span_note,
        payloads=payloads,
    )


def _validate_payloads(
    *,
    operation: str,
    payload: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> tuple[ValidatedPayload, ...]:
    if operation == "create_atomic_belief":
        return (_validate_atomic_draft(payload.get("atomic_belief_draft"), context),)
    if operation == "create_summary_belief":
        return (_validate_summary_draft(payload.get("summary_belief_draft"), context),)
    if operation == "profile_summary_candidate":
        return (_validate_summary_draft(payload.get("profile_summary_candidate"), context),)
    if operation == "update_belief":
        return (_validate_belief_update(payload.get("belief_update"), context, operation),)
    if operation in {"create", "pending-confirmation"}:
        return (_validate_atomic_draft(payload.get("atomic_belief_draft"), context),)
    if operation == "supersede":
        return (
            _validate_belief_update(payload.get("belief_update"), context, operation),
            _validate_atomic_draft(payload.get("atomic_belief_draft"), context),
        )
    if operation in {"strengthen", "retract", "archive"}:
        return (_validate_belief_update(payload.get("belief_update"), context, operation),)
    raise BackgroundLLMValidationError(f"unsupported operation: {operation}")


def _validate_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> None:
    stage = BackgroundStage(context.source_window.stage)
    if stage in _CONSOLIDATION_STAGES:
        _validate_consolidation_stage_output_shape(operation=operation, payload=payload)
        return
    if stage == BackgroundStage.SUMMARY:
        _validate_summary_stage_output_shape(operation=operation, payload=payload)
        return
    if stage != BackgroundStage.EXTRACTION:
        return
    if operation != _EXTRACTION_OPERATION:
        raise BackgroundLLMValidationError(
            "extraction stage accepts only create_atomic_belief outputs"
        )
    extra_payload_keys = {str(key) for key in payload} - _EXTRACTION_PAYLOAD_KEYS
    if extra_payload_keys:
        extra = ", ".join(sorted(extra_payload_keys))
        raise BackgroundLLMValidationError(
            "extraction payload accepts only atomic_belief_draft; "
            f"unexpected payload keys: {extra}"
        )


def _validate_consolidation_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    if operation not in _SEMANTIC_OPERATIONS:
        raise BackgroundLLMValidationError(
            "consolidation stages accept only semantic operations: create, strengthen, "
            "supersede, retract, archive, pending-confirmation"
        )
    keys = {str(key) for key in payload}
    if operation in {"create", "pending-confirmation"}:
        expected = {"atomic_belief_draft"}
    elif operation in {"strengthen", "retract", "archive"}:
        expected = {"belief_update"}
    else:
        expected = {"belief_update", "atomic_belief_draft"}
    if keys != expected:
        expected_text = ", ".join(sorted(expected))
        raise BackgroundLLMValidationError(
            f"{operation} payload must contain exactly: {expected_text}"
        )


def _validate_summary_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    if operation == "create_summary_belief":
        expected = {"summary_belief_draft"}
    elif operation == "profile_summary_candidate":
        expected = {"profile_summary_candidate"}
    else:
        raise BackgroundLLMValidationError(
            "summary stage accepts only create_summary_belief outputs"
        )
    keys = {str(key) for key in payload}
    if keys != expected:
        expected_text = ", ".join(sorted(expected))
        raise BackgroundLLMValidationError(
            f"{operation} payload must contain exactly: {expected_text}"
        )


def _validate_atomic_draft(
    raw: object,
    context: BackgroundLLMValidationContext,
) -> ValidatedAtomicBeliefDraft:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError("atomic_belief_draft must be an object")
    _reject_generated_draft_ids(raw, label="atomic_belief_draft")
    try:
        memory_kind = MemoryKind(_required_str(raw, "memory_kind"))
    except ValueError as exc:
        raise BackgroundLLMValidationError(
            f"unsupported memory_kind: {raw.get('memory_kind')}"
        ) from exc
    scope, about, project_descriptor = _validate_scope_about(raw, context)
    content = _required_str(raw, "content")
    _validate_source_window_content(content, context)
    return ValidatedAtomicBeliefDraft(
        memory_kind=memory_kind,
        scope=scope,
        about=about,
        content=content,
        object=_optional_str(raw.get("object")) or content,
        structure=_optional_dict(raw.get("structure")),
        validity=_validity(raw.get("validity")),
        update_policy=_optional_dict(raw.get("update_policy")) or {},
        project_descriptor=project_descriptor,
    )


def _validate_summary_draft(
    raw: object,
    context: BackgroundLLMValidationContext,
) -> ValidatedSummaryBeliefDraft:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError("summary_belief_draft must be an object")
    _reject_generated_draft_ids(raw, label="summary_belief_draft")
    try:
        summary_kind = SummaryKind(_required_str(raw, "summary_kind"))
    except ValueError as exc:
        raise BackgroundLLMValidationError(
            f"unsupported summary_kind: {raw.get('summary_kind')}"
        ) from exc
    if (
        context.allowed_summary_kinds is not None
        and summary_kind not in context.allowed_summary_kinds
    ):
        allowed = ", ".join(sorted(item.value for item in context.allowed_summary_kinds))
        raise BackgroundLLMValidationError(
            f"summary_kind {summary_kind.value!r} is outside allowed summary target: {allowed}"
        )
    scope, about, project_descriptor = _validate_scope_about(raw, context)
    if context.required_summary_scope is not None and scope != context.required_summary_scope:
        raise BackgroundLLMValidationError(
            "summary scope does not match selected summary target: "
            f"{scope.value} != {context.required_summary_scope.value}"
        )
    if context.required_summary_about_refs is not None:
        actual_about = frozenset((ref.kind, ref.id) for ref in about)
        if actual_about != context.required_summary_about_refs:
            raise BackgroundLLMValidationError(
                "summary about refs do not match selected summary target"
            )
    content = _required_str(raw, "content")
    _validate_source_window_content(content, context)
    structure = _optional_dict(raw.get("structure"))
    if context.required_summary_target_domain is not None:
        target_domain = (structure or {}).get("target_domain")
        if target_domain != context.required_summary_target_domain:
            raise BackgroundLLMValidationError(
                "summary target_domain does not match selected summary target"
            )
    return ValidatedSummaryBeliefDraft(
        summary_kind=summary_kind,
        scope=scope,
        about=about,
        content=content,
        object=_optional_str(raw.get("object")) or content,
        structure=structure,
        validity=_validity(raw.get("validity")),
        update_policy=_optional_dict(raw.get("update_policy")) or {},
        project_descriptor=project_descriptor,
    )


def _validate_belief_update(
    raw: object,
    context: BackgroundLLMValidationContext,
    operation: str,
) -> ValidatedBeliefUpdate:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError("belief_update must be an object")
    if operation == "update_belief":
        update_kind = _required_str(raw, "update_kind")
    else:
        update_kind = _optional_str(raw.get("update_kind")) or operation
        if update_kind != operation:
            raise BackgroundLLMValidationError(
                f"belief_update update_kind {update_kind!r} does not match operation {operation!r}"
            )
    target_belief_id = _required_str(raw, "target_belief_id")
    allowed_ids = context.allowed_target_belief_ids
    if target_belief_id not in allowed_ids:
        raise BackgroundLLMValidationError(
            f"target belief id {target_belief_id!r} was not included in LLM input"
        )
    return ValidatedBeliefUpdate(
        update_kind=update_kind,
        target_belief_id=target_belief_id,
        rationale=_required_str(raw, "rationale"),
    )


def _canonical_operation(operation: str) -> str:
    return _SEMANTIC_OPERATION_ALIASES.get(operation, operation)


def _validate_scope_about(
    raw: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> tuple[BeliefScope, tuple[Reference, ...], str | Mapping[str, Any] | None]:
    try:
        scope = BeliefScope(_required_str(raw, "scope"))
    except ValueError as exc:
        raise BackgroundLLMValidationError(f"unsupported scope: {raw.get('scope')}") from exc
    if "about" not in raw:
        raise BackgroundLLMValidationError("about is required")
    about_raw = raw.get("about")
    if not isinstance(about_raw, list):
        raise BackgroundLLMValidationError("about must be a list")
    about = tuple(_reference_from_record(item, label="about") for item in about_raw)
    project_descriptor = raw.get("project_descriptor")
    if project_descriptor is not None and not isinstance(project_descriptor, str | Mapping):
        raise BackgroundLLMValidationError("project_descriptor must be a string or object")
    if scope == BeliefScope.PROJECT:
        if about:
            raise BackgroundLLMValidationError(
                "project-scoped output must not include LLM-supplied about references; "
                "use project_descriptor"
            )
        if project_descriptor is None:
            raise BackgroundLLMValidationError("project scope requires project_descriptor")
        if not _resolvable_project_descriptor(project_descriptor):
            raise BackgroundLLMValidationError("project_descriptor is not resolvable")
        return scope, about, project_descriptor

    expected_kinds = _SCOPE_REFERENCE_KINDS.get(scope)
    if expected_kinds is not None and not any(ref.kind in expected_kinds for ref in about):
        expected = ", ".join(sorted(expected_kinds))
        raise BackgroundLLMValidationError(
            f"{scope.value}-scoped output requires about reference kind: {expected}"
        )
    _validate_allowed_about_refs(about, context)
    return scope, about, None


def _reference_from_record(raw: object, *, label: str) -> Reference:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError(f"{label} entries must be objects")
    kind = raw.get("kind")
    ref_id = raw.get("id")
    if not isinstance(kind, str) or not kind.strip():
        raise BackgroundLLMValidationError(f"{label} reference kind is required")
    if not isinstance(ref_id, str) or not ref_id.strip():
        raise BackgroundLLMValidationError(f"{label} reference id is required")
    return Reference(kind, ref_id)


def _reject_generated_draft_ids(raw: Mapping[str, Any], *, label: str) -> None:
    for key in raw:
        if _normalized_generated_key(key) in _NORMALIZED_GENERATED_DRAFT_ID_KEYS:
            raise BackgroundLLMValidationError(f"{label} must not include generated {key}")


def _reject_numeric_strength_fields(value: object, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).casefold()
            if any(part in key_text for part in _NUMERIC_STRENGTH_KEY_PARTS):
                raise BackgroundLLMValidationError(
                    f"confidence or numeric strength field is not allowed: {path}{key}"
                )
            _reject_numeric_strength_fields(nested, path=f"{path}{key}.")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_numeric_strength_fields(item, path=f"{path}{index}.")


def _reject_forbidden_provenance_keys(value: object, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _normalized_generated_key(key) in _NORMALIZED_FORBIDDEN_PROVENANCE_KEYS:
                raise BackgroundLLMValidationError(
                    "LLM output must not provide source refs, generated ids, "
                    f"or idempotency keys: {path}{key}"
                )
            _reject_forbidden_provenance_keys(nested, path=f"{path}{key}.")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_provenance_keys(item, path=f"{path}{index}.")


def _validate_allowed_about_refs(
    about: tuple[Reference, ...],
    context: BackgroundLLMValidationContext,
) -> None:
    if context.allowed_about_refs is None:
        return
    for ref in about:
        if (ref.kind, ref.id) not in context.allowed_about_refs:
            raise BackgroundLLMValidationError(
                f"about reference {ref.kind}:{ref.id} was not included in LLM input"
            )


def _resolvable_project_descriptor(descriptor: str | Mapping[str, Any]) -> bool:
    if isinstance(descriptor, str):
        return bool(descriptor.strip())
    if not descriptor:
        return False
    for value in descriptor.values():
        if isinstance(value, str):
            if value.strip():
                return True
        elif value is not None:
            return True
    return False


def _normalized_generated_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).casefold())


def _reject_prompt_injection(value: object) -> None:
    if isinstance(value, Mapping):
        for nested in value.values():
            _reject_prompt_injection(nested)
        return
    if isinstance(value, list):
        for item in value:
            _reject_prompt_injection(item)
        return
    if not isinstance(value, str):
        return
    normalized = value.casefold()
    if any(pattern in normalized for pattern in _PROMPT_INJECTION_PATTERNS):
        raise BackgroundLLMValidationError("prompt-injection content is not allowed")


def _validate_source_window_content(
    content: str,
    context: BackgroundLLMValidationContext,
) -> None:
    source_text = context.source_window.source_text
    if source_text is None:
        return
    content_tokens = _content_tokens(content)
    if not content_tokens:
        return
    source_tokens = _content_tokens(source_text)
    missing = content_tokens - source_tokens
    if missing:
        raise BackgroundLLMValidationError(
            "output appears outside selected source window: " + ", ".join(sorted(missing))
        )


def _content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_-]*", value.casefold())
        if len(token) >= 4 and token not in _SOURCE_WINDOW_STOPWORDS
    }


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BackgroundLLMValidationError(f"{key} is required")
    return value.strip()


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BackgroundLLMValidationError("optional string field must be a string")
    stripped = value.strip()
    return stripped or None


def _optional_dict(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise BackgroundLLMValidationError("optional object field must be an object")
    return dict(value)


def _validity(value: object) -> ValidityWindow | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise BackgroundLLMValidationError("validity must be an object")
    return ValidityWindow.from_record(dict(value))
