"""Structured-output contract validation for background cognition LLM calls."""

from __future__ import annotations

import hashlib
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
from alpha_agent.cognition.models.belief import validate_belief_topic
from alpha_agent.cognition.processing_ledger import BackgroundSourceRef, BackgroundStage

_SUPPORTED_OPERATIONS = frozenset(
    {
        "create_atomic_belief",
        "create_summary_belief",
        "update_belief",
        "profile_summary_candidate",
        "consolidate_atomic_beliefs",
        "create",
        "strengthen",
        "supersede",
        "retract",
        "archive",
        "skip",
    }
)
_EXTRACTION_OPERATION = "create_atomic_belief"
_EXTRACTION_PAYLOAD_KEYS = frozenset({"atomic_belief_inputs"})
_CONSOLIDATION_BATCH_OPERATION = "consolidate_atomic_beliefs"
_CONSOLIDATION_BATCH_PAYLOAD_KEYS = frozenset({"decisions"})
_SKIP_OPERATION = "skip"
_SKIP_PAYLOAD_KEYS = frozenset({"reason"})
_SEMANTIC_OPERATIONS = frozenset(
    {"create", "strengthen", "supersede", "retract", "archive", _SKIP_OPERATION}
)
_CONSOLIDATION_DECISION_OPERATIONS = frozenset(
    {"promote", "skip", "create", "strengthen", "supersede", "retract", "archive"}
)
_CONSOLIDATION_DECISION_BASE_KEYS = frozenset(
    {"operation", "source_atomic_belief_ids", "rationale"}
)
_FEEDBACK_ATTRIBUTION_VERDICTS = frozenset(
    {"confirmed", "contradicted", "corrected", "irrelevant"}
)
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
_TOPIC_SCHEMA = {"type": "string", "minLength": 1, "maxLength": 64}
_ATOMIC_DRAFT_KEYS = frozenset(
    {
        "memory_kind",
        "scope",
        "about",
        "topic",
        "content",
        "validity",
        "update_policy",
        "project_descriptor",
    }
)
_SUMMARY_DRAFT_KEYS = frozenset(
    {
        "summary_kind",
        "scope",
        "about",
        "topic",
        "content",
        "structure",
        "validity",
        "update_policy",
        "project_descriptor",
    }
)
_SUMMARY_SCHEDULING_UPDATE_POLICY_KEYS = frozenset(
    {"target_domain", "target_domains"}
)


def extraction_output_json_schema() -> dict[str, Any]:
    """Return the LLM-facing JSON schema for extraction-stage outputs."""

    return _background_output_schema(
        operation=_EXTRACTION_OPERATION,
        payload_schema=_payload_schema(
            {
                "atomic_belief_inputs": {
                    "type": "array",
                    "items": _atomic_belief_input_schema(),
                }
            }
        ),
    )


def consolidation_output_json_schema(
    *,
    allowed_source_atomic_belief_ids: Iterable[str] = (),
    allowed_target_belief_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the LLM-facing JSON schema for ordinary consolidation batches."""

    source_ids = tuple(
        sorted({item for item in allowed_source_atomic_belief_ids if item.strip()})
    )
    target_ids = tuple(sorted({item for item in allowed_target_belief_ids if item.strip()}))
    return _background_output_schema(
        operation=_CONSOLIDATION_BATCH_OPERATION,
        payload_schema=_consolidation_batch_payload_schema(
            allowed_source_atomic_belief_ids=source_ids,
            allowed_target_belief_ids=target_ids,
        ),
    )


def conflict_review_output_json_schema(
    *,
    allowed_target_belief_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the LLM-facing JSON schema for conflict-review outputs."""

    target_ids = tuple(sorted({item for item in allowed_target_belief_ids if item.strip()}))
    return _semantic_consolidation_output_json_schema(allowed_target_belief_ids=target_ids)


def _semantic_consolidation_output_json_schema(
    *,
    allowed_target_belief_ids: Iterable[str],
) -> dict[str, Any]:
    target_ids = tuple(allowed_target_belief_ids)
    atomic_payload = _payload_schema({"atomic_belief_input": _atomic_belief_input_schema()})
    return {
        "oneOf": [
            _background_output_schema(
                operation=_SKIP_OPERATION,
                payload_schema=_skip_payload_schema(),
            ),
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
        ]
    }


def consolidation_instruction_output_json_schema() -> dict[str, Any]:
    """Return the stable prompt schema for consolidation instructions."""

    return consolidation_output_json_schema()


def conflict_review_instruction_output_json_schema() -> dict[str, Any]:
    """Return the stable prompt schema for conflict-review instructions."""

    return conflict_review_output_json_schema()


def summary_output_json_schema(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about_refs: Iterable[tuple[str, str]],
    target_domain: str | None = None,
) -> dict[str, Any]:
    """Return the LLM-facing JSON schema for one selected summary target."""

    required = ["summary_kind", "scope", "about", "topic", "content"]
    structure_schema = (
        _summary_structure_schema(target_domain)
        if target_domain is not None
        else None
    )
    if target_domain is not None:
        required.append("structure")
    summary_draft = _summary_belief_input_schema(
        summary_kind=summary_kind,
        scope=scope,
        about_refs=about_refs,
        structure_schema=structure_schema,
        required=required,
    )
    return {
        "oneOf": [
            _background_output_schema(
                operation="create_summary_belief",
                payload_schema=_payload_schema({"summary_belief_input": summary_draft}),
            ),
            _background_output_schema(
                operation=_SKIP_OPERATION,
                payload_schema=_skip_payload_schema(),
            ),
        ]
    }


def summary_instruction_output_json_schema() -> dict[str, Any]:
    """Return the stable prompt schema for summary instructions."""

    summary_draft = _summary_belief_instruction_input_schema()
    return {
        "oneOf": [
            _background_output_schema(
                operation="create_summary_belief",
                payload_schema=_payload_schema({"summary_belief_input": summary_draft}),
            ),
            _background_output_schema(
                operation=_SKIP_OPERATION,
                payload_schema=_skip_payload_schema(),
            ),
        ]
    }


def feedback_attribution_output_json_schema() -> dict[str, Any]:
    """Return the LLM-facing JSON schema for feedback attribution outputs."""

    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["payload"],
        "properties": {
            "payload": {
                "type": "object",
                "additionalProperties": False,
                "required": ["verdicts"],
                "properties": {
                    "verdicts": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["belief_id", "verdict", "evidence_quote"],
                            "properties": {
                                "belief_id": {"type": "string", "minLength": 1},
                                "verdict": {
                                    "enum": sorted(_FEEDBACK_ATTRIBUTION_VERDICTS)
                                },
                                "evidence_quote": {"type": "string"},
                            },
                            "allOf": [
                                {
                                    "if": {
                                        "properties": {
                                            "verdict": {"const": "irrelevant"}
                                        }
                                    },
                                    "else": {
                                        "properties": {
                                            "evidence_quote": {
                                                "type": "string",
                                                "minLength": 1,
                                            }
                                        }
                                    },
                                }
                            ],
                        },
                    }
                },
            }
        },
    }


def _background_output_schema(
    *,
    operation: str,
    payload_schema: dict[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "operation",
            "authority",
            "rationale",
            "payload",
        ],
        "properties": {
            "operation": {"const": operation},
            "authority": {"const": Authority.BACKGROUND_SYNTHESIZED.value},
            "rationale": {"type": "string", "minLength": 1},
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


def _skip_payload_schema() -> dict[str, Any]:
    return _payload_schema(
        {
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            }
        }
    )


def _consolidation_batch_payload_schema(
    *,
    allowed_source_atomic_belief_ids: Iterable[str],
    allowed_target_belief_ids: Iterable[str],
) -> dict[str, Any]:
    source_ids = tuple(allowed_source_atomic_belief_ids)
    target_ids = tuple(allowed_target_belief_ids)
    return _payload_schema(
        {
            "decisions": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "oneOf": [
                        _consolidation_decision_schema(
                            operation="promote",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            source_min_items=1,
                            source_max_items=1,
                        ),
                        _consolidation_decision_schema(
                            operation="skip",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                        ),
                        _consolidation_decision_schema(
                            operation="create",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            require_atomic_input=True,
                        ),
                        _consolidation_decision_schema(
                            operation="strengthen",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            require_target=True,
                        ),
                        _consolidation_decision_schema(
                            operation="supersede",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            require_target=True,
                            require_atomic_input=True,
                        ),
                        _consolidation_decision_schema(
                            operation="retract",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            require_target=True,
                        ),
                        _consolidation_decision_schema(
                            operation="archive",
                            allowed_source_atomic_belief_ids=source_ids,
                            allowed_target_belief_ids=target_ids,
                            require_target=True,
                        ),
                    ]
                },
            }
        }
    )


def _consolidation_decision_schema(
    *,
    operation: str,
    allowed_source_atomic_belief_ids: Iterable[str],
    allowed_target_belief_ids: Iterable[str],
    require_target: bool = False,
    require_atomic_input: bool = False,
    source_min_items: int = 1,
    source_max_items: int | None = None,
) -> dict[str, Any]:
    source_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    source_ids = tuple(allowed_source_atomic_belief_ids)
    if source_ids:
        source_id_schema["enum"] = list(source_ids)
    source_ids_schema: dict[str, Any] = {
        "type": "array",
        "minItems": source_min_items,
        "items": source_id_schema,
    }
    if source_max_items is not None:
        source_ids_schema["maxItems"] = source_max_items
    target_id_schema: dict[str, Any] = {"type": "string", "minLength": 1}
    target_ids = tuple(allowed_target_belief_ids)
    if target_ids:
        target_id_schema["enum"] = list(target_ids)

    properties: dict[str, Any] = {
        "operation": {"const": operation},
        "source_atomic_belief_ids": source_ids_schema,
        "rationale": {"type": "string", "minLength": 1},
    }
    required = ["operation", "source_atomic_belief_ids", "rationale"]
    if require_target:
        properties["target_belief_id"] = target_id_schema
        required.append("target_belief_id")
    if require_atomic_input:
        properties["atomic_belief_input"] = _atomic_belief_input_schema()
        required.append("atomic_belief_input")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _atomic_belief_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["memory_kind", "scope", "about", "topic", "content"],
        "properties": {
            "memory_kind": {"enum": [item.value for item in MemoryKind]},
            "scope": {"enum": [item.value for item in BeliefScope]},
            "about": _reference_array_schema(),
            "topic": _TOPIC_SCHEMA,
            "content": {"type": "string", "minLength": 1},
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


def _summary_belief_input_schema(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about_refs: Iterable[tuple[str, str]],
    structure_schema: dict[str, Any] | None,
    required: list[str],
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "summary_kind": {"const": summary_kind.value},
        "scope": {"const": scope.value},
        "about": _reference_array_schema(const_refs=about_refs),
        "topic": _TOPIC_SCHEMA,
        "content": {"type": "string", "minLength": 1},
        "validity": {"type": "object"},
        "update_policy": {"type": "object"},
        "project_descriptor": {
            "oneOf": [
                {"type": "string", "minLength": 1},
                {"type": "object"},
            ],
        },
    }
    if structure_schema is not None:
        properties["structure"] = structure_schema
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _summary_belief_instruction_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary_kind", "scope", "about", "topic", "content"],
        "properties": {
            "summary_kind": {"enum": [item.value for item in SummaryKind]},
            "scope": {"enum": [item.value for item in BeliefScope]},
            "about": _reference_array_schema(),
            "topic": _TOPIC_SCHEMA,
            "content": {"type": "string", "minLength": 1},
            "structure": {
                "type": "object",
                "required": ["target_domain"],
                "properties": {"target_domain": {"type": "string", "minLength": 1}},
            },
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
            "atomic_belief_input": _atomic_belief_input_schema(),
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
    allow_summary_scheduling_hints: bool = False
    derivation_stage: DerivationStage = DerivationStage.BACKGROUND_EXTRACTED
    source_atomic_belief_records: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class FeedbackAttributionValidationContext:
    """Program-selected attribution inputs used to validate one LLM output."""

    allowed_belief_ids: frozenset[str]
    user_message_content: str


@dataclass(frozen=True)
class ValidatedAtomicBeliefDraft:
    """Id-less atomic belief creation input accepted from a background LLM output."""

    memory_kind: MemoryKind
    scope: BeliefScope
    about: tuple[Reference, ...]
    topic: str
    content: str
    validity: ValidityWindow | None = None
    update_policy: dict[str, Any] = field(default_factory=dict)
    project_descriptor: str | Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ValidatedSummaryBeliefDraft:
    """Id-less summary belief creation input accepted from a background LLM output."""

    summary_kind: SummaryKind
    scope: BeliefScope
    about: tuple[Reference, ...]
    topic: str
    content: str
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


@dataclass(frozen=True)
class ValidatedConsolidationDecision:
    """One source-consuming decision inside an ordinary consolidation batch."""

    operation: str
    source_atomic_belief_ids: tuple[str, ...]
    rationale: str
    target_belief_id: str | None = None
    atomic_belief_input: ValidatedAtomicBeliefDraft | None = None


@dataclass(frozen=True)
class ValidatedFeedbackAttributionVerdict:
    """Feedback verdict accepted from an attribution LLM output."""

    belief_id: str
    verdict: str
    evidence_quote: str


ValidatedPayload = (
    ValidatedAtomicBeliefDraft
    | ValidatedSummaryBeliefDraft
    | ValidatedBeliefUpdate
    | ValidatedConsolidationDecision
)


@dataclass(frozen=True)
class ValidatedBackgroundLLMOutput:
    """Validated common envelope plus one stage-specific payload."""

    operation: str
    authority: Authority
    rationale: str
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


def validate_feedback_attribution_json(
    raw_output: str,
    context: FeedbackAttributionValidationContext,
) -> tuple[ValidatedFeedbackAttributionVerdict, ...]:
    """Parse and validate a fixture or provider JSON string for attribution."""

    try:
        decoded = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise BackgroundLLMValidationError(
            f"malformed feedback attribution JSON: {exc}"
        ) from exc
    if not isinstance(decoded, dict):
        raise BackgroundLLMValidationError(
            "malformed feedback attribution output must be an object"
        )
    return validate_feedback_attribution_output(decoded, context)


def validate_feedback_attribution_output(
    output: Mapping[str, Any],
    context: FeedbackAttributionValidationContext,
) -> tuple[ValidatedFeedbackAttributionVerdict, ...]:
    """Validate decoded feedback attribution structured output."""

    _reject_numeric_strength_fields(output)
    _validate_exact_keys(output, {"payload"}, "feedback attribution output")
    _reject_prompt_injection_except_evidence_quote(output)

    payload = output.get("payload")
    if not isinstance(payload, Mapping):
        raise BackgroundLLMValidationError("payload must be an object")
    _validate_exact_keys(payload, {"verdicts"}, "feedback attribution payload")
    raw_verdicts = payload.get("verdicts")
    if not isinstance(raw_verdicts, list) or not raw_verdicts:
        raise BackgroundLLMValidationError("payload.verdicts must be a non-empty array")

    allowed_ids = frozenset(item for item in context.allowed_belief_ids if item.strip())
    if not allowed_ids:
        raise BackgroundLLMValidationError("allowed belief id whitelist must be non-empty")

    seen: set[str] = set()
    verdicts: list[ValidatedFeedbackAttributionVerdict] = []
    for index, raw_verdict in enumerate(raw_verdicts):
        if not isinstance(raw_verdict, Mapping):
            raise BackgroundLLMValidationError(
                f"payload.verdicts[{index}] must be an object"
            )
        _validate_exact_keys(
            raw_verdict,
            {"belief_id", "verdict", "evidence_quote"},
            f"payload.verdicts[{index}]",
        )
        raw_belief_id = raw_verdict.get("belief_id")
        if not isinstance(raw_belief_id, str) or not raw_belief_id.strip():
            raise BackgroundLLMValidationError("belief_id is required")
        belief_id = raw_belief_id
        if belief_id in seen:
            raise BackgroundLLMValidationError(f"duplicate belief id: {belief_id}")
        seen.add(belief_id)
        if belief_id not in allowed_ids:
            raise BackgroundLLMValidationError(
                f"belief id {belief_id!r} is outside the attribution whitelist"
            )
        raw_verdict_value = raw_verdict.get("verdict")
        if not isinstance(raw_verdict_value, str) or not raw_verdict_value.strip():
            raise BackgroundLLMValidationError("verdict is required")
        verdict = raw_verdict_value
        if verdict not in _FEEDBACK_ATTRIBUTION_VERDICTS:
            allowed = ", ".join(sorted(_FEEDBACK_ATTRIBUTION_VERDICTS))
            raise BackgroundLLMValidationError(
                f"unsupported feedback attribution verdict {verdict!r}; allowed: {allowed}"
            )
        quote = raw_verdict.get("evidence_quote")
        if not isinstance(quote, str):
            raise BackgroundLLMValidationError("evidence_quote must be a string")
        if verdict != "irrelevant" and not quote.strip():
            raise BackgroundLLMValidationError(
                "evidence_quote is required for non-irrelevant verdicts"
            )
        if quote and quote not in context.user_message_content:
            raise BackgroundLLMValidationError(
                "evidence_quote must be a verbatim substring of the user message"
            )
        verdicts.append(
            ValidatedFeedbackAttributionVerdict(
                belief_id=belief_id,
                verdict=verdict,
                evidence_quote=quote,
            )
        )

    missing = allowed_ids - seen
    if missing:
        raise BackgroundLLMValidationError(
            "missing feedback attribution verdicts for belief ids: "
            + ", ".join(sorted(missing))
        )
    return tuple(verdicts)


def validate_background_llm_output(
    output: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> ValidatedBackgroundLLMOutput:
    """Validate decoded background LLM structured output."""

    _reject_numeric_strength_fields(output)
    _reject_forbidden_provenance_keys(output)
    _reject_prompt_injection(output)
    _validate_background_output_keys(output)

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
        source_span_note=source_span_note,
        payloads=payloads,
    )


def _validate_payloads(
    *,
    operation: str,
    payload: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> tuple[ValidatedPayload, ...]:
    if operation == _CONSOLIDATION_BATCH_OPERATION:
        return _validate_consolidation_decisions(payload, context)
    if operation == "create_atomic_belief":
        if BackgroundStage(context.source_window.stage) == BackgroundStage.EXTRACTION:
            return _validate_atomic_drafts(payload.get("atomic_belief_inputs"), context)
        return (_validate_atomic_draft(payload.get("atomic_belief_input"), context),)
    if operation == "create_summary_belief":
        return (_validate_summary_draft(payload.get("summary_belief_input"), context),)
    if operation == "profile_summary_candidate":
        return (_validate_summary_draft(payload.get("profile_summary_candidate"), context),)
    if operation == "update_belief":
        return (_validate_belief_update(payload.get("belief_update"), context, operation),)
    if operation == _SKIP_OPERATION:
        _validate_skip_payload(payload)
        return ()
    if operation == "create":
        return (_validate_atomic_draft(payload.get("atomic_belief_input"), context),)
    if operation == "supersede":
        return (
            _validate_belief_update(payload.get("belief_update"), context, operation),
            _validate_atomic_draft(payload.get("atomic_belief_input"), context),
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
    if stage == BackgroundStage.CONSOLIDATION:
        _validate_ordinary_consolidation_stage_output_shape(
            operation=operation,
            payload=payload,
        )
        return
    if stage == BackgroundStage.CONFLICT_REVIEW:
        _validate_conflict_review_stage_output_shape(operation=operation, payload=payload)
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
    payload_keys = {str(key) for key in payload}
    if payload_keys != _EXTRACTION_PAYLOAD_KEYS:
        unexpected = payload_keys - _EXTRACTION_PAYLOAD_KEYS
        missing = _EXTRACTION_PAYLOAD_KEYS - payload_keys
        details = []
        if missing:
            details.append(f"missing payload keys: {', '.join(sorted(missing))}")
        if unexpected:
            details.append(f"unexpected payload keys: {', '.join(sorted(unexpected))}")
        detail_text = "; ".join(details)
        raise BackgroundLLMValidationError(
            "extraction payload must contain exactly atomic_belief_inputs"
            + (f"; {detail_text}" if detail_text else "")
        )


def _validate_ordinary_consolidation_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    if operation != _CONSOLIDATION_BATCH_OPERATION:
        raise BackgroundLLMValidationError(
            "ordinary consolidation accepts only consolidate_atomic_beliefs outputs"
        )
    keys = {str(key) for key in payload}
    if keys != _CONSOLIDATION_BATCH_PAYLOAD_KEYS:
        raise BackgroundLLMValidationError(
            "ordinary consolidation payload must contain exactly decisions"
        )


def _validate_conflict_review_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    if operation not in _SEMANTIC_OPERATIONS:
        raise BackgroundLLMValidationError(
            "consolidation stages accept only semantic operations: create, strengthen, "
            "supersede, retract, archive, skip"
        )
    keys = {str(key) for key in payload}
    expected: frozenset[str]
    if operation == _SKIP_OPERATION:
        expected = _SKIP_PAYLOAD_KEYS
    elif operation == "create":
        expected = frozenset({"atomic_belief_input"})
    elif operation in {"strengthen", "retract", "archive"}:
        expected = frozenset({"belief_update"})
    else:
        expected = frozenset({"belief_update", "atomic_belief_input"})
    if keys != expected:
        expected_text = ", ".join(sorted(expected))
        raise BackgroundLLMValidationError(
            f"{operation} payload must contain exactly: {expected_text}"
        )


def _validate_background_output_keys(output: Mapping[str, Any]) -> None:
    allowed = frozenset({"operation", "authority", "rationale", "source_span_note", "payload"})
    _validate_allowed_keys(output, allowed, "background LLM output")


def _validate_summary_stage_output_shape(
    *,
    operation: str,
    payload: Mapping[str, Any],
) -> None:
    expected: frozenset[str]
    if operation == "create_summary_belief":
        expected = frozenset({"summary_belief_input"})
    elif operation == "profile_summary_candidate":
        expected = frozenset({"profile_summary_candidate"})
    elif operation == _SKIP_OPERATION:
        expected = _SKIP_PAYLOAD_KEYS
    else:
        raise BackgroundLLMValidationError(
            "summary stage accepts only create_summary_belief, "
            "profile_summary_candidate, or skip outputs"
        )
    keys = {str(key) for key in payload}
    if keys != expected:
        expected_text = ", ".join(sorted(expected))
        raise BackgroundLLMValidationError(
            f"{operation} payload must contain exactly: {expected_text}"
        )


def _validate_skip_payload(payload: Mapping[str, Any]) -> None:
    _validate_exact_keys(payload, set(_SKIP_PAYLOAD_KEYS), "skip payload")
    reason = _required_str(payload, "reason")
    if len(reason) > 256:
        raise BackgroundLLMValidationError("skip reason must be at most 256 characters")


def _validate_consolidation_decisions(
    payload: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
) -> tuple[ValidatedConsolidationDecision, ...]:
    _validate_exact_keys(payload, set(_CONSOLIDATION_BATCH_PAYLOAD_KEYS), "consolidation payload")
    raw_decisions = payload.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise BackgroundLLMValidationError("payload.decisions must be a non-empty array")

    allowed_source_ids = tuple(
        source_ref.source_id
        for source_ref in context.source_window.source_refs
        if source_ref.source_type == "atomic_belief"
    )
    allowed_source_set = frozenset(allowed_source_ids)
    if not allowed_source_set:
        raise BackgroundLLMValidationError(
            "ordinary consolidation requires atomic_belief source refs"
        )
    if len(allowed_source_ids) != len(allowed_source_set):
        raise BackgroundLLMValidationError("duplicate source id in consolidation source window")

    seen_source_ids: set[str] = set()
    decisions: list[ValidatedConsolidationDecision] = []
    for index, raw_decision in enumerate(raw_decisions):
        decision = _validate_consolidation_decision(raw_decision, context, index=index)
        for source_id in decision.source_atomic_belief_ids:
            if source_id not in allowed_source_set:
                raise BackgroundLLMValidationError(
                    f"unknown source_atomic_belief_id {source_id!r}"
                )
            if source_id in seen_source_ids:
                raise BackgroundLLMValidationError(
                    f"duplicate source_atomic_belief_id {source_id!r}"
                )
            seen_source_ids.add(source_id)
        decisions.append(decision)

    missing = allowed_source_set - seen_source_ids
    if missing:
        raise BackgroundLLMValidationError(
            "missing source_atomic_belief_ids: " + ", ".join(sorted(missing))
        )
    return tuple(decisions)


def _validate_consolidation_decision(
    raw: object,
    context: BackgroundLLMValidationContext,
    *,
    index: int,
) -> ValidatedConsolidationDecision:
    label = f"payload.decisions[{index}]"
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError(f"{label} must be an object")
    allowed_keys = _CONSOLIDATION_DECISION_BASE_KEYS.union(
        {"target_belief_id", "atomic_belief_input"}
    )
    _validate_allowed_keys(raw, allowed_keys, label)

    operation = _required_str(raw, "operation")
    if operation not in _CONSOLIDATION_DECISION_OPERATIONS:
        allowed = ", ".join(sorted(_CONSOLIDATION_DECISION_OPERATIONS))
        raise BackgroundLLMValidationError(
            f"unsupported consolidation decision operation {operation!r}; allowed: {allowed}"
        )
    source_ids = _required_source_atomic_belief_ids(
        raw.get("source_atomic_belief_ids"),
        label=label,
    )
    rationale = _required_str(raw, "rationale")
    target_belief_id = _optional_target_belief_id(raw, context, label=label)
    atomic_belief_input = (
        _validate_atomic_draft(raw.get("atomic_belief_input"), context)
        if "atomic_belief_input" in raw
        else None
    )

    _validate_consolidation_decision_fields(
        operation=operation,
        source_ids=source_ids,
        target_belief_id=target_belief_id,
        atomic_belief_input=atomic_belief_input,
        label=label,
    )
    if operation == "create" and len(source_ids) == 1:
        assert atomic_belief_input is not None
        _reject_single_source_duplicate_create(
            atomic_belief_input,
            source_id=source_ids[0],
            context=context,
        )
    return ValidatedConsolidationDecision(
        operation=operation,
        source_atomic_belief_ids=source_ids,
        rationale=rationale,
        target_belief_id=target_belief_id,
        atomic_belief_input=atomic_belief_input,
    )


def _required_source_atomic_belief_ids(
    raw: object,
    *,
    label: str,
) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise BackgroundLLMValidationError(
            f"{label}.source_atomic_belief_ids must be a non-empty array"
        )
    source_ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise BackgroundLLMValidationError(
                f"{label}.source_atomic_belief_ids entries must be non-empty strings"
            )
        source_ids.append(item.strip())
    if len(source_ids) != len(set(source_ids)):
        raise BackgroundLLMValidationError("duplicate source_atomic_belief_id in decision")
    return tuple(source_ids)


def _optional_target_belief_id(
    raw: Mapping[str, Any],
    context: BackgroundLLMValidationContext,
    *,
    label: str,
) -> str | None:
    if "target_belief_id" not in raw:
        return None
    target_belief_id = _required_str(raw, "target_belief_id")
    if target_belief_id not in context.allowed_target_belief_ids:
        raise BackgroundLLMValidationError(
            f"{label}.target_belief_id {target_belief_id!r} was not included in LLM input"
        )
    return target_belief_id


def _validate_consolidation_decision_fields(
    *,
    operation: str,
    source_ids: tuple[str, ...],
    target_belief_id: str | None,
    atomic_belief_input: ValidatedAtomicBeliefDraft | None,
    label: str,
) -> None:
    if operation == "promote" and len(source_ids) != 1:
        raise BackgroundLLMValidationError(f"{label}.promote requires exactly one source")
    target_required = operation in {"strengthen", "supersede", "retract", "archive"}
    atomic_required = operation in {"create", "supersede"}
    if target_required and target_belief_id is None:
        raise BackgroundLLMValidationError(f"{label}.{operation} requires target_belief_id")
    if not target_required and target_belief_id is not None:
        raise BackgroundLLMValidationError(f"{label}.{operation} forbids target_belief_id")
    if atomic_required and atomic_belief_input is None:
        raise BackgroundLLMValidationError(f"{label}.{operation} requires atomic_belief_input")
    if not atomic_required and atomic_belief_input is not None:
        raise BackgroundLLMValidationError(f"{label}.{operation} forbids atomic_belief_input")


def _reject_single_source_duplicate_create(
    draft: ValidatedAtomicBeliefDraft,
    *,
    source_id: str,
    context: BackgroundLLMValidationContext,
) -> None:
    source_record = context.source_atomic_belief_records.get(source_id)
    if not isinstance(source_record, Mapping):
        raise BackgroundLLMValidationError(
            "single-source create requires source atomic belief record for duplicate-copy "
            "validation; use promote when preserving the source unchanged"
        )
    if _draft_matches_source_record(draft, source_record):
        raise BackgroundLLMValidationError(
            "single-source create duplicates the source atomic belief; use promote"
        )


def _draft_matches_source_record(
    draft: ValidatedAtomicBeliefDraft,
    source_record: Mapping[str, Any],
) -> bool:
    source_update_policy = source_record.get("update_policy") or {}
    if not isinstance(source_update_policy, Mapping):
        source_update_policy = {}
    source_validity = source_record.get("validity") or {}
    if not isinstance(source_validity, Mapping):
        source_validity = {}
    draft_validity = draft.validity.to_record() if draft.validity is not None else None
    validity_matches = draft_validity is None or dict(source_validity) == draft_validity
    return (
        source_record.get("memory_kind") == draft.memory_kind.value
        and source_record.get("scope") == draft.scope.value
        and _normalized_reference_records(source_record.get("about")) == (
            _materialized_draft_about_records_for_duplicate_check(draft)
        )
        and str(source_record.get("topic", "")).strip() == draft.topic
        and str(source_record.get("content", "")).strip() == draft.content
        and dict(source_update_policy) == draft.update_policy
        and validity_matches
    )


def _materialized_draft_about_records_for_duplicate_check(
    draft: ValidatedAtomicBeliefDraft,
) -> tuple[tuple[str, str], ...]:
    if draft.scope == BeliefScope.PROJECT and draft.project_descriptor is not None:
        return (("project", _project_reference_id(draft.project_descriptor)),)
    return _normalized_reference_records([ref.to_record() for ref in draft.about])


def _normalized_reference_records(raw: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(raw, list):
        return ()
    refs: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, Mapping):
            kind = item.get("kind")
            ref_id = item.get("id")
            if isinstance(kind, str) and isinstance(ref_id, str):
                refs.append((kind, ref_id))
    return tuple(sorted(refs))


def _validate_atomic_draft(
    raw: object,
    context: BackgroundLLMValidationContext,
) -> ValidatedAtomicBeliefDraft:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError("atomic_belief_input must be an object")
    _reject_generated_draft_ids(raw, label="atomic_belief_input")
    _validate_allowed_keys(raw, _ATOMIC_DRAFT_KEYS, "atomic_belief_input")
    try:
        memory_kind = MemoryKind(_required_str(raw, "memory_kind"))
    except ValueError as exc:
        raise BackgroundLLMValidationError(
            f"unsupported memory_kind: {raw.get('memory_kind')}"
        ) from exc
    scope, about, project_descriptor = _validate_scope_about(raw, context)
    content = _required_str(raw, "content")
    topic = _required_topic(raw, content)
    return ValidatedAtomicBeliefDraft(
        memory_kind=memory_kind,
        scope=scope,
        about=about,
        topic=topic,
        content=content,
        validity=_validity(raw.get("validity")),
        update_policy=_update_policy(raw.get("update_policy"), context),
        project_descriptor=project_descriptor,
    )


def _validate_atomic_drafts(
    raw: object,
    context: BackgroundLLMValidationContext,
) -> tuple[ValidatedAtomicBeliefDraft, ...]:
    if not isinstance(raw, list):
        raise BackgroundLLMValidationError("atomic_belief_inputs must be an array")
    return tuple(_validate_atomic_draft(item, context) for item in raw)


def _validate_summary_draft(
    raw: object,
    context: BackgroundLLMValidationContext,
) -> ValidatedSummaryBeliefDraft:
    if not isinstance(raw, Mapping):
        raise BackgroundLLMValidationError("summary_belief_input must be an object")
    _reject_generated_draft_ids(raw, label="summary_belief_input")
    _validate_allowed_keys(raw, _SUMMARY_DRAFT_KEYS, "summary_belief_input")
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
    topic = _required_topic(raw, content)
    if context.required_summary_target_domain is None:
        if "structure" in raw:
            if not _is_null_target_domain_structure(raw.get("structure")):
                raise BackgroundLLMValidationError(
                    "summary structure is only allowed for selected domain summary targets"
                )
        structure = None
    else:
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
        topic=topic,
        content=content,
        structure=structure,
        validity=_validity(raw.get("validity")),
        update_policy=_update_policy(raw.get("update_policy"), context),
        project_descriptor=project_descriptor,
    )


def _is_null_target_domain_structure(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"target_domain"}
        and value.get("target_domain") is None
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
    return operation


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
    try:
        _normalize_project_descriptor(descriptor)
    except ValueError:
        return False
    return True


def _project_reference_id(descriptor: str | Mapping[str, Any]) -> str:
    normalized = _normalize_project_descriptor(descriptor)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"project:{digest}"


def _normalize_project_descriptor(descriptor: str | Mapping[str, Any]) -> str:
    if isinstance(descriptor, str):
        normalized = _normalize_descriptor_text(descriptor)
        if not normalized:
            raise ValueError("project descriptor must be resolvable")
        return normalized
    for key in ("name", "repository", "repo"):
        value = descriptor.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_descriptor_text(value)
    normalized = _normalize_descriptor_text(
        json.dumps(dict(descriptor), ensure_ascii=False, sort_keys=True)
    )
    if not normalized or normalized == "{}":
        raise ValueError("project descriptor must be resolvable")
    return normalized


def _normalize_descriptor_text(value: str) -> str:
    normalized = value.replace("\\", "/").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"/+", "/", normalized)
    return normalized.rstrip("/")


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


def _reject_prompt_injection_except_evidence_quote(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) == "evidence_quote":
                continue
            _reject_prompt_injection_except_evidence_quote(nested)
        return
    if isinstance(value, list):
        for item in value:
            _reject_prompt_injection_except_evidence_quote(item)
        return
    if not isinstance(value, str):
        return
    normalized = value.casefold()
    if any(pattern in normalized for pattern in _PROMPT_INJECTION_PATTERNS):
        raise BackgroundLLMValidationError("prompt-injection content is not allowed")


def _validate_exact_keys(
    raw: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> None:
    keys = {str(key) for key in raw}
    if keys == expected:
        return
    missing = expected - keys
    unexpected = keys - expected
    details = []
    if missing:
        details.append(f"missing keys: {', '.join(sorted(missing))}")
    if unexpected:
        details.append(f"unknown keys: {', '.join(sorted(unexpected))}")
    raise BackgroundLLMValidationError(
        f"{label} must contain exactly {sorted(expected)}; {'; '.join(details)}"
    )


def _validate_allowed_keys(
    raw: Mapping[str, Any],
    allowed: frozenset[str],
    label: str,
) -> None:
    unexpected = sorted({str(key) for key in raw} - allowed)
    if unexpected:
        raise BackgroundLLMValidationError(
            f"{label} contains unknown keys: {', '.join(unexpected)}"
        )


def _required_str(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BackgroundLLMValidationError(f"{key} is required")
    return value.strip()


def _required_topic(raw: Mapping[str, Any], content: str) -> str:
    try:
        return validate_belief_topic(_required_str(raw, "topic"), content)
    except (TypeError, ValueError) as exc:
        raise BackgroundLLMValidationError(str(exc)) from exc


def _update_policy(
    value: object,
    context: BackgroundLLMValidationContext,
) -> dict[str, Any]:
    update_policy = _optional_dict(value) or {}
    if context.allow_summary_scheduling_hints:
        return update_policy
    forbidden = sorted(_SUMMARY_SCHEDULING_UPDATE_POLICY_KEYS.intersection(update_policy))
    if forbidden:
        raise BackgroundLLMValidationError(
            "update_policy summary scheduling targets are program-owned and cannot be "
            f"generated by ordinary background LLM outputs: {', '.join(forbidden)}"
        )
    return update_policy


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
