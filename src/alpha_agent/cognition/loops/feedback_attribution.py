"""Deterministic helpers for feedback attribution over recalled beliefs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundSourceProgress,
    BackgroundSourceRef,
    BackgroundStage,
    ProcessingLedger,
)
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore

MEMORY_RECALL_TOOL_NAME = "memory_recall"
_FEEDBACK_SOURCE_TYPE = "session_message"


@dataclass(frozen=True)
class RecalledBeliefHandle:
    """Belief handle recovered from a memory_recall tool message."""

    belief_id: str
    content: str
    memory_kind: str
    scope: str
    source_tool_message_ids: tuple[str, ...]


def recalled_beliefs_for_previous_turn(
    store: StateStore,
    session_id: str,
    before_ordinal: int,
) -> list[RecalledBeliefHandle]:
    """Return memory_recall belief handles from the immediately previous turn."""

    messages = store.list_session_messages(session_id, before_ordinal=before_ordinal)
    previous_turn: list[SessionMessage] = []
    for message in reversed(messages):
        if message.kind == "user_message":
            break
        if message.kind in {"assistant_message", "tool_message"}:
            previous_turn.append(message)

    handles: dict[str, RecalledBeliefHandle] = {}
    for message in reversed(previous_turn):
        if not _is_memory_recall_tool_message(message):
            continue
        for result in _memory_recall_results(message):
            belief_id = result["id"]
            existing = handles.get(belief_id)
            if existing is None:
                handles[belief_id] = RecalledBeliefHandle(
                    belief_id=belief_id,
                    content=result["content"],
                    memory_kind=result["memory_kind"],
                    scope=result["scope"],
                    source_tool_message_ids=(message.id,),
                )
                continue
            if message.id not in existing.source_tool_message_ids:
                handles[belief_id] = replace(
                    existing,
                    source_tool_message_ids=(
                        *existing.source_tool_message_ids,
                        message.id,
                    ),
                )
    return list(handles.values())


def feedback_attribution_target_unit(session_id: str) -> str:
    """Return the processing-ledger target unit for a session attribution job."""

    if not session_id.strip():
        raise ValueError("session_id must be non-empty")
    return f"session:{session_id}"


def feedback_attribution_idempotency_key(
    *,
    session_id: str,
    recall_tool_message_id: str,
) -> str:
    """Return the deterministic ledger key for one recall tool message."""

    return (
        f"{BackgroundStage.FEEDBACK_ATTRIBUTION.value}:"
        f"{feedback_attribution_target_unit(session_id)}:"
        f"{_FEEDBACK_SOURCE_TYPE}:{recall_tool_message_id}"
    )


def claim_feedback_attribution_sources(
    ledger: ProcessingLedger,
    *,
    session_id: str,
    recall_tool_message_ids: Sequence[str],
    claimed_by: str,
    worker_slot_acquired: bool = True,
) -> tuple[BackgroundSourceProgress, ...]:
    """Claim recall tool messages for feedback attribution if not already active."""

    if not worker_slot_acquired:
        return ()
    target_unit = feedback_attribution_target_unit(session_id)
    claimed: list[BackgroundSourceProgress] = []
    with ledger.store.immediate_transaction() as conn:
        for message_id in _stable_unique(recall_tool_message_ids):
            source_ref = _feedback_source_ref(message_id)
            existing = _get_existing_progress(
                ledger,
                source_ref=source_ref,
                target_unit=target_unit,
                conn=conn,
            )
            if existing is not None and existing.status in {
                BackgroundProgressStatus.CLAIMED,
                BackgroundProgressStatus.PROCESSED,
            }:
                continue
            claimed.append(
                ledger.claim_source(
                    source_ref,
                    stage=BackgroundStage.FEEDBACK_ATTRIBUTION,
                    target_unit=target_unit,
                    claimed_by=claimed_by,
                    idempotency_key=feedback_attribution_idempotency_key(
                        session_id=session_id,
                        recall_tool_message_id=message_id,
                    ),
                    conn=conn,
                )
            )
    return tuple(claimed)


def complete_feedback_attribution_sources(
    ledger: ProcessingLedger,
    *,
    session_id: str,
    recall_tool_message_ids: Sequence[str],
    checkpoint_id: str | None = None,
) -> tuple[BackgroundSourceProgress, ...]:
    """Mark claimed feedback attribution sources as processed."""

    return _mark_claimed_feedback_attribution_sources(
        ledger,
        session_id=session_id,
        recall_tool_message_ids=recall_tool_message_ids,
        status=BackgroundProgressStatus.PROCESSED,
        checkpoint_id=checkpoint_id,
    )


def fail_feedback_attribution_sources(
    ledger: ProcessingLedger,
    *,
    session_id: str,
    recall_tool_message_ids: Sequence[str],
    error: str,
) -> tuple[BackgroundSourceProgress, ...]:
    """Mark claimed feedback attribution sources as failed with the given error."""

    return _mark_claimed_feedback_attribution_sources(
        ledger,
        session_id=session_id,
        recall_tool_message_ids=recall_tool_message_ids,
        status=BackgroundProgressStatus.FAILED,
        error=error,
    )


def _mark_claimed_feedback_attribution_sources(
    ledger: ProcessingLedger,
    *,
    session_id: str,
    recall_tool_message_ids: Sequence[str],
    status: BackgroundProgressStatus,
    checkpoint_id: str | None = None,
    error: str | None = None,
) -> tuple[BackgroundSourceProgress, ...]:
    target_unit = feedback_attribution_target_unit(session_id)
    marked: list[BackgroundSourceProgress] = []
    with ledger.store.immediate_transaction() as conn:
        for message_id in _stable_unique(recall_tool_message_ids):
            source_ref = _feedback_source_ref(message_id)
            existing = _get_existing_progress(
                ledger,
                source_ref=source_ref,
                target_unit=target_unit,
                conn=conn,
            )
            if existing is None or existing.status != BackgroundProgressStatus.CLAIMED:
                continue
            key = feedback_attribution_idempotency_key(
                session_id=session_id,
                recall_tool_message_id=message_id,
            )
            if status == BackgroundProgressStatus.PROCESSED:
                marked.append(
                    ledger.mark_source_processed(
                        source_ref,
                        stage=BackgroundStage.FEEDBACK_ATTRIBUTION,
                        target_unit=target_unit,
                        checkpoint_id=checkpoint_id,
                        idempotency_key=key,
                        conn=conn,
                    )
                )
            elif status == BackgroundProgressStatus.FAILED:
                marked.append(
                    ledger.mark_source_failed(
                        source_ref,
                        stage=BackgroundStage.FEEDBACK_ATTRIBUTION,
                        target_unit=target_unit,
                        error=error or "feedback attribution failed",
                        idempotency_key=key,
                        conn=conn,
                    )
                )
            else:
                raise ValueError(f"unsupported feedback attribution terminal status: {status}")
    return tuple(marked)


def _is_memory_recall_tool_message(message: SessionMessage) -> bool:
    return (
        message.kind == "tool_message"
        and message.provider_metadata.get("tool_name") == MEMORY_RECALL_TOOL_NAME
    )


def _memory_recall_results(message: SessionMessage) -> list[dict[str, str]]:
    raw_content = (
        message.model_content
        if message.model_content is not None
        else message.raw_content
    )
    payload = _loads_mapping(raw_content)
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        return []
    results: list[dict[str, str]] = []
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        belief_id = _non_empty_str(item.get("id"))
        result_content = _string_value(item.get("content"))
        memory_kind = _string_value(item.get("memory_kind"))
        scope = _string_value(item.get("scope"))
        if (
            belief_id is None
            or result_content is None
            or memory_kind is None
            or scope is None
        ):
            continue
        results.append(
            {
                "id": belief_id,
                "content": result_content,
                "memory_kind": memory_kind,
                "scope": scope,
            }
        )
    return results


def _loads_mapping(raw_content: str) -> Mapping[str, Any]:
    try:
        loaded = json.loads(raw_content)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, Mapping) else {}


def _non_empty_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value.strip() or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _feedback_source_ref(recall_tool_message_id: str) -> BackgroundSourceRef:
    return BackgroundSourceRef(_FEEDBACK_SOURCE_TYPE, recall_tool_message_id)


def _get_existing_progress(
    ledger: ProcessingLedger,
    *,
    source_ref: BackgroundSourceRef,
    target_unit: str,
    conn: Any,
) -> BackgroundSourceProgress | None:
    try:
        return ledger.get_source_progress(
            source_ref,
            stage=BackgroundStage.FEEDBACK_ATTRIBUTION,
            target_unit=target_unit,
            conn=conn,
        )
    except KeyError:
        return None


__all__ = [
    "MEMORY_RECALL_TOOL_NAME",
    "RecalledBeliefHandle",
    "claim_feedback_attribution_sources",
    "complete_feedback_attribution_sources",
    "fail_feedback_attribution_sources",
    "feedback_attribution_idempotency_key",
    "feedback_attribution_target_unit",
    "recalled_beliefs_for_previous_turn",
]
