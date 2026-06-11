"""Deterministic helpers for feedback attribution over recalled beliefs."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from threading import BoundedSemaphore, Lock, Thread, current_thread
from typing import Any, cast

from alpha_agent.cognition.background_llm_contract import (
    FeedbackAttributionValidationContext,
    ValidatedFeedbackAttributionVerdict,
    feedback_attribution_output_json_schema,
    validate_feedback_attribution_json,
)
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops.workers._common import json_for_prompt
from alpha_agent.cognition.models import CognitiveEvent, CognitiveEventKind, EventId, Reference
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundSourceProgress,
    BackgroundSourceRef,
    BackgroundStage,
    ProcessingLedger,
)
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.llm.base import JSON_OBJECT_RESPONSE_FORMAT, ChatMessage, LLMProvider
from alpha_agent.llm.tracing import LLMTraceLogger, traced_llm_complete
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore

MEMORY_RECALL_TOOL_NAME = "memory_recall"
_FEEDBACK_SOURCE_TYPE = "session_message"
_FEEDBACK_ATTRIBUTION_WORKER = "feedback_attribution"
_FEEDBACK_KIND_BY_VERDICT = {
    "confirmed": "belief_confirmed",
    "contradicted": "belief_contradicted",
    "corrected": "belief_corrected",
}
_CONFLICT_VERDICTS = frozenset({"contradicted", "corrected"})
_MAX_BELIEF_PROMPT_CONTENT_CHARS = 500

_FEEDBACK_ATTRIBUTION_INSTRUCTION = """Attribute the newest user message against recalled beliefs.

Return exactly one JSON object and no markdown. The output must validate against this
JSON Schema:
{output_schema_json}

Rules:
- Return exactly one verdict for every recalled belief id.
- Use "confirmed" when the user explicitly supports the recalled belief.
- Use "contradicted" when the user says the recalled belief is false.
- Use "corrected" when the user provides a replacement or correction.
- Use "irrelevant" when the user message does not bear on the belief.
- For confirmed, contradicted, or corrected, evidence_quote must be a verbatim
  substring of the user message.
- For irrelevant, evidence_quote must be an empty string.
- Do not include confidence, scores, numeric strength, rationale, source ids, or
  any keys outside the schema.

Newest user message:
{user_message_json}

Recalled beliefs:
{recalled_beliefs_json}"""


@dataclass(frozen=True)
class RecalledBeliefHandle:
    """Belief handle recovered from a memory_recall tool message."""

    belief_id: str
    content: str
    memory_kind: str
    scope: str
    source_tool_message_ids: tuple[str, ...]


@dataclass(frozen=True)
class FeedbackAttributionJob:
    """Foreground snapshot needed to attribute feedback without reading session state."""

    session_id: str
    turn_id: str
    turn_received_event_id: str
    user_message_id: str
    user_message_text: str
    prompt_messages: Sequence[ChatMessage]
    recalled_beliefs: Sequence[RecalledBeliefHandle]
    recall_tool_message_ids: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_messages",
            tuple(cast(ChatMessage, dict(message)) for message in self.prompt_messages),
        )
        object.__setattr__(self, "recalled_beliefs", tuple(self.recalled_beliefs))
        recall_ids = tuple(self.recall_tool_message_ids) or tuple(
            message_id
            for handle in self.recalled_beliefs
            for message_id in handle.source_tool_message_ids
        )
        object.__setattr__(self, "recall_tool_message_ids", _stable_unique(recall_ids))


class RealtimeFeedbackAttributionService:
    """Submit realtime feedback attribution work in bounded daemon threads."""

    def __init__(
        self,
        *,
        store: StateStore,
        llm_provider: LLMProvider,
        max_workers: int = 2,
        enabled: bool = True,
        llm_trace_logger: LLMTraceLogger | None = None,
        worker_id: str = _FEEDBACK_ATTRIBUTION_WORKER,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.enabled = enabled
        self.llm_trace_logger = llm_trace_logger
        self.worker_id = worker_id
        self._slots = BoundedSemaphore(max(1, int(max_workers)))
        self._lock = Lock()
        self._threads: set[Thread] = set()
        self._closed = False

    def submit(self, job: FeedbackAttributionJob) -> bool:
        """Start one attribution job if capacity and ledger idempotency allow it."""

        if not self.enabled:
            return False
        recall_tool_message_ids = _stable_unique(job.recall_tool_message_ids)
        if not recall_tool_message_ids or not job.recalled_beliefs:
            return False
        if not self._slots.acquire(blocking=False):
            self._write_audit(
                "feedback_attribution_saturated",
                _job_audit_payload(job),
            )
            return False

        claimed: tuple[BackgroundSourceProgress, ...] = ()
        thread_started = False
        try:
            with self._lock:
                if self._closed:
                    self._slots.release()
                    return False
                claimed = claim_feedback_attribution_sources(
                    CognitionStateStore(self.store).ledger,
                    session_id=job.session_id,
                    recall_tool_message_ids=recall_tool_message_ids,
                    claimed_by=self.worker_id,
                    worker_slot_acquired=True,
                )
                if len(claimed) != len(recall_tool_message_ids):
                    self._slots.release()
                    return False
                thread = Thread(
                    target=self._run_job,
                    args=(job,),
                    name="alpha-feedback-attribution",
                    daemon=True,
                )
                self._threads.add(thread)
                thread.start()
                thread_started = True
        except Exception as exc:
            if not thread_started:
                if claimed:
                    try:
                        fail_feedback_attribution_sources(
                            CognitionStateStore(self.store).ledger,
                            session_id=job.session_id,
                            recall_tool_message_ids=recall_tool_message_ids,
                            error=str(exc) or type(exc).__name__,
                        )
                    except Exception:
                        pass
                self._slots.release()
            self._write_audit(
                "feedback_attribution_failed",
                {
                    **_job_audit_payload(job),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return False
        return True

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> None:
        """Prevent new submissions and optionally wait for already-started jobs."""

        with self._lock:
            self._closed = True
            threads = list(self._threads)
        if not wait:
            return
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        for thread in threads:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(timeout=remaining)

    def _run_job(self, job: FeedbackAttributionJob) -> None:
        event_ids: list[str] = []
        try:
            verdicts = self._validated_verdicts(job)
            emitter = EventEmitter(SQLiteEventLog(self.store))
            state_service = CognitionStateStore(self.store)
            for verdict in verdicts:
                if verdict.verdict == "irrelevant":
                    continue
                event = _emit_feedback_event(emitter, job, verdict)
                event_ids.append(str(event.id))
                state_service.record_belief_feedback(
                    verdict.belief_id,
                    kind=verdict.verdict,
                    event_id=str(event.id),
                    at=str(event.timestamp),
                )
                if verdict.verdict in _CONFLICT_VERDICTS:
                    state_service.enqueue_feedback_conflict_review(
                        belief_id=verdict.belief_id,
                        verdict=verdict.verdict,
                        evidence_quote=verdict.evidence_quote,
                        feedback_event_id=str(event.id),
                        session_id=job.session_id,
                        user_message_id=job.user_message_id,
                    )
            complete_feedback_attribution_sources(
                state_service.ledger,
                session_id=job.session_id,
                recall_tool_message_ids=job.recall_tool_message_ids,
                checkpoint_id=f"feedback_attribution:{job.turn_id}",
            )
            self._write_audit(
                "feedback_attribution_completed",
                {
                    **_job_audit_payload(job),
                    "event_ids": event_ids,
                    "verdict_count": len(verdicts),
                },
            )
        except Exception as exc:
            try:
                fail_feedback_attribution_sources(
                    CognitionStateStore(self.store).ledger,
                    session_id=job.session_id,
                    recall_tool_message_ids=job.recall_tool_message_ids,
                    error=str(exc) or type(exc).__name__,
                )
            finally:
                self._write_audit(
                    "feedback_attribution_failed",
                    {
                        **_job_audit_payload(job),
                        "event_ids": event_ids,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        finally:
            with self._lock:
                self._threads.discard(current_thread())
            self._slots.release()

    def _validated_verdicts(
        self,
        job: FeedbackAttributionJob,
    ) -> tuple[ValidatedFeedbackAttributionVerdict, ...]:
        messages = (
            *tuple(job.prompt_messages),
            _attribution_instruction_message(job),
        )
        response = traced_llm_complete(
            self.llm_provider,
            messages,
            trace_logger=self.llm_trace_logger,
            trace_metadata={
                "worker": {
                    "name": _FEEDBACK_ATTRIBUTION_WORKER,
                    "worker_id": self.worker_id,
                    "stage": BackgroundStage.FEEDBACK_ATTRIBUTION.value,
                    "session_id": job.session_id,
                    "turn_id": job.turn_id,
                    "user_message_id": job.user_message_id,
                    "recall_tool_message_ids": list(job.recall_tool_message_ids),
                }
            },
            tools=(),
            tool_choice="none",
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )
        return validate_feedback_attribution_json(
            response.content,
            FeedbackAttributionValidationContext(
                allowed_belief_ids=frozenset(
                    handle.belief_id for handle in job.recalled_beliefs
                ),
                user_message_content=job.user_message_text,
            ),
        )

    def _write_audit(self, kind: str, payload: dict[str, object]) -> None:
        try:
            CognitionStateStore(self.store).write_audit_record(kind, payload=payload)
        except Exception:
            return


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
    unique_message_ids = _stable_unique(recall_tool_message_ids)
    if not unique_message_ids:
        return ()
    claimed: list[BackgroundSourceProgress] = []
    with ledger.store.immediate_transaction() as conn:
        for message_id in unique_message_ids:
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
                return ()
        for message_id in unique_message_ids:
            source_ref = _feedback_source_ref(message_id)
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


def _attribution_instruction_message(job: FeedbackAttributionJob) -> ChatMessage:
    return {
        "role": "user",
        "content": _FEEDBACK_ATTRIBUTION_INSTRUCTION.format(
            output_schema_json=json_for_prompt(feedback_attribution_output_json_schema()),
            user_message_json=json.dumps(
                job.user_message_text,
                ensure_ascii=False,
                sort_keys=True,
            ),
            recalled_beliefs_json=json.dumps(
                [_belief_prompt_record(handle) for handle in job.recalled_beliefs],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    }


def _belief_prompt_record(handle: RecalledBeliefHandle) -> dict[str, object]:
    return {
        "belief_id": handle.belief_id,
        "content": _short_content(handle.content),
        "memory_kind": handle.memory_kind,
        "scope": handle.scope,
    }


def _short_content(content: str) -> str:
    if len(content) <= _MAX_BELIEF_PROMPT_CONTENT_CHARS:
        return content
    return content[: _MAX_BELIEF_PROMPT_CONTENT_CHARS - 3].rstrip() + "..."


def _emit_feedback_event(
    emitter: EventEmitter,
    job: FeedbackAttributionJob,
    verdict: ValidatedFeedbackAttributionVerdict,
) -> CognitiveEvent:
    return emitter.emit(
        CognitiveEventKind.RECEIVED_FEEDBACK,
        inputs=[
            Reference("belief", verdict.belief_id),
            Reference("session_message", job.user_message_id),
        ],
        rationale=f"Feedback attribution verdict: {verdict.verdict}",
        causal_parents=[EventId(job.turn_received_event_id)],
        payload={
            "turn_id": job.turn_id,
            "session_id": job.session_id,
            "feedback_kind": _FEEDBACK_KIND_BY_VERDICT[verdict.verdict],
            "matched_expected": verdict.verdict == "confirmed",
            "belief_id": verdict.belief_id,
            "verdict": verdict.verdict,
            "evidence_quote": verdict.evidence_quote,
            "user_message_id": job.user_message_id,
            "recall_tool_message_ids": list(job.recall_tool_message_ids),
        },
    )


def _job_audit_payload(job: FeedbackAttributionJob) -> dict[str, object]:
    return {
        "session_id": job.session_id,
        "turn_id": job.turn_id,
        "turn_received_event_id": job.turn_received_event_id,
        "user_message_id": job.user_message_id,
        "belief_ids": [handle.belief_id for handle in job.recalled_beliefs],
        "recall_tool_message_ids": list(job.recall_tool_message_ids),
    }


__all__ = [
    "MEMORY_RECALL_TOOL_NAME",
    "FeedbackAttributionJob",
    "RealtimeFeedbackAttributionService",
    "RecalledBeliefHandle",
    "claim_feedback_attribution_sources",
    "complete_feedback_attribution_sources",
    "fail_feedback_attribution_sources",
    "feedback_attribution_idempotency_key",
    "feedback_attribution_target_unit",
    "recalled_beliefs_for_previous_turn",
]
