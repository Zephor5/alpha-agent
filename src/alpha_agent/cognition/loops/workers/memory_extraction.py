"""LLM-mediated cognition-maintenance extraction from raw runtime sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar

from alpha_agent.cognition.authority import CognitionSourceKind
from alpha_agent.cognition.background_llm_contract import (
    BackgroundLLMValidationContext,
    SourceWindowValidationContext,
    extraction_output_json_schema,
)
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.loops.scheduler import (
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    WorkerStatus,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers._common import (
    background_llm_trace_metadata,
    json_for_prompt,
)
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    DerivationStage,
    Instant,
)
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundSourceRef,
    BackgroundSourceWindow,
    BackgroundStage,
    BackgroundStageRunStatus,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.llm.base import (
    JSON_OBJECT_RESPONSE_FORMAT,
    ChatMessage,
    LLMProvider,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.llm.tracing import LLMTraceLogger, traced_llm_complete
from alpha_agent.runtime.chat_messages import session_message_to_chat, source_message_to_chat
from alpha_agent.runtime.context_budget import stable_json
from alpha_agent.runtime.context_handover import (
    DEFAULT_MEMORY_EXTRACTION_VERSION,
    HandoverExtractionJob,
    handover_prompt_prefix_hash,
    handover_tools_schema_hash,
)
from alpha_agent.runtime.prompt_builder import default_runtime_system_message
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore
from alpha_agent.utils.system_reminder import SYSTEM_REMINDER_OPEN, SYSTEM_REMINDER_PLACEHOLDER
from alpha_agent.utils.time import utc_now_iso

_COMPACT_SOURCE_PATH = "compact_direct"
_BACKLOG_SOURCE_PATH = "inactive_backlog"
_RETRYABLE_SOURCE_STATUSES = {
    None,
    BackgroundProgressStatus.FAILED,
}
_ACTIVE_WINDOW_STATUSES = {
    BackgroundProgressStatus.PENDING,
    BackgroundProgressStatus.CLAIMED,
}
_EXTRACTION_INSTRUCTION = """Extract atomic memory candidates from the previous messages.

Return only one JSON object. Do not return markdown, code fences, top-level arrays, or
commentary. Put candidates in payload.atomic_belief_drafts, using an empty array when
nothing should be extracted. The output must validate against this JSON Schema:
{output_schema_json}

Allowed about references for this session:
{allowed_about_refs_json}

Scope and reference rules:
- For scope "global", set about to [].
- For scope "counterpart", use exactly one allowed reference with kind "counterpart".
- For scope "self", use exactly one allowed reference with kind "subject" or "self".
- For scope "session", use exactly one allowed reference with kind "session".
- For scope "project", set about to [] and include project_descriptor as a resolvable
  string or object; do not invent project ids.

Content rules:
- Each content value must be directly supported by the previous messages.
- Messages wrapped in {system_reminder_placeholder} are session context,
  not new user evidence. Use them only to interpret ordinary user, assistant, and
  tool messages. Do not extract a new memory whose only support is a
  {system_reminder_open} message.
- object should be the short subject of content; omit it if content is already short.
- requires_confirmation should be true when a memory is plausible but not safe to accept
  without human review.
- Do not include belief ids, source ids, provenance, idempotency keys, confidence, scores,
  numeric strength fields, or update/supersede decisions."""


@dataclass(frozen=True)
class _SourceWindowCandidate:
    source_path: str
    session_id: str
    target_unit: str
    source_refs: tuple[BackgroundSourceRef, ...]
    ordinal_start: int | None
    ordinal_end: int | None
    prompt_prefix_messages: tuple[ChatMessage, ...]
    metadata: dict[str, Any]


class _NeverYieldCoordinator:
    def yield_to_higher_priority(self) -> bool:
        return False

    def budget_exhausted(self) -> bool:
        return False

    def remaining_seconds(self) -> float:
        return float("inf")


class MemoryExtractionWorker:
    """Select raw source windows and ask an LLM for id-less atomic belief drafts."""

    name: ClassVar[str] = "memory_extraction"
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(seconds=0),
        max_interval=timedelta(seconds=0),
        watches=frozenset(),
        min_new_events=0,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset()

    def __init__(
        self,
        state_service: CognitionStateStore | None = None,
        llm_provider: LLMProvider | None = None,
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        active_session_ids: Iterable[str] = (),
        inactive_session_ids: Iterable[str] = (),
        extraction_version: str = DEFAULT_MEMORY_EXTRACTION_VERSION,
        worker_id: str | None = None,
        llm_trace_logger: LLMTraceLogger | None = None,
    ):
        self.state_service = state_service
        self.llm_provider = llm_provider
        self.tools = tuple(tools or ())
        self.active_session_ids = frozenset(active_session_ids)
        self.inactive_session_ids = frozenset(inactive_session_ids)
        self.extraction_version = extraction_version
        self.worker_id = worker_id or self.name
        self.llm_trace_logger = llm_trace_logger

    def run_once(
        self,
        *,
        checkpoint: WorkerCheckpoint | None = None,
        coordinator: YieldingCoordinator | None = None,
    ) -> WorkerReport:
        if self.state_service is None:
            raise ValueError("MemoryExtractionWorker.run_once requires state_service")
        if self.llm_provider is None:
            raise ValueError("MemoryExtractionWorker.run_once requires llm_provider")
        return self._run_with(
            state_service=self.state_service,
            llm_provider=self.llm_provider,
            tools=self.tools,
            checkpoint=checkpoint or WorkerCheckpoint(worker_name=self.name),
            coordinator=coordinator or _NeverYieldCoordinator(),
            active_session_ids=self.active_session_ids,
            inactive_session_ids=self.inactive_session_ids,
            llm_trace_logger=self.llm_trace_logger,
        )

    def run_compact_job(
        self,
        job: HandoverExtractionJob,
        *,
        checkpoint: WorkerCheckpoint | None = None,
        coordinator: YieldingCoordinator | None = None,
    ) -> WorkerReport:
        """Process one explicit handover compact extraction job."""

        if self.state_service is None:
            raise ValueError("MemoryExtractionWorker.run_compact_job requires state_service")
        if self.llm_provider is None:
            raise ValueError("MemoryExtractionWorker.run_compact_job requires llm_provider")
        checkpoint = checkpoint or WorkerCheckpoint(worker_name=self.name)
        try:
            candidate = _compact_job_candidate(
                self.state_service,
                job,
                tools=self.tools,
                extraction_version=self.extraction_version,
            )
        except ValueError as exc:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="error",
                notes=[str(exc)],
            )
        if candidate is None:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
            )
        return _run_candidate(
            worker_name=self.name,
            worker_id=self.worker_id,
            state_service=self.state_service,
            llm_provider=self.llm_provider,
            tools=self.tools,
            checkpoint=checkpoint,
            coordinator=coordinator or _NeverYieldCoordinator(),
            candidate=candidate,
            llm_trace_logger=self.llm_trace_logger,
        )

    def run(
        self,
        log: EventLog,
        projections: ProjectionRegistry,
        emitter: EventEmitter,
        coordinator: YieldingCoordinator,
        config: object,
        checkpoint: WorkerCheckpoint,
    ) -> WorkerReport:
        del log, emitter
        projection = projections.get_typed(BeliefProjection)
        state_service = self.state_service or CognitionStateStore(projection.store)
        provider = self.llm_provider or getattr(config, "llm_provider", None)
        if provider is None:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
                notes=["memory extraction skipped: no LLM provider configured"],
            )
        tools = tuple(getattr(config, "tools", self.tools) or ())
        active_session_ids = frozenset(
            getattr(config, "active_session_ids", self.active_session_ids) or ()
        )
        inactive_session_ids = frozenset(
            getattr(config, "inactive_session_ids", self.inactive_session_ids) or ()
        )
        llm_trace_logger = (
            getattr(config, "llm_trace_logger", self.llm_trace_logger)
            or self.llm_trace_logger
        )
        return self._run_with(
            state_service=state_service,
            llm_provider=provider,
            tools=tools,
            checkpoint=checkpoint,
            coordinator=coordinator,
            active_session_ids=active_session_ids,
            inactive_session_ids=inactive_session_ids,
            llm_trace_logger=llm_trace_logger,
        )

    def _run_with(
        self,
        *,
        state_service: CognitionStateStore,
        llm_provider: LLMProvider,
        tools: Sequence[LLMToolDefinitionInput],
        checkpoint: WorkerCheckpoint,
        coordinator: YieldingCoordinator,
        active_session_ids: frozenset[str],
        inactive_session_ids: frozenset[str],
        llm_trace_logger: LLMTraceLogger | None,
    ) -> WorkerReport:
        candidate = _next_source_window_candidate(
            state_service,
            tools=tools,
            active_session_ids=active_session_ids,
            inactive_session_ids=inactive_session_ids,
            extraction_version=self.extraction_version,
        )
        if candidate is None:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
            )
        return _run_candidate(
            worker_name=self.name,
            worker_id=self.worker_id,
            state_service=state_service,
            llm_provider=llm_provider,
            tools=tools,
            checkpoint=checkpoint,
            coordinator=coordinator,
            candidate=candidate,
            llm_trace_logger=llm_trace_logger,
        )


def _run_candidate(
    *,
    worker_name: str,
    worker_id: str,
    state_service: CognitionStateStore,
    llm_provider: LLMProvider,
    tools: Sequence[LLMToolDefinitionInput],
    checkpoint: WorkerCheckpoint,
    coordinator: YieldingCoordinator,
    candidate: _SourceWindowCandidate,
    llm_trace_logger: LLMTraceLogger | None,
) -> WorkerReport:
    if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
        return _worker_report(
            worker_name,
            checkpoint,
            inspected=len(candidate.source_refs),
            emitted=0,
            status="yielded",
            yielded=True,
            metadata={"last_session_id": candidate.session_id},
        )

    window = state_service.ledger.create_source_window(
        stage=BackgroundStage.EXTRACTION,
        target_unit=candidate.target_unit,
        source_refs=candidate.source_refs,
        idempotency_key=_window_idempotency_key(candidate),
        metadata=candidate.metadata,
    )
    if window.status == BackgroundProgressStatus.PROCESSED:
        return _worker_report(
            worker_name,
            checkpoint,
            inspected=len(candidate.source_refs),
            emitted=0,
            status="skipped_no_backlog",
            metadata={"last_window_id": window.window_id},
        )

    window = _claim_window_and_sources(
        state_service,
        window,
        claimed_by=worker_id,
    )
    run = state_service.ledger.start_stage_run(
        worker_id=worker_id,
        stage=BackgroundStage.EXTRACTION,
        target_unit=candidate.target_unit,
        window_id=window.window_id,
        input_refs=candidate.source_refs,
    )
    if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
        message = "memory extraction yielded before LLM call"
        _mark_failed_if_needed(
            state_service,
            window=window,
            run_id=run.run_id,
            error=message,
        )
        return _worker_report(
            worker_name,
            checkpoint,
            inspected=len(candidate.source_refs),
            emitted=0,
            status="yielded",
            yielded=True,
            notes=[message],
            metadata={"last_window_id": window.window_id},
        )
    try:
        context = _validation_context(
            state_service.store,
            window=window,
            candidate=candidate,
        )
        response = traced_llm_complete(
            llm_provider,
            [
                *candidate.prompt_prefix_messages,
                _extraction_instruction_message(context=context),
            ],
            trace_logger=llm_trace_logger,
            trace_metadata=background_llm_trace_metadata(
                worker_name=worker_name,
                worker_id=worker_id,
                stage=BackgroundStage.EXTRACTION,
                window=window,
                run_id=run.run_id,
                session_id=candidate.session_id,
            ),
            tools=list(tools) if tools else None,
            tool_choice=_tool_choice_for_extraction(tools),
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )
        written = state_service.accept_background_llm_json(
            response.content,
            context,
            window_id=window.window_id,
            run_id=run.run_id,
            checkpoint_id=f"checkpoint:{worker_name}:{window.window_id}",
        )
    except Exception as exc:
        _mark_failed_if_needed(
            state_service,
            window=window,
            run_id=run.run_id,
            error=str(exc),
        )
        return _worker_report(
            worker_name,
            checkpoint,
            inspected=len(candidate.source_refs),
            emitted=0,
            status="error",
            notes=[str(exc)],
            metadata={"last_window_id": window.window_id},
        )

    return _worker_report(
        worker_name,
        checkpoint,
        inspected=len(candidate.source_refs),
        emitted=len(written),
        status="ok",
        metadata={"last_window_id": window.window_id},
    )


def _next_source_window_candidate(
    state_service: CognitionStateStore,
    *,
    tools: Sequence[LLMToolDefinitionInput],
    active_session_ids: frozenset[str],
    inactive_session_ids: frozenset[str],
    extraction_version: str,
) -> _SourceWindowCandidate | None:
    return _inactive_backlog_candidate(
        state_service,
        active_session_ids=active_session_ids,
        inactive_session_ids=inactive_session_ids,
        extraction_version=extraction_version,
        tools=tools,
    )


def _compact_job_candidate(
    state_service: CognitionStateStore,
    job: HandoverExtractionJob,
    *,
    tools: Sequence[LLMToolDefinitionInput],
    extraction_version: str,
) -> _SourceWindowCandidate | None:
    store = state_service.store
    if job.extraction_version != extraction_version:
        raise ValueError("extraction version mismatch")
    prefix_messages = tuple(job.prompt_prefix_messages)
    prompt_hash = handover_prompt_prefix_hash(prefix_messages)
    if job.prompt_prefix_hash != prompt_hash:
        raise ValueError("prompt prefix hash mismatch")
    tools_hash = handover_tools_schema_hash(tools)
    if job.tools_schema_hash != tools_hash:
        raise ValueError("tools schema hash mismatch")

    source_messages = _compact_job_source_messages(store, job)
    extractable_messages = _extractable_source_messages(source_messages)
    source_refs = tuple(
        BackgroundSourceRef("session_message", message.id) for message in extractable_messages
    )
    selected_refs = _retryable_refs(
        state_service,
        source_refs,
        target_unit=f"session:{job.session_id}",
    )
    if not selected_refs:
        return None
    selected_ids = {ref.source_id for ref in selected_refs}
    selected_messages = [
        message for message in extractable_messages if message.id in selected_ids
    ]
    ordinals = [message.ordinal for message in selected_messages]
    return _SourceWindowCandidate(
        source_path=_COMPACT_SOURCE_PATH,
        session_id=job.session_id,
        target_unit=f"session:{job.session_id}",
        source_refs=tuple(selected_refs),
        ordinal_start=min(ordinals) if ordinals else None,
        ordinal_end=max(ordinals) if ordinals else None,
        prompt_prefix_messages=prefix_messages,
        metadata={
            "source_path": _COMPACT_SOURCE_PATH,
            "session_id": job.session_id,
            "ordinal_start": min(ordinals) if ordinals else None,
            "ordinal_end": max(ordinals) if ordinals else None,
            "source_message_ids": [ref.source_id for ref in selected_refs],
            "context_reminder_message_ids": [
                message.id for message in source_messages if _is_system_reminder(message)
            ],
            "provider": job.provider,
            "model": job.model,
            "prompt_prefix_hash": job.prompt_prefix_hash,
            "direct_prompt_prefix_hash": prompt_hash,
            "prompt_prefix_hash_matches": True,
            "tools_schema_hash": job.tools_schema_hash,
            "direct_tools_schema_hash": tools_hash,
            "tools_schema_hash_matches": True,
            "compression_trace_id": job.compression_trace_id,
            "compressed_message_id": job.compressed_message_id,
            "extraction_version": extraction_version,
        },
    )


def _inactive_backlog_candidate(
    state_service: CognitionStateStore,
    *,
    active_session_ids: frozenset[str],
    inactive_session_ids: frozenset[str],
    extraction_version: str,
    tools: Sequence[LLMToolDefinitionInput],
) -> _SourceWindowCandidate | None:
    if not inactive_session_ids:
        return None
    store = state_service.store
    for session_id in sorted(inactive_session_ids):
        if session_id in active_session_ids:
            continue
        target_unit = f"session:{session_id}"
        if _has_pending_handover(store, session_id):
            continue
        if _has_active_extraction_window(state_service, target_unit=target_unit):
            continue
        compressed = store.find_latest_compressed_message(session_id)
        boundary_ordinal = compressed.ordinal if compressed is not None else 0
        messages = [
            message
            for message in store.list_session_messages(
                session_id,
                after_ordinal=boundary_ordinal,
            )
            if message.kind != "compressed_message"
        ]
        message_refs = tuple(
            BackgroundSourceRef("session_message", message.id) for message in messages
            if not _is_system_reminder(message)
        )
        source_refs = _retryable_refs(
            state_service,
            message_refs,
            target_unit=target_unit,
        )
        if not source_refs:
            continue
        selected_message_ids = [
            ref.source_id for ref in source_refs if ref.source_type == "session_message"
        ]
        selected_ids = set(selected_message_ids)
        selected_messages = [
            message for message in messages if message.id in selected_ids
        ]
        prompt_source_messages = [
            message
            for message in messages
            if message.id in selected_ids or _is_system_reminder(message)
        ]
        ordinals = [message.ordinal for message in selected_messages]
        prompt_prefix_messages = [
            default_runtime_system_message(),
            *([session_message_to_chat(compressed)] if compressed is not None else []),
            *[source_message_to_chat(message) for message in prompt_source_messages],
        ]
        return _SourceWindowCandidate(
            source_path=_BACKLOG_SOURCE_PATH,
            session_id=session_id,
            target_unit=target_unit,
            source_refs=tuple(source_refs),
            ordinal_start=min(ordinals) if ordinals else None,
            ordinal_end=max(ordinals) if ordinals else None,
            prompt_prefix_messages=tuple(prompt_prefix_messages),
            metadata={
                "source_path": _BACKLOG_SOURCE_PATH,
                "session_id": session_id,
                "ordinal_start": min(ordinals) if ordinals else None,
                "ordinal_end": max(ordinals) if ordinals else None,
                "source_message_ids": selected_message_ids,
                "context_reminder_message_ids": [
                    message.id for message in prompt_source_messages if _is_system_reminder(message)
                ],
                "compressed_message_id": compressed.id if compressed is not None else None,
                "boundary_ordinal": boundary_ordinal,
                "prompt_prefix_hash": handover_prompt_prefix_hash(prompt_prefix_messages),
                "tools_schema_hash": handover_tools_schema_hash(tools),
                "extraction_version": extraction_version,
            },
        )
    return None


def _compact_job_source_messages(
    store: StateStore,
    job: HandoverExtractionJob,
) -> list[SessionMessage]:
    previous_compressed = store.find_latest_compressed_message(
        job.session_id,
        before_ordinal=job.compressed_message_ordinal,
    )
    boundary_ordinal = previous_compressed.ordinal if previous_compressed is not None else 0
    return [
        message
        for message in store.list_session_messages(
            job.session_id,
            after_ordinal=boundary_ordinal,
        )
        if message.kind != "compressed_message"
        and message.ordinal <= job.compression_point_ordinal
    ]


def _extractable_source_messages(
    messages: Sequence[SessionMessage],
) -> list[SessionMessage]:
    return [message for message in messages if not _is_system_reminder(message)]


def _is_system_reminder(message: SessionMessage) -> bool:
    return message.kind == "system_reminder"


def _retryable_refs(
    state_service: CognitionStateStore,
    source_refs: Sequence[BackgroundSourceRef],
    *,
    target_unit: str,
) -> list[BackgroundSourceRef]:
    selected: list[BackgroundSourceRef] = []
    for ref in source_refs:
        status = _source_status(state_service, ref, target_unit=target_unit)
        if status in _RETRYABLE_SOURCE_STATUSES:
            selected.append(ref)
    return selected


def _source_status(
    state_service: CognitionStateStore,
    source_ref: BackgroundSourceRef,
    *,
    target_unit: str,
) -> BackgroundProgressStatus | None:
    try:
        return state_service.ledger.get_source_progress(
            source_ref,
            stage=BackgroundStage.EXTRACTION,
            target_unit=target_unit,
        ).status
    except KeyError:
        return None


def _has_pending_handover(store: StateStore, session_id: str) -> bool:
    traces = store.list_runtime_traces(session_id)
    pending: set[int] = set()
    for trace in traces:
        if not trace.event_type.startswith("handover_compression."):
            continue
        point = _optional_int(trace.metadata.get("compression_point_ordinal"))
        if point is None:
            continue
        if trace.event_type == "handover_compression.started":
            pending.add(point)
        elif trace.event_type in {"handover_compression.completed", "handover_compression.failed"}:
            pending.discard(point)
    return bool(pending)


def _has_active_extraction_window(
    state_service: CognitionStateStore,
    *,
    target_unit: str,
) -> bool:
    for status in _ACTIVE_WINDOW_STATUSES:
        if state_service.ledger.list_source_windows(
            stage=BackgroundStage.EXTRACTION,
            target_unit=target_unit,
            status=status,
        ):
            return True
    return False


def _window_idempotency_key(candidate: _SourceWindowCandidate) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "stage": BackgroundStage.EXTRACTION.value,
                "source_path": candidate.source_path,
                "session_id": candidate.session_id,
                "ordinal_start": candidate.ordinal_start,
                "ordinal_end": candidate.ordinal_end,
                "source_refs": [ref.to_record() for ref in candidate.source_refs],
                "prompt_prefix_hash": candidate.metadata.get("prompt_prefix_hash"),
                "tools_schema_hash": candidate.metadata.get("tools_schema_hash"),
                "extraction_version": candidate.metadata.get("extraction_version"),
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"extraction:{candidate.source_path}:{digest[:32]}"


def _claim_window_and_sources(
    state_service: CognitionStateStore,
    window: BackgroundSourceWindow,
    *,
    claimed_by: str,
) -> BackgroundSourceWindow:
    with state_service.store.immediate_transaction() as conn:
        for source_ref in window.source_refs:
            state_service.ledger.mark_source_pending(
                source_ref,
                stage=BackgroundStage.EXTRACTION,
                target_unit=window.target_unit,
                idempotency_key=window.idempotency_key,
                conn=conn,
            )
            state_service.ledger.claim_source(
                source_ref,
                stage=BackgroundStage.EXTRACTION,
                target_unit=window.target_unit,
                claimed_by=claimed_by,
                idempotency_key=window.idempotency_key,
                conn=conn,
            )
        return state_service.ledger.claim_source_window(
            window.window_id,
            claimed_by=claimed_by,
            conn=conn,
        )


def _validation_context(
    store: StateStore,
    *,
    window: BackgroundSourceWindow,
    candidate: _SourceWindowCandidate,
) -> BackgroundLLMValidationContext:
    return BackgroundLLMValidationContext(
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        source_window=SourceWindowValidationContext(
            window_id=window.window_id,
            stage=BackgroundStage.EXTRACTION,
            target_unit=window.target_unit,
            session_id=candidate.session_id,
            ordinal_start=candidate.ordinal_start,
            ordinal_end=candidate.ordinal_end,
            source_refs=window.source_refs,
        ),
        allowed_about_refs=_allowed_about_refs(store, candidate.session_id),
        derivation_stage=DerivationStage.BACKGROUND_EXTRACTED,
    )


def _allowed_about_refs(
    store: StateStore,
    session_id: str,
) -> frozenset[tuple[str, str]]:
    refs = {
        ("session", session_id),
        ("subject", "subject:self"),
        ("self", "subject:self"),
    }
    counterpart = store.get_session_counterpart(session_id)
    if counterpart is not None:
        refs.add(("counterpart", counterpart.counterpart_id))
    return frozenset(refs)


def _mark_failed_if_needed(
    state_service: CognitionStateStore,
    *,
    window: BackgroundSourceWindow,
    run_id: str,
    error: str,
) -> None:
    refreshed_window = state_service.ledger.get_source_window(window.window_id)
    run = state_service.ledger.get_stage_run(run_id)
    if (
        refreshed_window.status == BackgroundProgressStatus.FAILED
        and run.status == BackgroundStageRunStatus.FAILED
    ):
        return
    with state_service.store.immediate_transaction() as conn:
        for source_ref in window.source_refs:
            state_service.ledger.mark_source_failed(
                source_ref,
                stage=BackgroundStage.EXTRACTION,
                target_unit=window.target_unit,
                error=error,
                idempotency_key=window.idempotency_key,
                conn=conn,
            )
        state_service.ledger.mark_source_window_failed(window.window_id, error=error, conn=conn)
        state_service.ledger.finish_stage_run(
            run_id,
            status=BackgroundStageRunStatus.FAILED,
            error=error,
            conn=conn,
        )


def _extraction_instruction_message(
    *,
    context: BackgroundLLMValidationContext,
) -> ChatMessage:
    return {
        "role": "user",
        "content": _EXTRACTION_INSTRUCTION.format(
            output_schema_json=_extraction_output_schema_json(),
            allowed_about_refs_json=_allowed_about_refs_json(context),
            system_reminder_placeholder=SYSTEM_REMINDER_PLACEHOLDER,
            system_reminder_open=SYSTEM_REMINDER_OPEN,
        ),
    }


def _extraction_output_schema_json() -> str:
    return json_for_prompt(extraction_output_json_schema())


def _allowed_about_refs_json(context: BackgroundLLMValidationContext) -> str:
    refs = [
        {"kind": kind, "id": ref_id}
        for kind, ref_id in sorted(context.allowed_about_refs or frozenset())
    ]
    return json.dumps(refs, ensure_ascii=False, sort_keys=True)


def _tool_choice_for_extraction(
    tools: Sequence[LLMToolDefinitionInput],
) -> LLMToolChoice | None:
    return "none" if tools else None


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None


def _worker_report(
    worker: str,
    checkpoint: WorkerCheckpoint,
    *,
    inspected: int,
    emitted: int,
    status: WorkerStatus,
    notes: list[str] | None = None,
    yielded: bool = False,
    metadata: dict[str, object] | None = None,
) -> WorkerReport:
    return WorkerReport(
        worker=worker,
        inspected=inspected,
        emitted=emitted,
        notes=notes or [],
        yielded_to_higher_priority=yielded,
        new_checkpoint=WorkerCheckpoint(
            worker_name=worker,
            last_run_at=Instant(utc_now_iso()),
            last_processed_event_id=checkpoint.last_processed_event_id,
            last_status=status,
            metadata=metadata if metadata is not None else checkpoint.metadata,
        ),
    )


__all__ = ["MemoryExtractionWorker"]
