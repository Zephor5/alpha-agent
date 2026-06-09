"""LLM-mediated cognition-maintenance extraction from raw runtime sources."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, ClassVar

from alpha_agent.cognition.authority import CognitionSourceKind
from alpha_agent.cognition.background_llm_contract import (
    BackgroundLLMValidationContext,
    SourceWindowValidationContext,
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
from alpha_agent.runtime.chat_messages import source_message_to_chat
from alpha_agent.runtime.context_budget import stable_json
from alpha_agent.runtime.context_handover import (
    DEFAULT_MEMORY_EXTRACTION_VERSION,
    HandoverExtractionJob,
    handover_prompt_prefix_hash,
    handover_tools_schema_hash,
)
from alpha_agent.runtime.prompt_builder import default_runtime_system_message
from alpha_agent.state.models import RuntimeTrace, SessionMessage
from alpha_agent.state.store import StateStore
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
_BACKLOG_TRACE_EVENT_TYPES = frozenset({"tool.completed", "tool.failed"})
_EXTRACTION_INSTRUCTION = """Extract atomic memory candidates from the selected source window.

Return exactly one JSON object using the background cognition contract:
- operation: "create_atomic_belief"
- authority: "background_synthesized"
- rationale: short reason
- requires_confirmation: boolean
- source_span_note: optional orientation
- payload.atomic_belief_draft: id-less draft with memory_kind, scope, about, object, content

Do not include belief ids, source ids, provenance, idempotency keys, confidence, scores,
or update/supersede decisions. Project-scoped drafts may include only project_descriptor;
the program will normalize it into the project reference.

Selected source window:
{source_text}"""


@dataclass(frozen=True)
class _RuntimeTraceSource:
    trace: RuntimeTrace


@dataclass(frozen=True)
class _SourceWindowCandidate:
    source_path: str
    session_id: str
    target_unit: str
    source_refs: tuple[BackgroundSourceRef, ...]
    source_message_ids: tuple[str, ...]
    source_trace_ids: tuple[str, ...]
    ordinal_start: int | None
    ordinal_end: int | None
    prompt_prefix_messages: tuple[ChatMessage, ...]
    metadata: dict[str, Any]
    source_text: str


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
        source_batch_size: int = 12,
        extraction_version: str = DEFAULT_MEMORY_EXTRACTION_VERSION,
        worker_id: str | None = None,
    ):
        self.state_service = state_service
        self.llm_provider = llm_provider
        self.tools = tuple(tools or ())
        self.active_session_ids = frozenset(active_session_ids)
        self.inactive_session_ids = frozenset(inactive_session_ids)
        self.source_batch_size = max(1, int(source_batch_size))
        self.extraction_version = extraction_version
        self.worker_id = worker_id or self.name

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
            source_batch_size=self.source_batch_size,
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
                source_batch_size=self.source_batch_size,
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
        source_batch_size = int(getattr(config, "source_batch_size", self.source_batch_size))
        return self._run_with(
            state_service=state_service,
            llm_provider=provider,
            tools=tools,
            checkpoint=checkpoint,
            coordinator=coordinator,
            active_session_ids=active_session_ids,
            inactive_session_ids=inactive_session_ids,
            source_batch_size=max(1, source_batch_size),
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
        source_batch_size: int,
    ) -> WorkerReport:
        candidate = _next_source_window_candidate(
            state_service,
            tools=tools,
            active_session_ids=active_session_ids,
            inactive_session_ids=inactive_session_ids,
            source_batch_size=source_batch_size,
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
        response = llm_provider.complete(
            [
                *candidate.prompt_prefix_messages,
                _extraction_instruction_message(candidate.source_text),
            ],
            tools=list(tools) if tools else None,
            tool_choice=_tool_choice_for_extraction(tools),
            response_format=JSON_OBJECT_RESPONSE_FORMAT,
        )
        context = _validation_context(
            state_service.store,
            window=window,
            candidate=candidate,
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
    source_batch_size: int,
    extraction_version: str,
) -> _SourceWindowCandidate | None:
    return _inactive_backlog_candidate(
        state_service,
        active_session_ids=active_session_ids,
        inactive_session_ids=inactive_session_ids,
        source_batch_size=source_batch_size,
        extraction_version=extraction_version,
        tools=tools,
    )


def _compact_job_candidate(
    state_service: CognitionStateStore,
    job: HandoverExtractionJob,
    *,
    tools: Sequence[LLMToolDefinitionInput],
    source_batch_size: int,
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

    source_records = _covered_source_records(
        {"covered_source_message_refs": list(job.covered_source_message_refs)}
    )
    source_refs = tuple(
        BackgroundSourceRef("session_message", record["source_id"])
        for record in source_records
    )
    selected_refs = _retryable_refs(
        state_service,
        source_refs,
        target_unit=f"session:{job.session_id}",
    )[:source_batch_size]
    if not selected_refs:
        return None
    selected_ids = {ref.source_id for ref in selected_refs}
    selected_records = [
        record for record in source_records if record["source_id"] in selected_ids
    ]
    source_messages = store.list_session_messages_by_ids(
        [ref.source_id for ref in selected_refs if ref.source_type == "session_message"]
    )
    source_text = _render_source_text(messages=source_messages, traces=())
    ordinals = [int(record["ordinal"]) for record in selected_records]
    return _SourceWindowCandidate(
        source_path=_COMPACT_SOURCE_PATH,
        session_id=job.session_id,
        target_unit=f"session:{job.session_id}",
        source_refs=tuple(selected_refs),
        source_message_ids=tuple(ref.source_id for ref in selected_refs),
        source_trace_ids=(),
        ordinal_start=min(ordinals) if ordinals else None,
        ordinal_end=max(ordinals) if ordinals else None,
        prompt_prefix_messages=prefix_messages,
        source_text=source_text,
        metadata={
            "source_path": _COMPACT_SOURCE_PATH,
            "session_id": job.session_id,
            "ordinal_start": min(ordinals) if ordinals else None,
            "ordinal_end": max(ordinals) if ordinals else None,
            "source_message_ids": [ref.source_id for ref in selected_refs],
            "source_trace_ids": [],
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
    source_batch_size: int,
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
        messages = [
            message
            for message in store.list_session_messages(session_id)
            if message.kind != "compressed_message"
        ]
        message_refs = tuple(
            BackgroundSourceRef("session_message", message.id) for message in messages
        )
        trace_sources = _backlog_runtime_trace_sources(store, session_id)
        trace_refs = tuple(
            BackgroundSourceRef("runtime_trace", item.trace.id) for item in trace_sources
        )
        source_refs = _retryable_refs(
            state_service,
            (*message_refs, *trace_refs),
            target_unit=target_unit,
        )[:source_batch_size]
        if not source_refs:
            continue
        selected_message_ids = [
            ref.source_id for ref in source_refs if ref.source_type == "session_message"
        ]
        selected_trace_ids = [
            ref.source_id for ref in source_refs if ref.source_type == "runtime_trace"
        ]
        selected_messages = store.list_session_messages_by_ids(selected_message_ids)
        selected_traces = _runtime_traces_by_ids(store, selected_trace_ids)
        ordinals = [message.ordinal for message in selected_messages]
        source_text = _render_source_text(messages=selected_messages, traces=selected_traces)
        prompt_prefix_messages = [
            default_runtime_system_message(),
            *[source_message_to_chat(message) for message in selected_messages],
        ]
        return _SourceWindowCandidate(
            source_path=_BACKLOG_SOURCE_PATH,
            session_id=session_id,
            target_unit=target_unit,
            source_refs=tuple(source_refs),
            source_message_ids=tuple(selected_message_ids),
            source_trace_ids=tuple(selected_trace_ids),
            ordinal_start=min(ordinals) if ordinals else None,
            ordinal_end=max(ordinals) if ordinals else None,
            prompt_prefix_messages=tuple(prompt_prefix_messages),
            source_text=source_text,
            metadata={
                "source_path": _BACKLOG_SOURCE_PATH,
                "session_id": session_id,
                "ordinal_start": min(ordinals) if ordinals else None,
                "ordinal_end": max(ordinals) if ordinals else None,
                "source_message_ids": selected_message_ids,
                "source_trace_ids": selected_trace_ids,
                "prompt_prefix_hash": handover_prompt_prefix_hash(prompt_prefix_messages),
                "tools_schema_hash": handover_tools_schema_hash(tools),
                "extraction_version": extraction_version,
            },
        )
    return None


def _covered_source_records(metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_records = metadata.get("covered_source_message_refs")
    if isinstance(raw_records, list):
        records: list[dict[str, Any]] = []
        for item in raw_records:
            if not isinstance(item, Mapping):
                continue
            source_id = item.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                continue
            if item.get("kind") == "compressed_message":
                continue
            ordinal = item.get("ordinal")
            records.append(
                {
                    "source_id": source_id,
                    "ordinal": int(ordinal) if isinstance(ordinal, int) else 0,
                }
            )
        return records
    raw_ids = metadata.get("covered_source_message_ids")
    if not isinstance(raw_ids, list):
        return []
    return [
        {"source_id": item, "ordinal": 0}
        for item in raw_ids
        if isinstance(item, str) and item.strip()
    ]


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


def _backlog_runtime_trace_sources(
    store: StateStore,
    session_id: str,
) -> tuple[_RuntimeTraceSource, ...]:
    return tuple(
        _RuntimeTraceSource(trace)
        for trace in store.list_runtime_traces(session_id)
        if trace.event_type in _BACKLOG_TRACE_EVENT_TYPES
    )


def _runtime_traces_by_ids(store: StateStore, trace_ids: Sequence[str]) -> list[RuntimeTrace]:
    if not trace_ids:
        return []
    placeholders = ",".join("?" for _ in trace_ids)
    with store.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT *
            FROM runtime_traces
            WHERE id IN ({placeholders})
            """,
            list(trace_ids),
        ).fetchall()
    by_id = {
        str(row["id"]): RuntimeTrace(
            id=row["id"],
            session_id=row["session_id"],
            event_type=row["event_type"],
            content=row["content"],
            timestamp=row["timestamp"],
            metadata=_loads_dict(row["metadata"]),
        )
        for row in rows
    }
    return [by_id[trace_id] for trace_id in trace_ids if trace_id in by_id]


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


def _render_source_text(
    *,
    messages: Sequence[SessionMessage],
    traces: Sequence[RuntimeTrace],
) -> str:
    parts: list[str] = []
    for message in messages:
        parts.append(
            f"[session_message:{message.ordinal}:{message.id}:{message.kind}] "
            f"{message.raw_content}"
        )
    for trace in traces:
        parts.append(f"[runtime_trace:{trace.id}:{trace.event_type}] {trace.content}")
    return "\n".join(parts)


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
            source_text=candidate.source_text,
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


def _extraction_instruction_message(source_text: str) -> ChatMessage:
    return {
        "role": "user",
        "content": _EXTRACTION_INSTRUCTION.format(source_text=source_text),
    }


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


def _loads_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


__all__ = ["MemoryExtractionWorker"]
