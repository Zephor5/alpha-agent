"""Daemon-owned background cognition service."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace
from typing import ClassVar

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops.scheduler import (
    CheckpointStore,
    ScheduledWorker,
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
    WorkerStatus,
    YieldingCoordinator,
)
from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker
from alpha_agent.cognition.loops.workers.memory_consolidation import (
    MemoryConflictReviewWorker,
    MemoryConsolidationWorker,
)
from alpha_agent.cognition.loops.workers.memory_extraction import MemoryExtractionWorker
from alpha_agent.cognition.loops.workers.memory_summary import (
    MemorySummaryWorker,
    pending_summary_target_count,
)
from alpha_agent.cognition.models import (
    AtomicBelief,
    BeliefLifecycle,
    CognitiveEventKind,
    DerivationStage,
    Instant,
    LoopPriority,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundSourceRef,
    BackgroundStage,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.config import CognitionBackgroundConfig
from alpha_agent.llm.base import LLMProvider, LLMToolDefinitionInput
from alpha_agent.llm.tracing import LLMTraceLogger
from alpha_agent.state.models import RuntimeTrace, SessionMessage
from alpha_agent.state.store import StateStore
from alpha_agent.utils.time import utc_now, utc_now_iso

_RETRYABLE_SOURCE_STATUSES = {None, BackgroundProgressStatus.FAILED}
_ACTIVE_SOURCE_STATUSES = {
    BackgroundProgressStatus.PENDING,
    BackgroundProgressStatus.CLAIMED,
}
_INTAKE_TRACE_EVENT_TYPES = frozenset({"tool.completed", "tool.failed"})


@dataclass(frozen=True, slots=True)
class BackgroundCognitionStatus:
    """Serializable background cognition lifecycle snapshot."""

    enabled: bool
    state: str
    last_tick: str | None = None
    last_success: str | None = None
    last_error: str | None = None
    next_tick: str | None = None


class SourceIntakeWorker:
    """Record raw message/trace intake progress in the processing ledger."""

    name: ClassVar[str] = "source_intake"
    trigger: ClassVar[ScheduleTrigger] = ScheduleTrigger(
        min_interval=timedelta(seconds=0),
        max_interval=timedelta(seconds=0),
        watches=frozenset(),
        min_new_events=0,
    )
    handles_event_kinds: ClassVar[frozenset[CognitiveEventKind]] = frozenset()

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
        state_service = CognitionStateStore(projection.store)
        batch_size = max(1, int(getattr(config, "intake_batch_size", 64)))
        pending = _pending_intake_sources(state_service, limit=batch_size)
        if not pending:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
            )

        emitted = 0
        for source_ref, target_unit in pending:
            if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
                return _worker_report(
                    self.name,
                    checkpoint,
                    inspected=emitted,
                    emitted=emitted,
                    status="yielded",
                    yielded=True,
                )
            state_service.ledger.mark_source_processed(
                source_ref,
                stage=BackgroundStage.INTAKE,
                target_unit=target_unit,
                checkpoint_id=f"checkpoint:{self.name}:{source_ref.source_type}:{source_ref.source_id}",
                idempotency_key=_source_idempotency_key(
                    BackgroundStage.INTAKE,
                    source_ref,
                    target_unit,
                ),
            )
            emitted += 1

        return _worker_report(
            self.name,
            checkpoint,
            inspected=len(pending),
            emitted=emitted,
            status="ok",
        )


class _BackgroundCoordinator:
    """Coordinator wrapper that treats immediate shutdown as a yield request."""

    def __init__(self, coordinator: LoopCoordinator, immediate_stop: Event):
        self._coordinator = coordinator
        self._immediate_stop = immediate_stop
        self._deadline: float | None = None

    def acquire(self, req: LoopAcquireRequest) -> AbstractContextManager[None]:
        return self._coordinator.acquire(req)

    def yield_to_higher_priority(self) -> bool:
        if self._immediate_stop.is_set() or self.budget_exhausted():
            return True
        return self._coordinator.yield_to_higher_priority()

    def set_deadline(self, deadline: float) -> None:
        self._deadline = deadline

    def clear_deadline(self) -> None:
        self._deadline = None

    def budget_exhausted(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def remaining_seconds(self) -> float:
        if self._deadline is None:
            return float("inf")
        return max(0.0, self._deadline - time.monotonic())


class BackgroundCognitionService:
    """Automatic daemon runner for target background cognition workers."""

    def __init__(
        self,
        *,
        store: StateStore,
        config: CognitionBackgroundConfig,
        coordinator: LoopCoordinator | None = None,
        llm_provider: LLMProvider | None = None,
        tools: Sequence[LLMToolDefinitionInput] = (),
        active_session_ids: Callable[[], Sequence[str]] | None = None,
        state_service: CognitionStateStore | None = None,
        workers: Sequence[ScheduledWorker] | None = None,
        llm_trace_logger: LLMTraceLogger | None = None,
    ):
        self.store = store
        self.config = config
        self.coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
        self.llm_provider = llm_provider
        self.tools = tuple(tools)
        self.llm_trace_logger = llm_trace_logger
        self.active_session_ids = active_session_ids or (lambda: ())
        self.state_service = state_service or CognitionStateStore(store)
        self.log = SQLiteEventLog(store)
        self.projections = default_projection_registry(self.log)
        self.emitter = EventEmitter(self.log)
        self.checkpoints = CheckpointStore(store)
        self._workers = tuple(workers or self._default_workers())
        self._background_coordinator = _BackgroundCoordinator(
            self.coordinator,
            immediate_stop=Event(),
        )
        self._stop_requested = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._tick_lock = Lock()
        self._state = "disabled" if not config.enabled else "stopped"
        self._last_tick: str | None = None
        self._last_success: str | None = None
        self._last_error: str | None = None
        self._next_tick: str | None = None

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def start(self) -> None:
        """Start automatic background ticks when enabled."""

        if not self.enabled:
            self._set_status_state("disabled", next_tick=None)
            return
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested.clear()
            self._background_coordinator._immediate_stop.clear()
            self._set_status_state(
                "running",
                next_tick=_iso_after(self.config.startup_delay_seconds),
            )
            self._thread = Thread(
                target=self._run_loop,
                name="alpha-background-cognition",
                daemon=True,
            )
            self._thread.start()

    def stop(
        self,
        *,
        immediate: bool = False,
        wait: bool = True,
        timeout: float | None = None,
    ) -> None:
        """Stop future ticks and optionally wait for the service thread."""

        if not self.enabled:
            self._set_status_state("disabled", next_tick=None)
            return
        if immediate:
            self._background_coordinator._immediate_stop.set()
        self._stop_requested.set()
        self._set_status_state("stopping", next_tick=None)
        thread = self._thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if self._thread is None or not self._thread.is_alive():
            self._set_status_state("stopped", next_tick=None)

    def tick_once(self) -> list[WorkerReport]:
        """Run one bounded check over currently eligible background work."""

        if not self.enabled:
            self._set_status_state("disabled", next_tick=None)
            return []
        if not self._tick_lock.acquire(blocking=False):
            return []
        try:
            return self._tick_once_locked()
        finally:
            self._tick_lock.release()

    def status(self) -> BackgroundCognitionStatus:
        """Return a thread-safe background lifecycle snapshot."""

        with self._lock:
            return BackgroundCognitionStatus(
                enabled=self.enabled,
                state=self._state,
                last_tick=self._last_tick,
                last_success=self._last_success,
                last_error=self._last_error,
                next_tick=self._next_tick,
            )

    def _run_loop(self) -> None:
        try:
            if self._stop_requested.wait(self.config.startup_delay_seconds):
                self._set_status_state("stopped", next_tick=None)
                return
            while not self._stop_requested.is_set():
                self.tick_once()
                if self._stop_requested.is_set():
                    break
                self._set_next_tick(_iso_after(self.config.interval_seconds))
                self._stop_requested.wait(self.config.interval_seconds)
        finally:
            self._set_status_state("stopped", next_tick=None)

    def _tick_once_locked(self) -> list[WorkerReport]:
        self._set_status_state("ticking", next_tick=None)
        tick_started = utc_now_iso()
        with self._lock:
            self._last_tick = tick_started
        deadline = time.monotonic() + self.config.tick_timeout_seconds
        reports: list[WorkerReport] = []
        had_error = False
        self._background_coordinator.set_deadline(deadline)
        try:
            for worker in self._eligible_workers():
                if (
                    self._stop_requested.is_set()
                    or self._background_coordinator.budget_exhausted()
                ):
                    break
                try:
                    report = self._run_worker(worker)
                except Exception as exc:
                    had_error = True
                    self._record_failure(f"{worker.name}: {exc}")
                    continue
                reports.append(report)
                if report.new_checkpoint.last_status == "error":
                    had_error = True
                    self._record_failure(
                        f"{report.worker}: {'; '.join(report.notes) or 'worker error'}"
                    )
                if report.yielded_to_higher_priority:
                    break

            with self._lock:
                if had_error:
                    self._state = "stopped" if self._stop_requested.is_set() else "error"
                else:
                    self._last_success = utc_now_iso()
                    self._last_error = None
                    self._state = "stopped" if self._stop_requested.is_set() else "running"
            return reports
        finally:
            self._background_coordinator.clear_deadline()

    def _run_worker(self, worker: ScheduledWorker) -> WorkerReport:
        req = LoopAcquireRequest(
            loop_name=f"background:{worker.name}",
            priority=LoopPriority.CONSOLIDATION,
            max_chunk_duration=timedelta(seconds=self.config.tick_timeout_seconds),
        )
        with self._background_coordinator.acquire(req):
            checkpoint = self.checkpoints.load(worker.name)
            if self._background_coordinator.budget_exhausted():
                report = _worker_report(
                    worker.name,
                    checkpoint,
                    inspected=0,
                    emitted=0,
                    status="yielded",
                    yielded=True,
                    notes=["background tick budget exhausted before worker start"],
                )
                stamped = _stamp_report(report)
                self.checkpoints.save(stamped.new_checkpoint)
                return stamped
            report = worker.run(
                self.log,
                self.projections,
                self.emitter,
                self._background_coordinator,
                self._worker_config(),
                checkpoint,
            )
            stamped = _stamp_report(report)
            self.checkpoints.save(stamped.new_checkpoint)
            return stamped

    def _worker_config(self) -> SimpleNamespace:
        active_session_ids = tuple(str(item) for item in self.active_session_ids())
        eligible_inactive_session_ids = _eligible_inactive_session_ids(
            self.store,
            active_session_ids=frozenset(active_session_ids),
            inactivity_threshold=timedelta(
                hours=self.config.extraction.inactivity_threshold_hours
            ),
        )
        return SimpleNamespace(
            dry_run=False,
            llm_provider=self.llm_provider,
            llm_trace_logger=self.llm_trace_logger,
            tools=self.tools,
            active_session_ids=active_session_ids,
            inactive_session_ids=eligible_inactive_session_ids,
            intake_batch_size=self.config.intake.batch_size,
            consolidation_batch_size=self.config.consolidation.batch_size,
            conflict_batch_size=self.config.conflict.batch_size,
            summary_batch_size=self.config.summary.batch_size,
            summary_initial_min_beliefs=self.config.summary.initial_min_beliefs,
            summary_changed_source_min=self.config.summary.changed_source_min,
            summary_invalidated_source_min=self.config.summary.invalidated_source_min,
        )

    def _eligible_workers(self) -> list[ScheduledWorker]:
        workers_by_name = {worker.name: worker for worker in self._workers}
        eligible: list[ScheduledWorker] = []
        if _pending_intake_count(self.state_service) >= self.config.intake.min_sources:
            _append_if_present(eligible, workers_by_name, SourceIntakeWorker.name)
        if (
            _pending_extraction_count(
                self.state_service,
                inactive_session_ids=_eligible_inactive_session_ids(
                    self.store,
                    active_session_ids=frozenset(
                        str(item) for item in self.active_session_ids()
                    ),
                    inactivity_threshold=timedelta(
                        hours=self.config.extraction.inactivity_threshold_hours
                    ),
                ),
            )
            >= self.config.extraction.min_sources
        ):
            _append_if_present(eligible, workers_by_name, MemoryExtractionWorker.name)
        if (
            _pending_consolidation_count(self.state_service)
            >= self.config.consolidation.min_drafts
        ):
            _append_if_present(eligible, workers_by_name, MemoryConsolidationWorker.name)
        if _pending_conflict_count(self.state_service) >= self.config.conflict.min_conflicts:
            _append_if_present(eligible, workers_by_name, MemoryConflictReviewWorker.name)
        if (
            pending_summary_target_count(
                self.state_service,
                initial_min_beliefs=self.config.summary.initial_min_beliefs,
                changed_source_min=self.config.summary.changed_source_min,
                invalidated_source_min=self.config.summary.invalidated_source_min,
            )
            > 0
        ):
            _append_if_present(eligible, workers_by_name, MemorySummaryWorker.name)
        if _expired_belief_count(self.state_service) > 0:
            _append_if_present(eligible, workers_by_name, ArchiveExpiredWorker.name)
        return eligible

    def _record_failure(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._state = "error"
        try:
            self.state_service.write_audit_record(
                "background_cognition_error",
                payload={"error": message},
            )
        except Exception:
            return

    def _set_status_state(self, state: str, *, next_tick: str | None = None) -> None:
        with self._lock:
            self._state = state
            self._next_tick = next_tick

    def _set_next_tick(self, next_tick: str | None) -> None:
        with self._lock:
            self._next_tick = next_tick

    def _default_workers(self) -> tuple[ScheduledWorker, ...]:
        return (
            SourceIntakeWorker(),
            MemoryExtractionWorker(llm_trace_logger=self.llm_trace_logger),
            MemoryConsolidationWorker(llm_trace_logger=self.llm_trace_logger),
            MemoryConflictReviewWorker(llm_trace_logger=self.llm_trace_logger),
            MemorySummaryWorker(llm_trace_logger=self.llm_trace_logger),
            ArchiveExpiredWorker(),
        )


def _append_if_present(
    eligible: list[ScheduledWorker],
    workers_by_name: dict[str, ScheduledWorker],
    name: str,
) -> None:
    worker = workers_by_name.get(name)
    if worker is not None:
        eligible.append(worker)


def _pending_intake_sources(
    state_service: CognitionStateStore,
    *,
    limit: int,
) -> list[tuple[BackgroundSourceRef, str]]:
    pending: list[tuple[BackgroundSourceRef, str]] = []
    for source_ref, target_unit in _raw_sources(state_service.store):
        if _source_status(
            state_service,
            source_ref,
            stage=BackgroundStage.INTAKE,
            target_unit=target_unit,
        ) in _RETRYABLE_SOURCE_STATUSES:
            pending.append((source_ref, target_unit))
        if len(pending) >= limit:
            break
    return pending


def _pending_intake_count(state_service: CognitionStateStore) -> int:
    return sum(
        1
        for source_ref, target_unit in _raw_sources(state_service.store)
        if _source_status(
            state_service,
            source_ref,
            stage=BackgroundStage.INTAKE,
            target_unit=target_unit,
        )
        in _RETRYABLE_SOURCE_STATUSES
    )


def _eligible_inactive_session_ids(
    store: StateStore,
    *,
    active_session_ids: frozenset[str],
    inactivity_threshold: timedelta,
) -> tuple[str, ...]:
    cutoff = utc_now() - inactivity_threshold
    eligible: list[str] = []
    for session_id in store.list_session_ids():
        if session_id in active_session_ids:
            continue
        latest_activity = _latest_session_message_activity(store, session_id)
        if latest_activity is not None and latest_activity <= cutoff:
            eligible.append(session_id)
    return tuple(eligible)


def _latest_session_message_activity(
    store: StateStore,
    session_id: str,
) -> datetime | None:
    latest: datetime | None = None
    for message in store.list_session_messages(session_id):
        for raw in (message.created_at, message.updated_at):
            parsed = _parse_instant(raw)
            if parsed is not None and (latest is None or parsed > latest):
                latest = parsed
    return latest


def _parse_instant(raw: object | None) -> datetime | None:
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _pending_extraction_count(
    state_service: CognitionStateStore,
    *,
    inactive_session_ids: Sequence[str],
) -> int:
    count = 0
    store = state_service.store
    for session_id in inactive_session_ids:
        if _has_pending_handover(store, session_id):
            continue
        target_unit = f"session:{session_id}"
        compressed = store.find_latest_compressed_message(session_id)
        boundary_ordinal = compressed.ordinal if compressed is not None else 0
        for message in store.list_session_messages(
            session_id,
            after_ordinal=boundary_ordinal,
        ):
            if message.kind == "compressed_message":
                continue
            source_ref = BackgroundSourceRef("session_message", message.id)
            if _source_status(
                state_service,
                source_ref,
                stage=BackgroundStage.EXTRACTION,
                target_unit=target_unit,
            ) in _RETRYABLE_SOURCE_STATUSES:
                count += 1
    return count


def _has_pending_handover(store: StateStore, session_id: str) -> bool:
    pending: set[int] = set()
    for trace in store.list_runtime_traces(session_id):
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


def _pending_consolidation_count(state_service: CognitionStateStore) -> int:
    count = 0
    for belief in state_service.beliefs.list_active():
        if belief.derivation_stage != DerivationStage.BACKGROUND_EXTRACTED:
            continue
        source_ref = BackgroundSourceRef("atomic_belief", str(belief.id))
        target_unit = _target_unit_for_belief(belief)
        if _source_status(
            state_service,
            source_ref,
            stage=BackgroundStage.CONSOLIDATION,
            target_unit=target_unit,
        ) in _RETRYABLE_SOURCE_STATUSES:
            count += 1
    return count


def _pending_conflict_count(state_service: CognitionStateStore) -> int:
    return sum(
        len(state_service.ledger.list_source_windows(
            stage=BackgroundStage.CONFLICT_REVIEW,
            status=status,
        ))
        for status in (
            BackgroundProgressStatus.PENDING,
            BackgroundProgressStatus.FAILED,
        )
    )


def _expired_belief_count(state_service: CognitionStateStore) -> int:
    return sum(
        1
        for belief in state_service.beliefs.list_active()
        if belief.lifecycle == BeliefLifecycle.ACTIVE
        and _is_expired_instant(belief.validity.valid_until)
    )


def _raw_sources(store: StateStore) -> list[tuple[BackgroundSourceRef, str]]:
    sources: list[tuple[BackgroundSourceRef, str]] = []
    for session_id in store.list_session_ids():
        target_unit = f"session:{session_id}"
        for message in store.list_session_messages(session_id):
            if _intake_message(message):
                sources.append(
                    (
                        BackgroundSourceRef("session_message", message.id),
                        target_unit,
                    )
                )
        for trace in store.list_runtime_traces(session_id):
            if _intake_trace(trace):
                sources.append((BackgroundSourceRef("runtime_trace", trace.id), target_unit))
    return sources


def _intake_message(message: SessionMessage) -> bool:
    return message.kind != "compressed_message"


def _intake_trace(trace: RuntimeTrace) -> bool:
    return trace.event_type in _INTAKE_TRACE_EVENT_TYPES


def _source_status(
    state_service: CognitionStateStore,
    source_ref: BackgroundSourceRef,
    *,
    stage: BackgroundStage,
    target_unit: str,
) -> BackgroundProgressStatus | None:
    try:
        status = state_service.ledger.get_source_progress(
            source_ref,
            stage=stage,
            target_unit=target_unit,
        ).status
    except KeyError:
        return None
    if status in _ACTIVE_SOURCE_STATUSES:
        return status
    return status


def _target_unit_for_belief(belief: AtomicBelief) -> str:
    scope = belief.scope
    if str(scope) == "global":
        return "scope:global"
    about = ",".join(
        f"{ref.kind}:{ref.id}"
        for ref in sorted(belief.about, key=lambda ref: (ref.kind, ref.id))
    )
    return f"scope:{scope.value}:{about}"


def _source_idempotency_key(
    stage: BackgroundStage,
    source_ref: BackgroundSourceRef,
    target_unit: str,
) -> str:
    return f"{stage.value}:{target_unit}:{source_ref.source_type}:{source_ref.source_id}"


def _is_expired_instant(raw: object | None) -> bool:
    if raw is None:
        return False
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed < datetime.now(UTC)


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
) -> WorkerReport:
    return WorkerReport(
        worker=worker,
        inspected=inspected,
        emitted=emitted,
        notes=notes or [],
        yielded_to_higher_priority=yielded,
        new_checkpoint=WorkerCheckpoint(
            worker_name=worker,
            last_run_at=checkpoint.last_run_at,
            last_processed_event_id=checkpoint.last_processed_event_id,
            last_status=status,
            metadata=checkpoint.metadata,
        ),
    )


def _stamp_report(report: WorkerReport) -> WorkerReport:
    checkpoint = report.new_checkpoint
    return WorkerReport(
        worker=report.worker,
        inspected=report.inspected,
        emitted=report.emitted,
        notes=report.notes,
        yielded_to_higher_priority=report.yielded_to_higher_priority,
        new_checkpoint=WorkerCheckpoint(
            worker_name=checkpoint.worker_name,
            last_run_at=Instant(utc_now_iso()),
            last_processed_event_id=checkpoint.last_processed_event_id,
            last_status=checkpoint.last_status,
            metadata=checkpoint.metadata,
        ),
    )


def _iso_after(seconds: int) -> str:
    return (utc_now() + timedelta(seconds=seconds)).isoformat()


__all__ = [
    "BackgroundCognitionService",
    "BackgroundCognitionStatus",
    "SourceIntakeWorker",
]
