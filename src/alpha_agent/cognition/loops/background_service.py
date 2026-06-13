"""Daemon-owned background cognition service."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, RLock, Thread
from types import SimpleNamespace

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.coordinator import LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops.scheduler import (
    CheckpointStore,
    ScheduledWorker,
    WorkerCheckpoint,
    WorkerReport,
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
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.config import CognitionBackgroundConfig
from alpha_agent.llm.base import LLMProvider, LLMToolDefinitionInput
from alpha_agent.llm.tracing import LLMTraceLogger
from alpha_agent.state.store import StateStore
from alpha_agent.utils.time import utc_now, utc_now_iso

_RETRYABLE_SOURCE_STATUSES = {None, BackgroundProgressStatus.FAILED}
_ACTIVE_SOURCE_STATUSES = {
    BackgroundProgressStatus.PENDING,
    BackgroundProgressStatus.CLAIMED,
}
_BACKGROUND_CHUNK_TTL = timedelta(seconds=60)


@dataclass(frozen=True, slots=True)
class BackgroundCognitionStatus:
    """Serializable background cognition lifecycle snapshot."""

    enabled: bool
    state: str
    last_tick: str | None = None
    last_success: str | None = None
    last_error: str | None = None
    next_tick: str | None = None


@dataclass(frozen=True, slots=True)
class _StageDrainResult:
    reports: tuple[WorkerReport, ...] = ()
    made_progress: bool = False
    abort_tick: bool = False
    had_error: bool = False


class _BackgroundCoordinator:
    """Coordinator wrapper that treats immediate shutdown as a yield request."""

    def __init__(self, coordinator: LoopCoordinator, immediate_stop: Event):
        self._coordinator = coordinator
        self._immediate_stop = immediate_stop

    def acquire(self, req: LoopAcquireRequest) -> AbstractContextManager[None]:
        return self._coordinator.acquire(req)

    def yield_to_higher_priority(self) -> bool:
        if self._immediate_stop.is_set():
            return True
        return self._coordinator.yield_to_higher_priority()

    def budget_exhausted(self) -> bool:
        return False

    def remaining_seconds(self) -> float:
        return float("inf")


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
        self._wake_requested = Event()
        self._thread: Thread | None = None
        self._lock = RLock()
        self._wake_lock = Lock()
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
            self.state_service.ledger.recover_abandoned_background_work()
            self._stop_requested.clear()
            self._wake_requested.clear()
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
        self._wake_requested.set()
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

    def wake(self) -> bool:
        """Request one asynchronous background tick from the service run loop."""

        if not self.enabled:
            self._set_status_state("disabled", next_tick=None)
            return False
        with self._wake_lock:
            if self._wake_requested.is_set() or not self._can_accept_wake():
                return False
            if not self._tick_lock.acquire(blocking=False):
                return False
            self._tick_lock.release()
            self._wake_requested.set()
            return True

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
            if self._wake_requested.wait(self.config.startup_delay_seconds):
                self._wake_requested.clear()
            if self._stop_requested.is_set():
                self._set_status_state("stopped", next_tick=None)
                return
            while not self._stop_requested.is_set():
                self._wake_requested.clear()
                self.tick_once()
                if self._stop_requested.is_set():
                    break
                self._set_next_tick(_iso_after(self.config.interval_seconds))
                self._wake_requested.wait(self.config.interval_seconds)
        finally:
            self._set_status_state("stopped", next_tick=None)

    def _can_accept_wake(self) -> bool:
        with self._lock:
            thread = self._thread
            return (
                self.enabled
                and not self._stop_requested.is_set()
                and self._state in {"running", "ticking", "error"}
                and thread is not None
                and thread.is_alive()
            )

    def _tick_once_locked(self) -> list[WorkerReport]:
        self._set_status_state("ticking", next_tick=None)
        tick_started = utc_now_iso()
        with self._lock:
            self._last_tick = tick_started
        reports: list[WorkerReport] = []
        had_error = False
        while True:
            pass_made_progress = False
            for drain_stage in (
                self._drain_extraction,
                self._drain_consolidation,
                self._drain_conflict_review,
                self._drain_summary,
                self._drain_archive,
            ):
                result = drain_stage()
                reports.extend(result.reports)
                pass_made_progress = pass_made_progress or result.made_progress
                had_error = had_error or result.had_error
                if result.abort_tick:
                    break
            if had_error or result.abort_tick or not pass_made_progress:
                break

        with self._lock:
            if had_error:
                self._state = "stopped" if self._stop_requested.is_set() else "error"
            else:
                self._last_success = utc_now_iso()
                self._last_error = None
                self._state = "stopped" if self._stop_requested.is_set() else "running"
        return reports

    def _run_worker(self, worker: ScheduledWorker) -> WorkerReport:
        report = self._invoke_worker(worker)
        self._save_report_checkpoint(report)
        return report

    def _invoke_worker(self, worker: ScheduledWorker) -> WorkerReport:
        req = LoopAcquireRequest(
            loop_name=f"background:{worker.name}",
            priority=LoopPriority.CONSOLIDATION,
            max_chunk_duration=_BACKGROUND_CHUNK_TTL,
        )
        with self._background_coordinator.acquire(req):
            checkpoint = self.checkpoints.load(worker.name)
            report = worker.run(
                self.log,
                self.projections,
                self.emitter,
                self._background_coordinator,
                self._worker_config(),
                checkpoint,
            )
            stamped = _stamp_report(report)
            return stamped

    def _run_extraction_session(
        self,
        worker: MemoryExtractionWorker,
        session_id: str,
    ) -> WorkerReport:
        req = LoopAcquireRequest(
            loop_name=f"background:{worker.name}",
            priority=LoopPriority.CONSOLIDATION,
            max_chunk_duration=_BACKGROUND_CHUNK_TTL,
        )
        with self._background_coordinator.acquire(req):
            checkpoint = self.checkpoints.load(worker.name)
            report = worker.run_session_once(
                session_id,
                checkpoint=checkpoint,
                coordinator=self._background_coordinator,
            )
            return _stamp_report(report)

    def _drain_extraction(self) -> _StageDrainResult:
        worker = self._memory_extraction_worker()
        if worker is None:
            return _StageDrainResult()
        selected_sessions = _eligible_extraction_session_ids(
            self.state_service,
            inactivity_threshold=timedelta(
                hours=self.config.extraction.inactivity_threshold_hours
            ),
            max_sessions=max(1, int(self.config.extraction.max_sessions_per_pass)),
        )
        reports: list[WorkerReport] = []
        made_progress = False
        for session_id in selected_sessions:
            while True:
                if self._yield_or_stop_requested():
                    return _StageDrainResult(
                        reports=tuple(reports),
                        made_progress=made_progress,
                        abort_tick=True,
                    )
                backlog_before = _pending_session_extraction_count(
                    self.state_service,
                    session_id,
                )
                try:
                    report = self._run_extraction_session(worker, session_id)
                except Exception as exc:
                    report = self._error_report(worker.name, f"{worker.name}: {exc}")
                status = report.new_checkpoint.last_status
                if status == "skipped_no_backlog":
                    if backlog_before > 0:
                        invariant = self._error_report(
                            worker.name,
                            (
                                "memory extraction invariant failed: worker returned "
                                "skipped_no_backlog while service backlog predicate was positive"
                            ),
                        )
                        self._save_report_checkpoint(invariant)
                        self._record_report_failure(invariant)
                        return _StageDrainResult(
                            reports=(*reports, invariant),
                            made_progress=made_progress,
                            abort_tick=True,
                            had_error=True,
                        )
                    break
                self._save_report_checkpoint(report)
                reports.append(report)
                if status == "error":
                    self._record_report_failure(report)
                    return _StageDrainResult(
                        reports=tuple(reports),
                        made_progress=made_progress,
                        abort_tick=True,
                        had_error=True,
                    )
                made_progress = True
                if report.yielded_to_higher_priority:
                    return _StageDrainResult(
                        reports=tuple(reports),
                        made_progress=made_progress,
                        abort_tick=True,
                    )
        return _StageDrainResult(reports=tuple(reports), made_progress=made_progress)

    def _drain_consolidation(self) -> _StageDrainResult:
        return self._drain_worker_until_idle(
            MemoryConsolidationWorker.name,
            _pending_consolidation_count,
        )

    def _drain_conflict_review(self) -> _StageDrainResult:
        return self._drain_worker_until_idle(
            MemoryConflictReviewWorker.name,
            _pending_conflict_count,
        )

    def _drain_summary(self) -> _StageDrainResult:
        return self._drain_worker_until_idle(
            MemorySummaryWorker.name,
            lambda state_service: pending_summary_target_count(
                state_service,
                initial_min_beliefs=self.config.summary.initial_min_beliefs,
                changed_source_min=self.config.summary.changed_source_min,
                invalidated_source_min=self.config.summary.invalidated_source_min,
            ),
        )

    def _drain_archive(self) -> _StageDrainResult:
        worker = self._worker_by_name(ArchiveExpiredWorker.name)
        if worker is None or _expired_belief_count(self.state_service) <= 0:
            return _StageDrainResult()
        if self._yield_or_stop_requested():
            return _StageDrainResult(abort_tick=True)
        try:
            report = self._invoke_worker(worker)
        except Exception as exc:
            report = self._error_report(worker.name, f"{worker.name}: {exc}")
        self._save_report_checkpoint(report)
        if report.new_checkpoint.last_status == "error":
            self._record_report_failure(report)
            return _StageDrainResult(
                reports=(report,),
                abort_tick=True,
                had_error=True,
            )
        return _StageDrainResult(
            reports=(report,),
            made_progress=report.emitted > 0,
            abort_tick=report.yielded_to_higher_priority,
        )

    def _drain_worker_until_idle(
        self,
        worker_name: str,
        backlog_count: Callable[[CognitionStateStore], int],
    ) -> _StageDrainResult:
        worker = self._worker_by_name(worker_name)
        if worker is None:
            return _StageDrainResult()
        reports: list[WorkerReport] = []
        made_progress = False
        while True:
            backlog_before = backlog_count(self.state_service)
            if backlog_before <= 0:
                return _StageDrainResult(
                    reports=tuple(reports),
                    made_progress=made_progress,
                )
            if self._yield_or_stop_requested():
                return _StageDrainResult(
                    reports=tuple(reports),
                    made_progress=made_progress,
                    abort_tick=True,
                )
            try:
                report = self._invoke_worker(worker)
            except Exception as exc:
                report = self._error_report(worker.name, f"{worker.name}: {exc}")
            status = report.new_checkpoint.last_status
            if status == "skipped_no_backlog":
                report = self._error_report(
                    worker.name,
                    (
                        f"{worker.name} invariant failed: worker returned "
                        "skipped_no_backlog while service backlog predicate was positive"
                    ),
                )
            self._save_report_checkpoint(report)
            reports.append(report)
            status = report.new_checkpoint.last_status
            if status == "error":
                self._record_report_failure(report)
                return _StageDrainResult(
                    reports=tuple(reports),
                    made_progress=made_progress,
                    abort_tick=True,
                    had_error=True,
                )
            made_progress = True
            if report.yielded_to_higher_priority:
                return _StageDrainResult(
                    reports=tuple(reports),
                    made_progress=made_progress,
                    abort_tick=True,
                )

    def _memory_extraction_worker(self) -> MemoryExtractionWorker | None:
        worker = self._worker_by_name(MemoryExtractionWorker.name)
        return worker if isinstance(worker, MemoryExtractionWorker) else None

    def _worker_by_name(self, name: str) -> ScheduledWorker | None:
        return next((worker for worker in self._workers if worker.name == name), None)

    def _yield_or_stop_requested(self) -> bool:
        return (
            self._stop_requested.is_set()
            or self._background_coordinator.yield_to_higher_priority()
            or self._background_coordinator.budget_exhausted()
        )

    def _save_report_checkpoint(self, report: WorkerReport) -> None:
        self.checkpoints.save(report.new_checkpoint)

    def _record_report_failure(self, report: WorkerReport) -> None:
        self._record_failure(
            f"{report.worker}: {'; '.join(report.notes) or 'worker error'}"
        )

    def _error_report(self, worker_name: str, message: str) -> WorkerReport:
        return _stamp_report(
            WorkerReport(
                worker=worker_name,
                inspected=0,
                emitted=0,
                notes=[message],
                yielded_to_higher_priority=False,
                new_checkpoint=WorkerCheckpoint(
                    worker_name=worker_name,
                    last_status="error",
                ),
            )
        )

    def _worker_config(self) -> SimpleNamespace:
        eligible_inactive_session_ids = _eligible_inactive_session_ids(
            self.store,
            inactivity_threshold=timedelta(
                hours=self.config.extraction.inactivity_threshold_hours
            ),
        )
        return SimpleNamespace(
            llm_provider=self.llm_provider,
            llm_trace_logger=self.llm_trace_logger,
            tools=self.tools,
            inactive_session_ids=eligible_inactive_session_ids,
            summary_initial_min_beliefs=self.config.summary.initial_min_beliefs,
            summary_changed_source_min=self.config.summary.changed_source_min,
            summary_invalidated_source_min=self.config.summary.invalidated_source_min,
        )

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
            MemoryExtractionWorker(
                self.state_service,
                self.llm_provider,
                tools=self.tools,
                llm_trace_logger=self.llm_trace_logger,
            ),
            MemoryConsolidationWorker(
                self.state_service,
                self.llm_provider,
                llm_trace_logger=self.llm_trace_logger,
            ),
            MemoryConflictReviewWorker(
                self.state_service,
                self.llm_provider,
                llm_trace_logger=self.llm_trace_logger,
            ),
            MemorySummaryWorker(
                self.state_service,
                self.llm_provider,
                llm_trace_logger=self.llm_trace_logger,
            ),
            ArchiveExpiredWorker(),
        )


def _eligible_inactive_session_ids(
    store: StateStore,
    *,
    inactivity_threshold: timedelta,
) -> tuple[str, ...]:
    cutoff = utc_now() - inactivity_threshold
    eligible: list[str] = []
    for session in store.list_session_records():
        latest_activity = _latest_session_message_activity(store, session.session_id)
        if latest_activity is not None and latest_activity <= cutoff:
            eligible.append(session.session_id)
    return tuple(eligible)


def _eligible_extraction_session_ids(
    state_service: CognitionStateStore,
    *,
    inactivity_threshold: timedelta,
    max_sessions: int,
) -> tuple[str, ...]:
    cutoff = utc_now() - inactivity_threshold
    eligible: list[str] = []
    for session in state_service.store.list_session_records():
        latest_activity = _latest_session_message_activity(
            state_service.store,
            session.session_id,
        )
        if latest_activity is None or latest_activity > cutoff:
            continue
        if _pending_session_extraction_count(state_service, session.session_id) <= 0:
            continue
        eligible.append(session.session_id)
        if len(eligible) >= max_sessions:
            break
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
    return sum(
        _pending_session_extraction_count(state_service, session_id)
        for session_id in inactive_session_ids
    )


def _pending_session_extraction_count(
    state_service: CognitionStateStore,
    session_id: str,
) -> int:
    target_unit = f"session:{session_id}"
    count = 0
    for source_ref in _session_extraction_source_refs(state_service.store, session_id):
        if _source_status(
            state_service,
            source_ref,
            stage=BackgroundStage.EXTRACTION,
            target_unit=target_unit,
        ) in _RETRYABLE_SOURCE_STATUSES:
            count += 1
    return count


def _session_extraction_source_refs(
    store: StateStore,
    session_id: str,
) -> tuple[BackgroundSourceRef, ...]:
    imported_session_message_ids: set[str] | None = None
    if store.is_import_session(session_id):
        imported_conversation = store.get_imported_conversation_by_session(session_id)
        if imported_conversation is None:
            return ()
        imported_session_message_ids = {
            message.session_message_id
            for message in store.list_imported_messages(
                source_provider=imported_conversation.source_provider,
                external_conversation_id=imported_conversation.external_conversation_id,
            )
        }
    compressed = store.find_latest_compressed_message(session_id)
    boundary_ordinal = compressed.ordinal if compressed is not None else 0
    refs: list[BackgroundSourceRef] = []
    for message in store.list_session_messages(
        session_id,
        after_ordinal=boundary_ordinal,
    ):
        if message.kind in {"compressed_message", "system_reminder"}:
            continue
        if imported_session_message_ids is not None and (
            message.id not in imported_session_message_ids
        ):
            continue
        refs.append(BackgroundSourceRef("session_message", message.id))
    return tuple(refs)


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
            last_status=checkpoint.last_status,
            metadata=checkpoint.metadata,
        ),
    )


def _iso_after(seconds: int) -> str:
    return (utc_now() + timedelta(seconds=seconds)).isoformat()


__all__ = [
    "BackgroundCognitionService",
    "BackgroundCognitionStatus",
]
