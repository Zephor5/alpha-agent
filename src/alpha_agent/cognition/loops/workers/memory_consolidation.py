"""LLM-mediated cognition-maintenance consolidation and conflict review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
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
    AtomicBelief,
    BeliefLifecycle,
    BeliefScope,
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
from alpha_agent.llm.base import JSON_OBJECT_RESPONSE_FORMAT, ChatMessage, LLMProvider
from alpha_agent.runtime.context_budget import stable_json
from alpha_agent.utils.time import utc_now_iso

_RETRYABLE_SOURCE_STATUSES = {None, BackgroundProgressStatus.FAILED}
_CONSOLIDATION_INSTRUCTION = """Compare extracted atomic belief drafts with active beliefs.

Return exactly one JSON object using the background cognition contract.
Use only these semantic operations:
- create: accept a draft as a new consolidated atomic belief
- strengthen: reaffirm one active belief with corroborating evidence
- supersede: replace one active belief with a new consolidated atomic belief
- retract: mark one active belief retracted
- archive: mark one active belief archived
- pending-confirmation: create a pending atomic belief candidate for human confirmation

Update-like operations must target one of the active belief ids shown below. Do not include
source ids, provenance, idempotency keys, generated ids, confidence, scores, or numeric
strength fields.

Extracted drafts:
{drafts_json}

Active beliefs included as valid update targets:
{active_beliefs_json}"""

_CONFLICT_REVIEW_INSTRUCTION = """Review the queued memory conflict.

Return exactly one JSON object using the background cognition contract. If resolving the
conflict automatically is unsafe, use operation "pending-confirmation" and set
requires_confirmation to true. Do not mutate active memory unless the conflict can be safely
resolved from the supplied evidence. Do not include generated ids, source refs, provenance,
idempotency keys, confidence, scores, or numeric strength fields.

Conflict source:
{conflict_json}

Active beliefs included as valid update targets:
{active_beliefs_json}"""


@dataclass(frozen=True)
class _ConsolidationCandidate:
    target_unit: str
    source_refs: tuple[BackgroundSourceRef, ...]
    drafts: tuple[AtomicBelief, ...]
    active_beliefs: tuple[AtomicBelief, ...]
    source_text: str
    metadata: dict[str, Any]


class _NeverYieldCoordinator:
    def yield_to_higher_priority(self) -> bool:
        return False

    def budget_exhausted(self) -> bool:
        return False

    def remaining_seconds(self) -> float:
        return float("inf")


class MemoryConsolidationWorker:
    """Ask an LLM to consolidate extracted atomic drafts against active beliefs."""

    name: ClassVar[str] = "memory_consolidation"
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
        batch_size: int = 12,
        worker_id: str | None = None,
    ):
        self.state_service = state_service
        self.llm_provider = llm_provider
        self.batch_size = max(1, int(batch_size))
        self.worker_id = worker_id or self.name

    def run_once(
        self,
        *,
        checkpoint: WorkerCheckpoint | None = None,
        coordinator: YieldingCoordinator | None = None,
    ) -> WorkerReport:
        if self.state_service is None:
            raise ValueError("MemoryConsolidationWorker.run_once requires state_service")
        if self.llm_provider is None:
            raise ValueError("MemoryConsolidationWorker.run_once requires llm_provider")
        return self._run_with(
            state_service=self.state_service,
            llm_provider=self.llm_provider,
            checkpoint=checkpoint or WorkerCheckpoint(worker_name=self.name),
            coordinator=coordinator or _NeverYieldCoordinator(),
            batch_size=self.batch_size,
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
                notes=["memory consolidation skipped: no LLM provider configured"],
            )
        return self._run_with(
            state_service=state_service,
            llm_provider=provider,
            checkpoint=checkpoint,
            coordinator=coordinator,
            batch_size=max(1, int(getattr(config, "consolidation_batch_size", self.batch_size))),
        )

    def _run_with(
        self,
        *,
        state_service: CognitionStateStore,
        llm_provider: LLMProvider,
        checkpoint: WorkerCheckpoint,
        coordinator: YieldingCoordinator,
        batch_size: int,
    ) -> WorkerReport:
        candidate = _next_consolidation_candidate(state_service, batch_size=batch_size)
        if candidate is None:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
            )
        if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(candidate.source_refs),
                emitted=0,
                status="yielded",
                yielded=True,
            )
        window = state_service.ledger.create_source_window(
            stage=BackgroundStage.CONSOLIDATION,
            target_unit=candidate.target_unit,
            source_refs=candidate.source_refs,
            idempotency_key=_candidate_idempotency_key(
                BackgroundStage.CONSOLIDATION,
                candidate.target_unit,
                candidate.source_refs,
                candidate.metadata,
            ),
            metadata=candidate.metadata,
        )
        if window.status == BackgroundProgressStatus.PROCESSED:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(candidate.source_refs),
                emitted=0,
                status="skipped_no_backlog",
                metadata={"last_window_id": window.window_id},
            )
        window = _claim_window_and_sources(
            state_service,
            window,
            stage=BackgroundStage.CONSOLIDATION,
            claimed_by=self.worker_id,
        )
        run = state_service.ledger.start_stage_run(
            worker_id=self.worker_id,
            stage=BackgroundStage.CONSOLIDATION,
            target_unit=window.target_unit,
            window_id=window.window_id,
            input_refs=window.source_refs,
        )
        if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
            message = "memory consolidation yielded before LLM call"
            _mark_failed_if_needed(
                state_service,
                window=window,
                run_id=run.run_id,
                stage=BackgroundStage.CONSOLIDATION,
                error=message,
            )
            return _worker_report(
                self.name,
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
                [_consolidation_instruction_message(candidate)],
                tool_choice="none",
                response_format=JSON_OBJECT_RESPONSE_FORMAT,
            )
            written = state_service.accept_background_llm_json(
                response.content,
                _validation_context_for_candidate(window, candidate),
                window_id=window.window_id,
                run_id=run.run_id,
                checkpoint_id=f"checkpoint:{self.name}:{window.window_id}",
            )
        except Exception as exc:
            _mark_failed_if_needed(
                state_service,
                window=window,
                run_id=run.run_id,
                stage=BackgroundStage.CONSOLIDATION,
                error=str(exc),
            )
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(candidate.source_refs),
                emitted=0,
                status="error",
                notes=[str(exc)],
                metadata={"last_window_id": window.window_id},
            )
        return _worker_report(
            self.name,
            checkpoint,
            inspected=len(candidate.source_refs),
            emitted=len(written),
            status="ok",
            metadata={"last_window_id": window.window_id},
        )


class MemoryConflictReviewWorker:
    """Review queued conflicts and persist only validated safe or pending outcomes."""

    name: ClassVar[str] = "memory_conflict_review"
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
        worker_id: str | None = None,
    ):
        self.state_service = state_service
        self.llm_provider = llm_provider
        self.worker_id = worker_id or self.name

    def run_once(
        self,
        *,
        checkpoint: WorkerCheckpoint | None = None,
        coordinator: YieldingCoordinator | None = None,
    ) -> WorkerReport:
        if self.state_service is None:
            raise ValueError("MemoryConflictReviewWorker.run_once requires state_service")
        if self.llm_provider is None:
            raise ValueError("MemoryConflictReviewWorker.run_once requires llm_provider")
        return self._run_with(
            state_service=self.state_service,
            llm_provider=self.llm_provider,
            checkpoint=checkpoint or WorkerCheckpoint(worker_name=self.name),
            coordinator=coordinator or _NeverYieldCoordinator(),
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
                notes=["memory conflict review skipped: no LLM provider configured"],
            )
        return self._run_with(
            state_service=state_service,
            llm_provider=provider,
            checkpoint=checkpoint,
            coordinator=coordinator,
        )

    def _run_with(
        self,
        *,
        state_service: CognitionStateStore,
        llm_provider: LLMProvider,
        checkpoint: WorkerCheckpoint,
        coordinator: YieldingCoordinator,
    ) -> WorkerReport:
        window = _next_conflict_review_window(state_service)
        if window is None:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=0,
                emitted=0,
                status="skipped_no_backlog",
            )
        if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(window.source_refs),
                emitted=0,
                status="yielded",
                yielded=True,
                metadata={"last_window_id": window.window_id},
            )
        window = _claim_window_and_sources(
            state_service,
            window,
            stage=BackgroundStage.CONFLICT_REVIEW,
            claimed_by=self.worker_id,
        )
        run = state_service.ledger.start_stage_run(
            worker_id=self.worker_id,
            stage=BackgroundStage.CONFLICT_REVIEW,
            target_unit=window.target_unit,
            window_id=window.window_id,
            input_refs=window.source_refs,
        )
        try:
            active_beliefs = _beliefs_by_ids(
                state_service,
                _active_belief_ids_from_metadata(window.metadata),
            )
            if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
                message = "memory conflict review yielded before LLM call"
                _mark_failed_if_needed(
                    state_service,
                    window=window,
                    run_id=run.run_id,
                    stage=BackgroundStage.CONFLICT_REVIEW,
                    error=message,
                )
                return _worker_report(
                    self.name,
                    checkpoint,
                    inspected=len(window.source_refs),
                    emitted=0,
                    status="yielded",
                    yielded=True,
                    notes=[message],
                    metadata={"last_window_id": window.window_id},
                )
            response = llm_provider.complete(
                [_conflict_review_instruction_message(window, active_beliefs)],
                tool_choice="none",
                response_format=JSON_OBJECT_RESPONSE_FORMAT,
            )
            written = state_service.accept_background_llm_json(
                response.content,
                _validation_context_for_conflict(window, active_beliefs),
                window_id=window.window_id,
                run_id=run.run_id,
                checkpoint_id=f"checkpoint:{self.name}:{window.window_id}",
            )
        except Exception as exc:
            _mark_failed_if_needed(
                state_service,
                window=window,
                run_id=run.run_id,
                stage=BackgroundStage.CONFLICT_REVIEW,
                error=str(exc),
            )
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(window.source_refs),
                emitted=0,
                status="error",
                notes=[str(exc)],
                metadata={"last_window_id": window.window_id},
            )
        return _worker_report(
            self.name,
            checkpoint,
            inspected=len(window.source_refs),
            emitted=len(written),
            status="ok",
            metadata={"last_window_id": window.window_id},
        )


def _next_consolidation_candidate(
    state_service: CognitionStateStore,
    *,
    batch_size: int,
) -> _ConsolidationCandidate | None:
    active = sorted(state_service.beliefs.list_active(), key=lambda item: str(item.id))
    drafts = [
        belief
        for belief in active
        if belief.derivation_stage == DerivationStage.BACKGROUND_EXTRACTED
    ]
    for draft in drafts:
        target_unit = _target_unit_for_belief(draft)
        source_ref = BackgroundSourceRef("atomic_belief", str(draft.id))
        if _source_status(
            state_service,
            source_ref,
            stage=BackgroundStage.CONSOLIDATION,
            target_unit=target_unit,
        ) not in _RETRYABLE_SOURCE_STATUSES:
            continue
        selected_drafts = (draft,)
        source_refs = (source_ref,)
        active_beliefs = tuple(
            item
            for item in active
            if item.derivation_stage != DerivationStage.BACKGROUND_EXTRACTED
            and _same_consolidation_bucket(item, draft)
        )
        source_text = str(draft.content)
        return _ConsolidationCandidate(
            target_unit=target_unit,
            source_refs=source_refs,
            drafts=selected_drafts,
            active_beliefs=active_beliefs,
            source_text=source_text,
            metadata={
                "draft_belief_ids": [str(item.id) for item in selected_drafts],
                "active_belief_ids": [str(item.id) for item in active_beliefs],
            },
        )
    return None


def _next_conflict_review_window(
    state_service: CognitionStateStore,
) -> BackgroundSourceWindow | None:
    for status in (BackgroundProgressStatus.PENDING, BackgroundProgressStatus.FAILED):
        windows = state_service.ledger.list_source_windows(
            stage=BackgroundStage.CONFLICT_REVIEW,
            status=status,
        )
        if windows:
            return windows[0]
    return None


def _claim_window_and_sources(
    state_service: CognitionStateStore,
    window: BackgroundSourceWindow,
    *,
    stage: BackgroundStage,
    claimed_by: str,
) -> BackgroundSourceWindow:
    with state_service.store.immediate_transaction() as conn:
        for source_ref in window.source_refs:
            state_service.ledger.mark_source_pending(
                source_ref,
                stage=stage,
                target_unit=window.target_unit,
                idempotency_key=window.idempotency_key,
                conn=conn,
            )
            state_service.ledger.claim_source(
                source_ref,
                stage=stage,
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


def _validation_context_for_candidate(
    window: BackgroundSourceWindow,
    candidate: _ConsolidationCandidate,
) -> BackgroundLLMValidationContext:
    target_ids = frozenset(str(item.id) for item in candidate.active_beliefs)
    return BackgroundLLMValidationContext(
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        source_window=SourceWindowValidationContext(
            window_id=window.window_id,
            stage=BackgroundStage.CONSOLIDATION,
            target_unit=window.target_unit,
            source_refs=window.source_refs,
            source_text=candidate.source_text,
        ),
        allowed_target_belief_ids=target_ids,
        input_belief_ids=frozenset(
            [*target_ids, *(str(item.id) for item in candidate.drafts)]
        ),
        allowed_about_refs=_allowed_about_refs((*candidate.drafts, *candidate.active_beliefs)),
        derivation_stage=DerivationStage.BACKGROUND_CONSOLIDATED,
    )


def _validation_context_for_conflict(
    window: BackgroundSourceWindow,
    active_beliefs: Sequence[AtomicBelief],
) -> BackgroundLLMValidationContext:
    target_ids = frozenset(str(item.id) for item in active_beliefs)
    return BackgroundLLMValidationContext(
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        source_window=SourceWindowValidationContext(
            window_id=window.window_id,
            stage=BackgroundStage.CONFLICT_REVIEW,
            target_unit=window.target_unit,
            source_refs=window.source_refs,
            source_text=_source_text_from_window(window),
        ),
        allowed_target_belief_ids=target_ids,
        input_belief_ids=target_ids,
        allowed_about_refs=_allowed_about_refs(active_beliefs),
        derivation_stage=DerivationStage.BACKGROUND_CONSOLIDATED,
    )


def _active_belief_ids_from_metadata(metadata: dict[str, Any]) -> tuple[str, ...]:
    raw_ids = metadata.get("active_belief_ids", ())
    if not isinstance(raw_ids, list | tuple):
        return ()
    return tuple(str(item) for item in raw_ids if isinstance(item, str) and item.strip())


def _beliefs_by_ids(
    state_service: CognitionStateStore,
    belief_ids: Sequence[str],
) -> tuple[AtomicBelief, ...]:
    beliefs: list[AtomicBelief] = []
    for belief_id in belief_ids:
        belief = state_service.beliefs.get_by_id(belief_id)
        if isinstance(belief, AtomicBelief) and belief.lifecycle == BeliefLifecycle.ACTIVE:
            beliefs.append(belief)
    return tuple(beliefs)


def _source_text_from_window(window: BackgroundSourceWindow) -> str | None:
    raw = window.metadata.get("source_text")
    return raw if isinstance(raw, str) and raw.strip() else None


def _allowed_about_refs(beliefs: Sequence[AtomicBelief]) -> frozenset[tuple[str, str]]:
    refs = {(ref.kind, ref.id) for belief in beliefs for ref in belief.about}
    return frozenset(refs) if refs else frozenset()


def _consolidation_instruction_message(candidate: _ConsolidationCandidate) -> ChatMessage:
    return {
        "role": "user",
        "content": _CONSOLIDATION_INSTRUCTION.format(
            drafts_json=json.dumps(
                [_belief_prompt_record(item) for item in candidate.drafts],
                ensure_ascii=False,
                sort_keys=True,
            ),
            active_beliefs_json=json.dumps(
                [_belief_prompt_record(item) for item in candidate.active_beliefs],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    }


def _conflict_review_instruction_message(
    window: BackgroundSourceWindow,
    active_beliefs: Sequence[AtomicBelief],
) -> ChatMessage:
    return {
        "role": "user",
        "content": _CONFLICT_REVIEW_INSTRUCTION.format(
            conflict_json=json.dumps(window.metadata, ensure_ascii=False, sort_keys=True),
            active_beliefs_json=json.dumps(
                [_belief_prompt_record(item) for item in active_beliefs],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    }


def _belief_prompt_record(belief: AtomicBelief) -> dict[str, Any]:
    return {
        "id": str(belief.id),
        "memory_kind": belief.memory_kind.value,
        "scope": belief.scope.value,
        "about": [ref.to_record() for ref in belief.about],
        "object": belief.object,
        "content": str(belief.content),
        "derivation_stage": belief.derivation_stage.value,
        "authority": belief.authority.value,
        "lifecycle": belief.lifecycle.value,
    }


def _target_unit_for_belief(belief: AtomicBelief) -> str:
    if belief.scope == BeliefScope.GLOBAL:
        return "scope:global"
    about = ",".join(
        f"{ref.kind}:{ref.id}" for ref in sorted(belief.about, key=lambda ref: (ref.kind, ref.id))
    )
    return f"scope:{belief.scope.value}:{about}"


def _same_consolidation_bucket(left: AtomicBelief, right: AtomicBelief) -> bool:
    return _target_unit_for_belief(left) == _target_unit_for_belief(right)


def _source_status(
    state_service: CognitionStateStore,
    source_ref: BackgroundSourceRef,
    *,
    stage: BackgroundStage,
    target_unit: str,
) -> BackgroundProgressStatus | None:
    try:
        return state_service.ledger.get_source_progress(
            source_ref,
            stage=stage,
            target_unit=target_unit,
        ).status
    except KeyError:
        return None


def _candidate_idempotency_key(
    stage: BackgroundStage,
    target_unit: str,
    source_refs: Sequence[BackgroundSourceRef],
    metadata: dict[str, Any],
) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "stage": stage.value,
                "target_unit": target_unit,
                "source_refs": [ref.to_record() for ref in source_refs],
                "metadata": metadata,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"{stage.value}:{target_unit}:{digest[:32]}"


def _mark_failed_if_needed(
    state_service: CognitionStateStore,
    *,
    window: BackgroundSourceWindow,
    run_id: str,
    stage: BackgroundStage,
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
                stage=stage,
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


__all__ = ["MemoryConflictReviewWorker", "MemoryConsolidationWorker"]
