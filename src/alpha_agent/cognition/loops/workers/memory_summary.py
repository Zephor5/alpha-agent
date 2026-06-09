"""LLM-mediated cognition-maintenance summaries from consolidated memories."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar

from alpha_agent.cognition.authority import CognitionSourceKind
from alpha_agent.cognition.background_llm_contract import (
    BackgroundLLMValidationContext,
    SourceWindowValidationContext,
    summary_output_json_schema,
)
from alpha_agent.cognition.domain_guidance import active_domain_guidance, summary_target_domain
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
    AtomicBelief,
    BeliefLifecycle,
    BeliefScope,
    CognitiveEventKind,
    DerivationStage,
    Instant,
    Reference,
    SummaryBelief,
    SummaryKind,
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
from alpha_agent.llm.tracing import LLMTraceLogger, traced_llm_complete
from alpha_agent.runtime.context_budget import stable_json
from alpha_agent.utils.time import utc_now_iso

_RETRYABLE_WINDOW_STATUSES = {
    BackgroundProgressStatus.PENDING,
    BackgroundProgressStatus.FAILED,
}
_SUMMARY_INSTRUCTION = """Synthesize one summary belief from selected consolidated memories.

Return only one JSON object. Do not return markdown, code fences, arrays, commentary, or
multiple summaries. The output must validate against this JSON Schema:
{output_schema_json}

Do not include belief ids, summary ids, source ids, provenance, idempotency keys,
confidence, scores, or numeric strength fields. Preserve the selected summary target exactly.
For domain summaries, structure.target_domain is required and must match the selected target.

Selected summary target:
{summary_target_json}

Applicable domain guidance for this worker:
{domain_guidance_json}

Selected consolidated memories:
{source_beliefs_json}"""


@dataclass(frozen=True)
class _SummaryTarget:
    summary_kind: SummaryKind
    scope: BeliefScope
    about: tuple[Reference, ...]
    target_domain: str | None
    source_beliefs: tuple[AtomicBelief, ...]
    active_summary: SummaryBelief | None
    gate: str
    target_unit: str
    source_text: str
    metadata: dict[str, Any]


class _NeverYieldCoordinator:
    def yield_to_higher_priority(self) -> bool:
        return False

    def budget_exhausted(self) -> bool:
        return False

    def remaining_seconds(self) -> float:
        return float("inf")


class MemorySummaryWorker:
    """Ask an LLM to synthesize summary beliefs from consolidated atomic beliefs."""

    name: ClassVar[str] = "memory_summary"
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
        batch_size: int = 2,
        initial_min_beliefs: int = 12,
        changed_source_min: int = 6,
        invalidated_source_min: int = 1,
        worker_id: str | None = None,
        llm_trace_logger: LLMTraceLogger | None = None,
    ):
        self.state_service = state_service
        self.llm_provider = llm_provider
        self.batch_size = max(1, int(batch_size))
        self.initial_min_beliefs = max(1, int(initial_min_beliefs))
        self.changed_source_min = max(1, int(changed_source_min))
        self.invalidated_source_min = max(1, int(invalidated_source_min))
        self.worker_id = worker_id or self.name
        self.llm_trace_logger = llm_trace_logger

    def run_once(
        self,
        *,
        checkpoint: WorkerCheckpoint | None = None,
        coordinator: YieldingCoordinator | None = None,
    ) -> WorkerReport:
        if self.state_service is None:
            raise ValueError("MemorySummaryWorker.run_once requires state_service")
        if self.llm_provider is None:
            raise ValueError("MemorySummaryWorker.run_once requires llm_provider")
        return self._run_with(
            state_service=self.state_service,
            llm_provider=self.llm_provider,
            checkpoint=checkpoint or WorkerCheckpoint(worker_name=self.name),
            coordinator=coordinator or _NeverYieldCoordinator(),
            batch_size=self.batch_size,
            initial_min_beliefs=self.initial_min_beliefs,
            changed_source_min=self.changed_source_min,
            invalidated_source_min=self.invalidated_source_min,
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
                notes=["memory summary skipped: no LLM provider configured"],
            )
        return self._run_with(
            state_service=state_service,
            llm_provider=provider,
            checkpoint=checkpoint,
            coordinator=coordinator,
            batch_size=max(1, int(getattr(config, "summary_batch_size", self.batch_size))),
            initial_min_beliefs=max(
                1,
                int(getattr(config, "summary_initial_min_beliefs", self.initial_min_beliefs)),
            ),
            changed_source_min=max(
                1,
                int(getattr(config, "summary_changed_source_min", self.changed_source_min)),
            ),
            invalidated_source_min=max(
                1,
                int(
                    getattr(
                        config,
                        "summary_invalidated_source_min",
                        self.invalidated_source_min,
                    )
                ),
            ),
            llm_trace_logger=(
                getattr(config, "llm_trace_logger", self.llm_trace_logger)
                or self.llm_trace_logger
            ),
        )

    def _run_with(
        self,
        *,
        state_service: CognitionStateStore,
        llm_provider: LLMProvider,
        checkpoint: WorkerCheckpoint,
        coordinator: YieldingCoordinator,
        batch_size: int,
        initial_min_beliefs: int,
        changed_source_min: int,
        invalidated_source_min: int,
        llm_trace_logger: LLMTraceLogger | None,
    ) -> WorkerReport:
        del batch_size
        target = _next_summary_target(
            state_service,
            initial_min_beliefs=initial_min_beliefs,
            changed_source_min=changed_source_min,
            invalidated_source_min=invalidated_source_min,
        )
        if target is None:
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
                inspected=len(target.source_beliefs),
                emitted=0,
                status="yielded",
                yielded=True,
            )

        source_refs = tuple(
            BackgroundSourceRef("atomic_belief", str(belief.id))
            for belief in target.source_beliefs
        )
        window = state_service.ledger.create_source_window(
            stage=BackgroundStage.SUMMARY,
            target_unit=target.target_unit,
            source_refs=source_refs,
            idempotency_key=_target_idempotency_key(target),
            metadata=target.metadata,
        )
        if window.status == BackgroundProgressStatus.PROCESSED:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(source_refs),
                emitted=0,
                status="skipped_no_backlog",
                metadata={"last_window_id": window.window_id},
            )
        if window.status not in _RETRYABLE_WINDOW_STATUSES:
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(source_refs),
                emitted=0,
                status="skipped_no_backlog",
                metadata={"last_window_id": window.window_id},
            )

        window = _claim_window_and_sources(
            state_service,
            window,
            claimed_by=self.worker_id,
        )
        run = state_service.ledger.start_stage_run(
            worker_id=self.worker_id,
            stage=BackgroundStage.SUMMARY,
            target_unit=window.target_unit,
            window_id=window.window_id,
            input_refs=window.source_refs,
        )
        if coordinator.budget_exhausted() or coordinator.yield_to_higher_priority():
            message = "memory summary yielded before LLM call"
            _mark_failed_if_needed(state_service, window=window, run_id=run.run_id, error=message)
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(source_refs),
                emitted=0,
                status="yielded",
                yielded=True,
                notes=[message],
                metadata={"last_window_id": window.window_id},
            )
        try:
            context = _validation_context(window=window, target=target)
            response = traced_llm_complete(
                llm_provider,
                [_summary_instruction_message(state_service, target, context=context)],
                trace_logger=llm_trace_logger,
                trace_metadata=background_llm_trace_metadata(
                    worker_name=self.name,
                    worker_id=self.worker_id,
                    stage=BackgroundStage.SUMMARY,
                    window=window,
                    run_id=run.run_id,
                ),
                tool_choice="none",
                response_format=JSON_OBJECT_RESPONSE_FORMAT,
            )
            written = state_service.accept_background_llm_json(
                response.content,
                context,
                window_id=window.window_id,
                run_id=run.run_id,
                checkpoint_id=f"checkpoint:{self.name}:{window.window_id}",
            )
            if target.active_summary is not None and written:
                state_service.mark_belief_lifecycle(
                    target.active_summary.id,
                    BeliefLifecycle.SUPERSEDED,
                    at=utc_now_iso(),
                    audit={
                        "kind": "background_summary_supersede",
                        "payload": {
                            "window_id": window.window_id,
                            "replacement_summary_id": str(written[0].id),
                        },
                    },
                )
        except Exception as exc:
            _mark_failed_if_needed(
                state_service,
                window=window,
                run_id=run.run_id,
                error=str(exc),
            )
            return _worker_report(
                self.name,
                checkpoint,
                inspected=len(source_refs),
                emitted=0,
                status="error",
                notes=[str(exc)],
                metadata={"last_window_id": window.window_id},
            )
        return _worker_report(
            self.name,
            checkpoint,
            inspected=len(source_refs),
            emitted=len(written),
            status="ok",
            metadata={"last_window_id": window.window_id},
        )


def pending_summary_target_count(
    state_service: CognitionStateStore,
    *,
    initial_min_beliefs: int,
    changed_source_min: int,
    invalidated_source_min: int,
) -> int:
    return len(
        _summary_targets(
            state_service,
            initial_min_beliefs=max(1, int(initial_min_beliefs)),
            changed_source_min=max(1, int(changed_source_min)),
            invalidated_source_min=max(1, int(invalidated_source_min)),
        )
    )


def _next_summary_target(
    state_service: CognitionStateStore,
    *,
    initial_min_beliefs: int,
    changed_source_min: int,
    invalidated_source_min: int,
) -> _SummaryTarget | None:
    targets = _summary_targets(
        state_service,
        initial_min_beliefs=initial_min_beliefs,
        changed_source_min=changed_source_min,
        invalidated_source_min=invalidated_source_min,
    )
    return targets[0] if targets else None


def _summary_targets(
    state_service: CognitionStateStore,
    *,
    initial_min_beliefs: int,
    changed_source_min: int,
    invalidated_source_min: int,
) -> list[_SummaryTarget]:
    groups: dict[tuple[str, str, tuple[tuple[str, str], ...], str], list[AtomicBelief]] = {}
    for belief in _eligible_consolidated_beliefs(state_service):
        for key in _target_keys_for_belief(belief):
            groups.setdefault(key, []).append(belief)

    targets: list[_SummaryTarget] = []
    for key, beliefs in groups.items():
        summary_kind_value, scope_value, about_key, target_domain = key
        summary_kind = SummaryKind(summary_kind_value)
        scope = BeliefScope(scope_value)
        about = tuple(Reference(kind, ref_id) for kind, ref_id in about_key)
        domain = target_domain or None
        source_beliefs = tuple(sorted(beliefs, key=lambda item: str(item.id)))
        active_summary = _active_summary_for_target(
            state_service,
            summary_kind=summary_kind,
            scope=scope,
            about=about,
            target_domain=domain,
        )
        gate = _summary_gate(
            state_service,
            source_beliefs=source_beliefs,
            active_summary=active_summary,
            initial_min_beliefs=initial_min_beliefs,
            changed_source_min=changed_source_min,
            invalidated_source_min=invalidated_source_min,
        )
        if gate is None:
            continue
        targets.append(
            _build_summary_target(
                summary_kind=summary_kind,
                scope=scope,
                about=about,
                target_domain=domain,
                source_beliefs=source_beliefs,
                active_summary=active_summary,
                gate=gate,
            )
        )
    targets.sort(key=lambda item: item.target_unit)
    return targets


def _eligible_consolidated_beliefs(
    state_service: CognitionStateStore,
) -> list[AtomicBelief]:
    return [
        belief
        for belief in state_service.beliefs.list_active()
        if belief.derivation_stage == DerivationStage.BACKGROUND_CONSOLIDATED
        and not _is_expired_belief(belief)
    ]


def _target_keys_for_belief(
    belief: AtomicBelief,
) -> list[tuple[str, str, tuple[tuple[str, str], ...], str]]:
    about_key = _about_key(belief.about)
    keys: list[tuple[str, str, tuple[tuple[str, str], ...], str]] = []
    if belief.scope == BeliefScope.COUNTERPART:
        keys.append(
            (
                SummaryKind.COUNTERPART_PROFILE.value,
                belief.scope.value,
                about_key,
                "",
            )
        )
    if belief.scope == BeliefScope.SELF:
        keys.append(
            (
                SummaryKind.SELF_MEMORY_SUMMARY.value,
                belief.scope.value,
                about_key,
                "",
            )
        )
    for target_domain in _target_domains_for_belief(belief):
        keys.append(
            (
                SummaryKind.DOMAIN_SUMMARY.value,
                belief.scope.value,
                about_key,
                target_domain,
            )
        )
    return keys


def _target_domains_for_belief(belief: AtomicBelief) -> tuple[str, ...]:
    values: list[str] = []
    for container in (belief.structure, belief.update_policy):
        if not isinstance(container, dict):
            continue
        raw = container.get("target_domain")
        if isinstance(raw, str) and raw.strip():
            values.append(raw.strip())
        raw_many = container.get("target_domains")
        if isinstance(raw_many, list | tuple):
            values.extend(
                item.strip()
                for item in raw_many
                if isinstance(item, str) and item.strip()
            )
    return tuple(sorted(set(values)))


def _active_summary_for_target(
    state_service: CognitionStateStore,
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about: tuple[Reference, ...],
    target_domain: str | None,
) -> SummaryBelief | None:
    summaries = [
        summary
        for summary in state_service.beliefs.list_active_summaries(
            summary_kind=summary_kind,
            scope=scope,
        )
        if _about_key(summary.about) == _about_key(about)
        and summary_target_domain(summary) == target_domain
        and not _is_expired_summary(summary)
    ]
    summaries.sort(key=lambda item: (str(item.held_since), str(item.id)), reverse=True)
    return summaries[0] if summaries else None


def _summary_gate(
    state_service: CognitionStateStore,
    *,
    source_beliefs: tuple[AtomicBelief, ...],
    active_summary: SummaryBelief | None,
    initial_min_beliefs: int,
    changed_source_min: int,
    invalidated_source_min: int,
) -> str | None:
    if active_summary is None:
        return "initial" if len(source_beliefs) >= initial_min_beliefs else None
    source_ids = {str(item.id) for item in source_beliefs}
    summary_source_ids = {str(item) for item in active_summary.source_belief_ids}
    invalidated = sum(
        1
        for source_id in summary_source_ids
        if _source_belief_invalidated(state_service, source_id)
    )
    if invalidated >= invalidated_source_min and source_beliefs:
        return "invalidated_source"
    changed = len(source_ids - summary_source_ids)
    if changed >= changed_source_min:
        return "changed_source"
    return None


def _source_belief_invalidated(state_service: CognitionStateStore, source_id: str) -> bool:
    belief = state_service.beliefs.get_by_id(source_id)
    if not isinstance(belief, AtomicBelief):
        return True
    if belief.lifecycle != BeliefLifecycle.ACTIVE:
        return True
    return _is_expired_belief(belief)


def _build_summary_target(
    *,
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about: tuple[Reference, ...],
    target_domain: str | None,
    source_beliefs: tuple[AtomicBelief, ...],
    active_summary: SummaryBelief | None,
    gate: str,
) -> _SummaryTarget:
    summary_target = {
        "summary_kind": summary_kind.value,
        "scope": scope.value,
        "about": [ref.to_record() for ref in about],
        "target_domain": target_domain,
    }
    metadata = {
        "summary_target": summary_target,
        "gate": gate,
        "source_belief_ids": [str(item.id) for item in source_beliefs],
        "active_summary_id": str(active_summary.id) if active_summary is not None else None,
    }
    target_unit = _summary_target_unit(summary_kind, scope, about, target_domain)
    return _SummaryTarget(
        summary_kind=summary_kind,
        scope=scope,
        about=about,
        target_domain=target_domain,
        source_beliefs=source_beliefs,
        active_summary=active_summary,
        gate=gate,
        target_unit=target_unit,
        source_text=_render_source_text(source_beliefs),
        metadata=metadata,
    )


def _summary_instruction_message(
    state_service: CognitionStateStore,
    target: _SummaryTarget,
    *,
    context: BackgroundLLMValidationContext,
) -> ChatMessage:
    guidance = active_domain_guidance(
        state_service.beliefs,
        target_domain=MemorySummaryWorker.name,
    )
    return {
        "role": "user",
        "content": _SUMMARY_INSTRUCTION.format(
            output_schema_json=_summary_output_schema_json(context),
            summary_target_json=json.dumps(
                target.metadata["summary_target"],
                ensure_ascii=False,
                sort_keys=True,
            ),
            domain_guidance_json=json.dumps(
                [
                    {
                        "id": str(item.belief.id),
                        "content": str(item.belief.content),
                        "target_domain": item.target_domain,
                    }
                    for item in guidance
                ],
                ensure_ascii=False,
                sort_keys=True,
            ),
            source_beliefs_json=json.dumps(
                [_belief_prompt_record(item) for item in target.source_beliefs],
                ensure_ascii=False,
                sort_keys=True,
            ),
        ),
    }


def _summary_output_schema_json(context: BackgroundLLMValidationContext) -> str:
    if context.allowed_summary_kinds is None or len(context.allowed_summary_kinds) != 1:
        raise ValueError("summary worker prompt requires exactly one allowed summary kind")
    if context.required_summary_scope is None:
        raise ValueError("summary worker prompt requires a selected summary scope")
    return json_for_prompt(
        summary_output_json_schema(
            summary_kind=next(iter(context.allowed_summary_kinds)),
            scope=context.required_summary_scope,
            about_refs=context.required_summary_about_refs or frozenset(),
            target_domain=context.required_summary_target_domain,
        )
    )


def _validation_context(
    *,
    window: BackgroundSourceWindow,
    target: _SummaryTarget,
) -> BackgroundLLMValidationContext:
    return BackgroundLLMValidationContext(
        source_kind=CognitionSourceKind.BACKGROUND_SYNTHESIS,
        source_window=SourceWindowValidationContext(
            window_id=window.window_id,
            stage=BackgroundStage.SUMMARY,
            target_unit=window.target_unit,
            source_refs=window.source_refs,
            source_text=target.source_text,
        ),
        input_belief_ids=frozenset(str(item.id) for item in target.source_beliefs),
        allowed_about_refs=frozenset((ref.kind, ref.id) for ref in target.about),
        allowed_summary_kinds=frozenset({target.summary_kind}),
        required_summary_scope=target.scope,
        required_summary_about_refs=frozenset((ref.kind, ref.id) for ref in target.about),
        required_summary_target_domain=target.target_domain,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
    )


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
                stage=BackgroundStage.SUMMARY,
                target_unit=window.target_unit,
                idempotency_key=window.idempotency_key,
                conn=conn,
            )
            state_service.ledger.claim_source(
                source_ref,
                stage=BackgroundStage.SUMMARY,
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
                stage=BackgroundStage.SUMMARY,
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


def _target_idempotency_key(target: _SummaryTarget) -> str:
    digest = hashlib.sha256(
        stable_json(
            {
                "stage": BackgroundStage.SUMMARY.value,
                "target": target.metadata["summary_target"],
                "source_belief_ids": [str(item.id) for item in target.source_beliefs],
                "gate": target.gate,
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"summary:{target.summary_kind.value}:{digest[:32]}"


def _summary_target_unit(
    summary_kind: SummaryKind,
    scope: BeliefScope,
    about: tuple[Reference, ...],
    target_domain: str | None,
) -> str:
    about_text = ",".join(f"{kind}:{ref_id}" for kind, ref_id in _about_key(about))
    domain_text = target_domain or ""
    return (
        f"summary:{summary_kind.value}:scope:{scope.value}:"
        f"about:{about_text}:domain:{domain_text}"
    )


def _render_source_text(source_beliefs: Sequence[AtomicBelief]) -> str:
    return "\n".join(
        f"[atomic_belief:{belief.id}:{belief.memory_kind.value}:{belief.scope.value}] "
        f"{belief.content}"
        for belief in source_beliefs
    )


def _belief_prompt_record(belief: AtomicBelief) -> dict[str, Any]:
    return {
        "id": str(belief.id),
        "memory_kind": belief.memory_kind.value,
        "scope": belief.scope.value,
        "about": [ref.to_record() for ref in belief.about],
        "object": belief.object,
        "content": str(belief.content),
        "structure": belief.structure or {},
        "authority": belief.authority.value,
        "derivation_stage": belief.derivation_stage.value,
    }


def _about_key(about: Sequence[Reference]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((ref.kind, ref.id) for ref in about))


def _is_expired_belief(belief: AtomicBelief) -> bool:
    return _is_expired_instant(belief.validity.valid_until)


def _is_expired_summary(summary: SummaryBelief) -> bool:
    return _is_expired_instant(summary.validity.valid_until)


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


__all__ = ["MemorySummaryWorker", "pending_summary_target_count"]
