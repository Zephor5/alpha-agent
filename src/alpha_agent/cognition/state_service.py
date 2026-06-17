"""Shared write boundary for current cognition state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from alpha_agent.cognition.authority import (
    CognitionSourceKind,
    require_authority_within_ceiling,
)
from alpha_agent.cognition.background_llm_contract import (
    BackgroundLLMValidationError,
    ValidatedAtomicBeliefDraft,
    ValidatedBackgroundLLMOutput,
    ValidatedBeliefUpdate,
    ValidatedConsolidationDecision,
    ValidatedSummaryBeliefDraft,
    validate_background_llm_json,
)
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefRecord,
    BeliefScope,
    DerivationStage,
    DerivationTrace,
    FeedbackEntry,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    Role,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
    belief_ref,
)
from alpha_agent.cognition.models.belief import unknown_subject_ref
from alpha_agent.cognition.processing_ledger import (
    BackgroundSourceRef,
    BackgroundSourceWindow,
    BackgroundStage,
    BackgroundStageRunStatus,
    ProcessingLedger,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.state.store import StateStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS cognition_state_audit (
    audit_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    entity_refs TEXT NOT NULL DEFAULT '[]',
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cognition_state_audit_kind_time
    ON cognition_state_audit(kind, created_at);
"""

_BACKGROUND_LLM_RAW_OUTPUT_PREVIEW_CHARS = 2048
_FEEDBACK_ENTRY_KINDS = frozenset({"confirmed", "contradicted", "corrected"})
_FEEDBACK_CONFLICT_VERDICTS = frozenset({"contradicted", "corrected"})
_FEEDBACK_CONFLICT_TARGET_UNIT = "scope:global"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    loaded = json.loads(value)
    return loaded if loaded is not None else default


@dataclass(frozen=True)
class CognitionStateAuditRecord:
    """Forensic state-write record; never canonical cognition state."""

    audit_id: str
    kind: str
    entity_refs: tuple[Reference, ...]
    payload: dict[str, Any]
    created_at: str


class CognitionStateStore:
    """Shared service for writing current cognition state and support ledgers."""

    def __init__(self, store: StateStore):
        self.store = store
        self.store.initialize()
        self.beliefs = BeliefProjection(store)
        self.ledger = ProcessingLedger(store)
        self._ensure_schema()

    def write_atomic_belief(
        self,
        belief: AtomicBelief,
        *,
        source_kind: CognitionSourceKind | str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief:
        require_authority_within_ceiling(belief.authority, source_kind=source_kind)

        def op(db: sqlite3.Connection) -> AtomicBelief:
            self.beliefs.upsert_atomic(belief, conn=db)
            self._write_optional_audit(
                db,
                audit,
                default_kind="atomic_belief_write",
                entity_refs=(Reference("belief", str(belief.id)),),
            )
            return belief

        return self._write(conn, op)

    def promote_extracted_atomic_belief(
        self,
        belief_id: BeliefId | str,
        *,
        source_kind: CognitionSourceKind | str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief:
        def op(db: sqlite3.Connection) -> AtomicBelief:
            belief = self.beliefs.get_by_id(belief_id, conn=db)
            if not isinstance(belief, AtomicBelief):
                raise BackgroundLLMValidationError(
                    f"promote source id {belief_id!r} does not reference an atomic belief"
                )
            if belief.lifecycle != BeliefLifecycle.ACTIVE:
                raise BackgroundLLMValidationError(
                    "promote requires an active BACKGROUND_EXTRACTED atomic belief"
                )
            if belief.derivation_stage != DerivationStage.BACKGROUND_EXTRACTED:
                raise BackgroundLLMValidationError(
                    "promote requires a BACKGROUND_EXTRACTED atomic belief"
                )
            record = belief.to_record()
            record["derivation_stage"] = DerivationStage.BACKGROUND_CONSOLIDATED.value
            promoted = AtomicBelief.from_record(record)
            self.write_atomic_belief(
                promoted,
                source_kind=source_kind,
                audit=audit,
                conn=db,
            )
            return promoted

        return self._write(conn, op)

    def write_summary_belief(
        self,
        belief: SummaryBelief,
        *,
        source_kind: CognitionSourceKind | str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> SummaryBelief:
        require_authority_within_ceiling(belief.authority, source_kind=source_kind)

        def op(db: sqlite3.Connection) -> SummaryBelief:
            self.beliefs.upsert_summary(belief, conn=db)
            self._write_optional_audit(
                db,
                audit,
                default_kind="summary_belief_write",
                entity_refs=(Reference("belief", str(belief.id)),),
            )
            return belief

        return self._write(conn, op)

    def reaffirm_atomic_belief(
        self,
        belief_id: BeliefId | str,
        *,
        source: Reference | None = None,
        sources: Sequence[Reference] = (),
        observed_at: str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief | None:
        def op(db: sqlite3.Connection) -> AtomicBelief | None:
            source_refs = tuple([*(sources or ()), *(() if source is None else (source,))])
            if not source_refs:
                raise ValueError("reaffirm_atomic_belief requires at least one source")
            updated: AtomicBelief | None = None
            for source_ref in source_refs:
                updated = self.beliefs.reaffirm(
                    belief_id,
                    source=source_ref,
                    observed_at=observed_at,
                    conn=db,
                )
            if updated is not None:
                self._write_optional_audit(
                    db,
                    audit,
                    default_kind="atomic_belief_reaffirm",
                    entity_refs=(Reference("belief", str(updated.id)), *source_refs),
                )
            return updated

        return self._write(conn, op)

    def supersede_atomic_beliefs(
        self,
        old_belief_ids: Sequence[BeliefId | str],
        new_belief: AtomicBelief,
        *,
        source_kind: CognitionSourceKind | str,
        at: str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief:
        require_authority_within_ceiling(new_belief.authority, source_kind=source_kind)

        def op(db: sqlite3.Connection) -> AtomicBelief:
            written = self.beliefs.supersede_many(old_belief_ids, new_belief, at=at, conn=db)
            self._write_optional_audit(
                db,
                audit,
                default_kind="atomic_belief_supersede",
                entity_refs=tuple(
                    [Reference("belief", str(new_belief.id))]
                    + [Reference("belief", str(item)) for item in old_belief_ids]
                ),
            )
            return written

        return self._write(conn, op)

    def mark_belief_lifecycle(
        self,
        belief_id: BeliefId | str,
        lifecycle: BeliefLifecycle,
        *,
        at: str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> None:
        def op(db: sqlite3.Connection) -> None:
            self.beliefs.mark_lifecycle(belief_id, lifecycle, at=at, conn=db)
            self._write_optional_audit(
                db,
                audit,
                default_kind="belief_lifecycle_mark",
                entity_refs=(Reference("belief", str(belief_id)),),
            )

        self._write(conn, op)

    def record_belief_feedback(
        self,
        belief_id: BeliefId | str,
        *,
        kind: str,
        event_id: str,
        at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief | None:
        """Append one compact feedback-history entry to an active atomic belief."""

        feedback_kind = str(kind)
        if feedback_kind not in _FEEDBACK_ENTRY_KINDS:
            allowed = ", ".join(sorted(_FEEDBACK_ENTRY_KINDS))
            raise ValueError(f"unsupported feedback kind {feedback_kind!r}; allowed: {allowed}")
        feedback_event_id = str(event_id).strip()
        if not feedback_event_id:
            raise ValueError("event_id must be non-empty")
        recorded_at = at or utc_now_iso()
        target_date = _utc_date(recorded_at)

        def op(db: sqlite3.Connection) -> AtomicBelief | None:
            belief = self.beliefs.get_by_id(belief_id, conn=db)
            if not isinstance(belief, AtomicBelief):
                return None
            if belief.lifecycle != BeliefLifecycle.ACTIVE:
                return None
            if any(
                _feedback_entry_matches(entry, kind=feedback_kind, utc_date=target_date)
                for entry in belief.feedback_history
            ):
                return None
            entry = FeedbackEntry(
                json.dumps(
                    {
                        "at": recorded_at,
                        "event_id": feedback_event_id,
                        "kind": feedback_kind,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
            record = belief.to_record()
            record["feedback_history"] = [
                *(str(item) for item in belief.feedback_history),
                str(entry),
            ]
            updated = AtomicBelief.from_record(record)
            self.beliefs.upsert_atomic(updated, conn=db)
            self._insert_audit(
                db,
                kind="belief_feedback_recorded",
                payload={
                    "at": recorded_at,
                    "belief_id": str(updated.id),
                    "event_id": feedback_event_id,
                    "kind": feedback_kind,
                },
                entity_refs=(
                    Reference("belief", str(updated.id)),
                    Reference("cognitive_event", feedback_event_id),
                ),
                created_at=utc_now_iso(),
            )
            return updated

        return self._write(conn, op)

    def enqueue_feedback_conflict_review(
        self,
        *,
        belief_id: BeliefId | str,
        verdict: str,
        evidence_quote: str,
        feedback_event_id: str,
        session_id: str,
        user_message_id: str,
        user_message_created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> BackgroundSourceWindow | None:
        """Create an idempotent conflict-review source window for feedback."""

        verdict_value = str(verdict)
        if verdict_value not in _FEEDBACK_CONFLICT_VERDICTS:
            return None
        raw_belief_id = str(belief_id).strip()
        raw_user_message_id = str(user_message_id).strip()
        raw_user_message_created_at = (
            str(user_message_created_at).strip()
            if user_message_created_at is not None
            else None
        )
        if not raw_belief_id:
            raise ValueError("belief_id must be non-empty")
        if not raw_user_message_id:
            raise ValueError("user_message_id must be non-empty")
        if user_message_created_at is not None and not raw_user_message_created_at:
            raise ValueError("user_message_created_at must be non-empty when provided")

        def op(db: sqlite3.Connection) -> BackgroundSourceWindow | None:
            belief = self.beliefs.get_by_id(raw_belief_id, conn=db)
            if not isinstance(belief, AtomicBelief):
                return None
            if belief.lifecycle != BeliefLifecycle.ACTIVE:
                return None
            source_ref = BackgroundSourceRef(
                "conflict",
                f"belief_feedback:{raw_belief_id}:{raw_user_message_id}",
            )
            idempotency_key = (
                f"conflict_review:belief_feedback:{raw_belief_id}:{raw_user_message_id}"
            )
            metadata = {
                "active_belief_ids": [raw_belief_id],
                "belief_id": raw_belief_id,
                "belief_content": str(belief.content),
                "verdict": verdict_value,
                "evidence_quote": str(evidence_quote),
                "feedback_event_id": str(feedback_event_id),
                "session_id": str(session_id),
                "user_message_id": raw_user_message_id,
            }
            if raw_user_message_created_at is not None:
                metadata["user_message_created_at"] = raw_user_message_created_at
            return self.ledger.create_source_window(
                stage=BackgroundStage.CONFLICT_REVIEW,
                target_unit=_FEEDBACK_CONFLICT_TARGET_UNIT,
                source_refs=(source_ref,),
                idempotency_key=idempotency_key,
                metadata=metadata,
                conn=db,
            )

        return self._write(conn, op)

    def write_audit_record(
        self,
        kind: str,
        *,
        payload: Mapping[str, Any] | None = None,
        entity_refs: Sequence[Reference] = (),
        created_at: str | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> CognitionStateAuditRecord:
        def op(db: sqlite3.Connection) -> CognitionStateAuditRecord:
            return self._insert_audit(
                db,
                kind=kind,
                payload=dict(payload or {}),
                entity_refs=tuple(entity_refs),
                created_at=created_at or utc_now_iso(),
            )

        return self._write(conn, op)

    def audit_records(
        self,
        *,
        kind: str | None = None,
        limit: int | None = None,
    ) -> list[CognitionStateAuditRecord]:
        conditions: list[str] = []
        params: list[Any] = []
        if kind is not None:
            conditions.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT *
            FROM cognition_state_audit
            {where}
            ORDER BY created_at ASC, audit_id ASC
        """
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(1, int(limit)))
        with self.store.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._audit_from_row(row) for row in rows]

    def project_reference(self, descriptor: str | Mapping[str, Any]) -> Reference:
        """Normalize a project descriptor into a stable program-owned project ref."""

        normalized = normalize_project_descriptor(descriptor)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        return Reference("project", f"project:{digest}")

    def accept_background_llm_json(
        self,
        raw_output: str,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        checkpoint_id: str | None,
    ) -> list[BeliefRecord]:
        try:
            validated = validate_background_llm_json(raw_output, context)
            return self._accept_validated_background_llm_output(
                validated,
                context,
                window_id=window_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
            )
        except BackgroundLLMValidationError as exc:
            _log_background_llm_validation_failed(
                raw_output,
                context,
                window_id=window_id,
                run_id=run_id,
                error=str(exc),
            )
            self._mark_background_validation_failed(
                context,
                window_id=window_id,
                run_id=run_id,
                error=str(exc),
            )
            raise

    def _accept_validated_background_llm_output(
        self,
        validated: ValidatedBackgroundLLMOutput,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        checkpoint_id: str | None,
    ) -> list[BeliefRecord]:
        written: list[BeliefRecord] = []
        target_unit = _target_unit_for_context(context)
        source_refs = tuple(context.source_window.source_refs)
        output_refs: list[BackgroundSourceRef] = []
        now = utc_now_iso()

        with self.store.immediate_transaction() as conn:
            stage = BackgroundStage(context.source_window.stage)
            if stage in {BackgroundStage.CONSOLIDATION, BackgroundStage.CONFLICT_REVIEW}:
                written, output_refs = self._apply_consolidation_output(
                    validated,
                    context,
                    window_id=window_id,
                    run_id=run_id,
                    now=now,
                    conn=conn,
                )
            else:
                if any(
                    isinstance(payload, ValidatedBeliefUpdate)
                    for payload in validated.payloads
                ):
                    raise BackgroundLLMValidationError(
                        "belief_update persistence is reserved for consolidation stages"
                    )
                written, output_refs = self._apply_create_like_output(
                    validated,
                    context,
                    window_id=window_id,
                    run_id=run_id,
                    now=now,
                    conn=conn,
                )

            for source_ref in source_refs:
                self.ledger.mark_source_processed(
                    source_ref,
                    stage=BackgroundStage(context.source_window.stage),
                    target_unit=target_unit,
                    checkpoint_id=checkpoint_id,
                    conn=conn,
                )
            self.ledger.mark_source_window_processed(window_id, conn=conn)
            if run_id is not None:
                self.ledger.finish_stage_run(
                    run_id,
                    status=BackgroundStageRunStatus.SUCCEEDED,
                    output_refs=output_refs,
                    conn=conn,
                )
        return written

    def _apply_create_like_output(
        self,
        validated: ValidatedBackgroundLLMOutput,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        now: str,
        conn: sqlite3.Connection,
    ) -> tuple[list[BeliefRecord], list[BackgroundSourceRef]]:
        written: list[BeliefRecord] = []
        output_refs: list[BackgroundSourceRef] = []
        for payload in validated.payloads:
            if isinstance(payload, ValidatedAtomicBeliefDraft):
                belief = self._atomic_belief_from_draft(
                    payload,
                    authority=validated.authority,
                    context=context,
                    run_id=run_id,
                    now=now,
                )
                self.write_atomic_belief(
                    belief,
                    source_kind=context.source_kind,
                    audit={
                        "kind": "background_atomic_belief_write",
                        "payload": {
                            "operation": validated.operation,
                            "window_id": window_id,
                            "run_id": run_id,
                            "source_span_note": validated.source_span_note,
                        },
                    },
                    conn=conn,
                )
                written.append(belief)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(belief.id)))
            elif isinstance(payload, ValidatedSummaryBeliefDraft):
                summary_belief = self._summary_belief_from_draft(
                    payload,
                    authority=validated.authority,
                    context=context,
                    run_id=run_id,
                    now=now,
                )
                self.write_summary_belief(
                    summary_belief,
                    source_kind=context.source_kind,
                    audit={
                        "kind": "background_summary_belief_write",
                        "payload": {
                            "operation": validated.operation,
                            "window_id": window_id,
                            "run_id": run_id,
                            "source_span_note": validated.source_span_note,
                        },
                    },
                    conn=conn,
                )
                written.append(summary_belief)
                output_refs.append(BackgroundSourceRef("summary_belief", str(summary_belief.id)))
        return written, output_refs

    def _apply_consolidation_output(
        self,
        validated: ValidatedBackgroundLLMOutput,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        now: str,
        conn: sqlite3.Connection,
    ) -> tuple[list[BeliefRecord], list[BackgroundSourceRef]]:
        stage = BackgroundStage(context.source_window.stage)
        if stage == BackgroundStage.CONSOLIDATION:
            return self._apply_consolidation_batch_output(
                validated,
                context,
                window_id=window_id,
                run_id=run_id,
                now=now,
                conn=conn,
            )
        return self._apply_semantic_consolidation_output(
            validated,
            context,
            window_id=window_id,
            run_id=run_id,
            now=now,
            conn=conn,
        )

    def _apply_semantic_consolidation_output(
        self,
        validated: ValidatedBackgroundLLMOutput,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        now: str,
        conn: sqlite3.Connection,
    ) -> tuple[list[BeliefRecord], list[BackgroundSourceRef]]:
        operation = validated.operation
        if operation == "skip":
            return [], []

        updates = [
            payload for payload in validated.payloads if isinstance(payload, ValidatedBeliefUpdate)
        ]
        drafts = [
            payload
            for payload in validated.payloads
            if isinstance(payload, ValidatedAtomicBeliefDraft)
        ]
        targets = [
            self._require_active_atomic_target(update.target_belief_id, conn=conn)
            for update in updates
        ]
        written: list[BeliefRecord] = []
        output_refs: list[BackgroundSourceRef] = []
        protected_source_ids = {str(target.id) for target in targets}

        if operation == "create":
            belief = self._atomic_belief_from_draft(
                drafts[0],
                authority=validated.authority,
                context=context,
                run_id=run_id,
                now=now,
            )
            self.write_atomic_belief(
                belief,
                source_kind=context.source_kind,
                audit=_background_operation_audit(validated, window_id=window_id, run_id=run_id),
                conn=conn,
            )
            written.append(belief)
            output_refs.append(BackgroundSourceRef("atomic_belief", str(belief.id)))
        elif operation == "strengthen":
            updated = self.reaffirm_atomic_belief(
                targets[0].id,
                sources=_program_attached_sources(context, run_id=run_id),
                observed_at=now,
                audit=_background_operation_audit(
                    validated,
                    window_id=window_id,
                    run_id=run_id,
                ),
                conn=conn,
            )
            if updated is not None:
                written.append(updated)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(updated.id)))
        elif operation == "supersede":
            new_belief = self._atomic_belief_from_draft(
                drafts[0],
                authority=validated.authority,
                context=context,
                run_id=run_id,
                now=now,
                supersedes=targets[0].id,
            )
            self.supersede_atomic_beliefs(
                [targets[0].id],
                new_belief,
                source_kind=context.source_kind,
                at=now,
                audit=_background_operation_audit(
                    validated,
                    window_id=window_id,
                    run_id=run_id,
                ),
                conn=conn,
            )
            written.append(new_belief)
            output_refs.append(BackgroundSourceRef("atomic_belief", str(new_belief.id)))
        elif operation in {"retract", "archive"}:
            lifecycle = (
                BeliefLifecycle.RETRACTED
                if operation == "retract"
                else BeliefLifecycle.ARCHIVED
            )
            self.mark_belief_lifecycle(
                targets[0].id,
                lifecycle,
                at=now,
                audit=_background_operation_audit(
                    validated,
                    window_id=window_id,
                    run_id=run_id,
                ),
                conn=conn,
            )
            materialized = self.beliefs.get_by_id(targets[0].id, conn=conn)
            if isinstance(materialized, AtomicBelief):
                written.append(materialized)
            output_refs.append(BackgroundSourceRef("atomic_belief", str(targets[0].id)))
        else:
            raise BackgroundLLMValidationError(f"unsupported consolidation operation: {operation}")

        self._archive_consolidated_source_drafts(
            context,
            at=now,
            protected_source_ids=protected_source_ids,
            conn=conn,
        )
        return written, output_refs

    def _apply_consolidation_batch_output(
        self,
        validated: ValidatedBackgroundLLMOutput,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        now: str,
        conn: sqlite3.Connection,
    ) -> tuple[list[BeliefRecord], list[BackgroundSourceRef]]:
        if validated.operation != "consolidate_atomic_beliefs":
            raise BackgroundLLMValidationError(
                f"unsupported ordinary consolidation operation: {validated.operation}"
            )
        decisions = [
            payload
            for payload in validated.payloads
            if isinstance(payload, ValidatedConsolidationDecision)
        ]
        if len(decisions) != len(validated.payloads):
            raise BackgroundLLMValidationError(
                "ordinary consolidation payloads must be batch decisions"
            )

        written: list[BeliefRecord] = []
        output_refs: list[BackgroundSourceRef] = []
        for decision in decisions:
            decision_source_refs = tuple(
                BackgroundSourceRef("atomic_belief", source_id)
                for source_id in decision.source_atomic_belief_ids
            )
            self._require_active_extracted_sources(decision_source_refs, conn=conn)
            target = (
                self._require_active_atomic_target(decision.target_belief_id, conn=conn)
                if decision.target_belief_id is not None
                else None
            )
            audit = _background_operation_audit(
                validated,
                window_id=window_id,
                run_id=run_id,
                operation=decision.operation,
                decision_source_ids=decision.source_atomic_belief_ids,
                target_belief_id=decision.target_belief_id,
                rationale=decision.rationale,
            )

            if decision.operation == "promote":
                promoted = self.promote_extracted_atomic_belief(
                    decision.source_atomic_belief_ids[0],
                    source_kind=context.source_kind,
                    audit=audit,
                    conn=conn,
                )
                written.append(promoted)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(promoted.id)))
                continue

            if decision.operation == "skip":
                self._write_optional_audit(
                    conn,
                    audit,
                    default_kind="background_consolidation_operation",
                    entity_refs=tuple(
                        Reference("belief", source_id)
                        for source_id in decision.source_atomic_belief_ids
                    ),
                )
                self._archive_consumed_extracted_sources(
                    decision_source_refs,
                    at=now,
                    operation=decision.operation,
                    conn=conn,
                )
                continue

            if decision.operation == "create":
                if decision.atomic_belief_input is None:
                    raise BackgroundLLMValidationError("create decision missing atomic draft")
                belief = self._atomic_belief_from_draft(
                    decision.atomic_belief_input,
                    authority=validated.authority,
                    context=context,
                    run_id=run_id,
                    now=now,
                    source_refs=decision_source_refs,
                )
                self.write_atomic_belief(
                    belief,
                    source_kind=context.source_kind,
                    audit=audit,
                    conn=conn,
                )
                written.append(belief)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(belief.id)))
            elif decision.operation == "strengthen":
                if target is None:
                    raise BackgroundLLMValidationError("strengthen decision missing target")
                updated = self.reaffirm_atomic_belief(
                    target.id,
                    sources=_program_attached_sources(
                        context,
                        run_id=run_id,
                        source_refs=decision_source_refs,
                    ),
                    observed_at=now,
                    audit=audit,
                    conn=conn,
                )
                if updated is not None:
                    written.append(updated)
                    output_refs.append(BackgroundSourceRef("atomic_belief", str(updated.id)))
            elif decision.operation == "supersede":
                if target is None or decision.atomic_belief_input is None:
                    raise BackgroundLLMValidationError("supersede decision missing target or draft")
                new_belief = self._atomic_belief_from_draft(
                    decision.atomic_belief_input,
                    authority=validated.authority,
                    context=context,
                    run_id=run_id,
                    now=now,
                    source_refs=decision_source_refs,
                    supersedes=target.id,
                )
                self.supersede_atomic_beliefs(
                    [target.id],
                    new_belief,
                    source_kind=context.source_kind,
                    at=now,
                    audit=audit,
                    conn=conn,
                )
                written.append(new_belief)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(new_belief.id)))
            elif decision.operation in {"retract", "archive"}:
                if target is None:
                    raise BackgroundLLMValidationError(
                        f"{decision.operation} decision missing target"
                    )
                lifecycle = (
                    BeliefLifecycle.RETRACTED
                    if decision.operation == "retract"
                    else BeliefLifecycle.ARCHIVED
                )
                self.mark_belief_lifecycle(
                    target.id,
                    lifecycle,
                    at=now,
                    audit=audit,
                    conn=conn,
                )
                materialized = self.beliefs.get_by_id(target.id, conn=conn)
                if isinstance(materialized, AtomicBelief):
                    written.append(materialized)
                output_refs.append(BackgroundSourceRef("atomic_belief", str(target.id)))
            else:
                raise BackgroundLLMValidationError(
                    f"unsupported consolidation decision operation: {decision.operation}"
                )

            self._archive_consumed_extracted_sources(
                decision_source_refs,
                at=now,
                operation=decision.operation,
                conn=conn,
            )

        self._assert_no_consumed_sources_remain_extracted(decisions, conn=conn)
        return written, output_refs

    def _require_active_atomic_target(
        self,
        belief_id: str,
        *,
        conn: sqlite3.Connection,
    ) -> AtomicBelief:
        target = self.beliefs.get_by_id(belief_id, conn=conn)
        if not isinstance(target, AtomicBelief):
            raise BackgroundLLMValidationError(
                f"target belief id {belief_id!r} does not reference an atomic belief"
            )
        if target.lifecycle != BeliefLifecycle.ACTIVE:
            raise BackgroundLLMValidationError(
                "invalid lifecycle transition: update-like consolidation operations "
                f"require an active target, got {target.lifecycle.value}"
            )
        return target

    def _require_active_extracted_sources(
        self,
        source_refs: Sequence[BackgroundSourceRef],
        *,
        conn: sqlite3.Connection,
    ) -> tuple[AtomicBelief, ...]:
        beliefs: list[AtomicBelief] = []
        for source_ref in source_refs:
            if source_ref.source_type != "atomic_belief":
                raise BackgroundLLMValidationError(
                    "ordinary consolidation decisions may consume only atomic_belief sources"
                )
            source_belief = self.beliefs.get_by_id(source_ref.source_id, conn=conn)
            if not isinstance(source_belief, AtomicBelief):
                raise BackgroundLLMValidationError(
                    f"consumed source {source_ref.source_id!r} is not an atomic belief"
                )
            if source_belief.lifecycle != BeliefLifecycle.ACTIVE:
                raise BackgroundLLMValidationError(
                    "consumed consolidation source must be active"
                )
            if source_belief.derivation_stage != DerivationStage.BACKGROUND_EXTRACTED:
                raise BackgroundLLMValidationError(
                    "consumed consolidation source must be BACKGROUND_EXTRACTED"
                )
            beliefs.append(source_belief)
        return tuple(beliefs)

    def _archive_consumed_extracted_sources(
        self,
        source_refs: Sequence[BackgroundSourceRef],
        *,
        at: str,
        operation: str,
        conn: sqlite3.Connection,
    ) -> None:
        self._require_active_extracted_sources(source_refs, conn=conn)
        for source_ref in source_refs:
            self.mark_belief_lifecycle(
                source_ref.source_id,
                BeliefLifecycle.ARCHIVED,
                at=at,
                audit={
                    "kind": "background_consolidation_source_archive",
                    "payload": {
                        "operation": "archive_consumed_extracted_source",
                        "decision_operation": operation,
                    },
                },
                conn=conn,
            )

    def _assert_no_consumed_sources_remain_extracted(
        self,
        decisions: Sequence[ValidatedConsolidationDecision],
        *,
        conn: sqlite3.Connection,
    ) -> None:
        consumed_ids = {
            source_id
            for decision in decisions
            for source_id in decision.source_atomic_belief_ids
        }
        for source_id in consumed_ids:
            belief = self.beliefs.get_by_id(source_id, conn=conn)
            if (
                isinstance(belief, AtomicBelief)
                and belief.lifecycle == BeliefLifecycle.ACTIVE
                and belief.derivation_stage == DerivationStage.BACKGROUND_EXTRACTED
            ):
                raise BackgroundLLMValidationError(
                    f"consumed source {source_id!r} remained active BACKGROUND_EXTRACTED"
                )

    def _archive_consolidated_source_drafts(
        self,
        context: Any,
        *,
        at: str,
        protected_source_ids: set[str],
        conn: sqlite3.Connection,
    ) -> None:
        for source_ref in context.source_window.source_refs:
            if source_ref.source_type != "atomic_belief":
                continue
            if source_ref.source_id in protected_source_ids:
                continue
            source_belief = self.beliefs.get_by_id(source_ref.source_id, conn=conn)
            if not isinstance(source_belief, AtomicBelief):
                continue
            if (
                source_belief.derivation_stage != DerivationStage.BACKGROUND_EXTRACTED
                or source_belief.lifecycle != BeliefLifecycle.ACTIVE
            ):
                continue
            self.mark_belief_lifecycle(
                source_belief.id,
                BeliefLifecycle.ARCHIVED,
                at=at,
                audit={
                    "kind": "background_consolidation_source_archive",
                    "payload": {"operation": "archive_extracted_draft"},
                },
                conn=conn,
            )

    def _atomic_belief_from_draft(
        self,
        draft: Any,
        *,
        authority: Authority,
        context: Any,
        run_id: str | None,
        now: str,
        source_refs: Sequence[BackgroundSourceRef] | None = None,
        supersedes: BeliefId | str | None = None,
    ) -> AtomicBelief:
        about = self._materialized_about(draft, context)
        return AtomicBelief(
            id=BeliefId(new_id("belief")),
            subject=unknown_subject_ref(),
            about=about,
            topic=draft.topic,
            content=NLStatement(draft.content),
            memory_kind=MemoryKind(draft.memory_kind),
            derivation_stage=DerivationStage(context.derivation_stage),
            scope=BeliefScope(draft.scope),
            authority=authority,
            lifecycle=BeliefLifecycle.ACTIVE,
            sources=_program_attached_sources(
                context,
                run_id=run_id,
                source_refs=source_refs,
            ),
            validity=draft.validity or ValidityWindow(observed_at=Instant(now)),
            update_policy=draft.update_policy,
            formed_in=Reference("situation", "situation:background"),
            holder_role=Role("agent"),
            held_since=Instant(now),
            derivation=DerivationTrace(
                deterministic_json(
                    {
                        "source": "background_llm_contract",
                        "source_window_id": context.source_window.window_id,
                        "stage": str(context.source_window.stage),
                        "run_id": run_id or "",
                    }
                )
            ),
            supersedes=(
                belief_ref(BeliefId(str(supersedes))) if supersedes is not None else None
            ),
        )

    def _summary_belief_from_draft(
        self,
        draft: Any,
        *,
        authority: Authority,
        context: Any,
        run_id: str | None,
        now: str,
    ) -> SummaryBelief:
        about = self._materialized_about(draft, context)
        return SummaryBelief(
            id=BeliefId(new_id("belief")),
            subject=unknown_subject_ref(),
            about=about,
            topic=draft.topic,
            content=NLStatement(draft.content),
            summary_kind=SummaryKind(draft.summary_kind),
            derivation_stage=DerivationStage(context.derivation_stage),
            scope=BeliefScope(draft.scope),
            authority=authority,
            lifecycle=BeliefLifecycle.ACTIVE,
            structure=draft.structure,
            sources=_program_attached_sources(context, run_id=run_id),
            validity=draft.validity or ValidityWindow(observed_at=Instant(now)),
            update_policy=draft.update_policy,
            source_belief_ids=[BeliefId(item) for item in sorted(context.input_belief_ids)],
            formed_in=Reference("situation", "situation:background"),
            holder_role=Role("agent"),
            held_since=Instant(now),
            derivation=DerivationTrace(
                deterministic_json(
                    {
                        "source": "background_llm_contract",
                        "source_window_id": context.source_window.window_id,
                        "stage": str(context.source_window.stage),
                        "run_id": run_id or "",
                    }
                )
            ),
        )

    def _materialized_about(self, draft: Any, context: Any) -> list[Reference]:
        if BeliefScope(draft.scope) == BeliefScope.PROJECT and draft.project_descriptor is not None:
            return [self.project_reference(draft.project_descriptor)]
        return list(draft.about)

    def _mark_background_validation_failed(
        self,
        context: Any,
        *,
        window_id: str,
        run_id: str | None,
        error: str,
    ) -> None:
        target_unit = _target_unit_for_context(context)
        source_refs = tuple(context.source_window.source_refs)
        with self.store.immediate_transaction() as conn:
            for source_ref in source_refs:
                self.ledger.mark_source_failed(
                    source_ref,
                    stage=BackgroundStage(context.source_window.stage),
                    target_unit=target_unit,
                    error=error,
                    conn=conn,
                )
            self.ledger.mark_source_window_failed(window_id, error=error, conn=conn)
            if run_id is not None:
                self.ledger.finish_stage_run(
                    run_id,
                    status=BackgroundStageRunStatus.FAILED,
                    error=error,
                    conn=conn,
                )

    def _write_optional_audit(
        self,
        conn: sqlite3.Connection,
        audit: Mapping[str, Any] | None,
        *,
        default_kind: str,
        entity_refs: Sequence[Reference],
    ) -> None:
        if audit is False:
            return
        if audit is None:
            kind = default_kind
            payload: dict[str, Any] = {}
        else:
            kind = str(audit.get("kind") or default_kind)
            raw_payload = audit.get("payload")
            payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        self._insert_audit(
            conn,
            kind=kind,
            payload=payload,
            entity_refs=tuple(entity_refs),
            created_at=utc_now_iso(),
        )

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        payload: dict[str, Any],
        entity_refs: Sequence[Reference],
        created_at: str,
    ) -> CognitionStateAuditRecord:
        record = CognitionStateAuditRecord(
            audit_id=new_id("cogaudit"),
            kind=kind,
            entity_refs=tuple(entity_refs),
            payload=payload,
            created_at=created_at,
        )
        conn.execute(
            """
            INSERT INTO cognition_state_audit
                (audit_id, kind, entity_refs, payload, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record.audit_id,
                record.kind,
                _dumps([item.to_record() for item in record.entity_refs]),
                _dumps(record.payload),
                record.created_at,
            ),
        )
        return record

    def _audit_from_row(self, row: sqlite3.Row) -> CognitionStateAuditRecord:
        entity_records = _loads(row["entity_refs"], [])
        refs = tuple(
            Reference.from_record(item)
            for item in entity_records
            if isinstance(item, dict)
        )
        payload = _loads(row["payload"], {})
        return CognitionStateAuditRecord(
            audit_id=row["audit_id"],
            kind=row["kind"],
            entity_refs=refs,
            payload=payload if isinstance(payload, dict) else {},
            created_at=row["created_at"],
        )

    def _ensure_schema(self) -> None:
        with self.store.transaction() as conn:
            conn.executescript(_AUDIT_SCHEMA)

    def _write(
        self,
        conn: sqlite3.Connection | None,
        op: Any,
    ) -> Any:
        if conn is not None:
            return op(conn)
        with self.store.immediate_transaction() as local:
            return op(local)


def normalize_project_descriptor(descriptor: str | Mapping[str, Any]) -> str:
    """Return a stable canonical project descriptor string."""

    if isinstance(descriptor, str):
        normalized = _normalize_descriptor_text(descriptor)
        if not normalized:
            raise ValueError("project descriptor must be resolvable")
        return normalized
    for key in ("name", "repository", "repo"):
        value = descriptor.get(key)
        if isinstance(value, str) and value.strip():
            return _normalize_descriptor_text(value)
    normalized = _normalize_descriptor_text(_dumps(dict(descriptor)))
    if not normalized or normalized == "{}":
        raise ValueError("project descriptor must be resolvable")
    return normalized


def _normalize_descriptor_text(value: str) -> str:
    normalized = value.replace("\\", "/").strip().casefold()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"/+", "/", normalized)
    return normalized.rstrip("/")


def _target_unit_for_context(context: Any) -> str:
    target_unit = getattr(context.source_window, "target_unit", None)
    if isinstance(target_unit, str) and target_unit:
        return target_unit
    session_id = getattr(context.source_window, "session_id", "")
    if isinstance(session_id, str) and session_id:
        return f"session:{session_id}"
    return "global"


def _feedback_entry_matches(
    entry: FeedbackEntry | str,
    *,
    kind: str,
    utc_date: str,
) -> bool:
    try:
        loaded = json.loads(str(entry))
    except json.JSONDecodeError:
        return False
    if not isinstance(loaded, Mapping):
        return False
    if loaded.get("kind") != kind:
        return False
    at = loaded.get("at")
    return isinstance(at, str) and _utc_date(at) == utc_date


def _utc_date(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:10]
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).date().isoformat()


def _program_attached_sources(
    context: Any,
    *,
    run_id: str | None,
    source_refs: Sequence[BackgroundSourceRef] | None = None,
) -> list[Reference]:
    refs = [Reference("background_source_window", context.source_window.window_id)]
    selected_source_refs = (
        tuple(source_refs) if source_refs is not None else context.source_window.source_refs
    )
    refs.extend(
        Reference(item.source_type, item.source_id)
        for item in selected_source_refs
    )
    if run_id is not None:
        refs.append(Reference("background_stage_run", run_id))
    return refs


def _log_background_llm_validation_failed(
    raw_output: str,
    context: Any,
    *,
    window_id: str,
    run_id: str | None,
    error: str,
) -> None:
    try:
        source_window = getattr(context, "source_window", None)
        stage = getattr(source_window, "stage", None)
        stage_value = stage.value if isinstance(stage, BackgroundStage) else stage
        raw_output_preview = raw_output[:_BACKGROUND_LLM_RAW_OUTPUT_PREVIEW_CHARS]
        payload = {
            "error": error,
            "logged_at": utc_now_iso(),
            "run_id": run_id,
            "window_id": window_id,
            "stage": str(stage_value) if stage_value is not None else None,
            "target_unit": _target_unit_for_context(context),
            "raw_output_length": len(raw_output),
            "raw_output_preview": raw_output_preview,
            "raw_output_truncated": len(raw_output)
            > _BACKGROUND_LLM_RAW_OUTPUT_PREVIEW_CHARS,
        }
        print(
            f"background_llm_validation_failed {deterministic_json(payload)}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        return


def _background_operation_audit(
    validated: ValidatedBackgroundLLMOutput,
    *,
    window_id: str,
    run_id: str | None,
    operation: str | None = None,
    decision_source_ids: Sequence[str] = (),
    target_belief_id: str | None = None,
    rationale: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "operation": operation or validated.operation,
        "window_id": window_id,
        "run_id": run_id,
        "source_span_note": validated.source_span_note,
        "rationale": rationale if rationale is not None else validated.rationale,
    }
    if decision_source_ids:
        payload["source_atomic_belief_ids"] = list(decision_source_ids)
    if target_belief_id is not None:
        payload["target_belief_id"] = target_belief_id
    return {
        "kind": "background_consolidation_operation",
        "payload": payload,
    }
