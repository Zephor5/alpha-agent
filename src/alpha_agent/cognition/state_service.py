"""Shared write boundary for current cognition state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
                    requires_confirmation=validated.requires_confirmation,
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
                    requires_confirmation=validated.requires_confirmation,
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
        operation = validated.operation
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
        requires_confirmation = (
            validated.requires_confirmation or operation == "pending-confirmation"
        )
        written: list[BeliefRecord] = []
        output_refs: list[BackgroundSourceRef] = []
        protected_source_ids = {str(target.id) for target in targets}

        if operation in {"create", "pending-confirmation"}:
            belief = self._atomic_belief_from_draft(
                drafts[0],
                authority=validated.authority,
                requires_confirmation=requires_confirmation,
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
            if not requires_confirmation:
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
            else:
                self._write_optional_audit(
                    conn,
                    _background_operation_audit(
                        validated,
                        window_id=window_id,
                        run_id=run_id,
                    ),
                    default_kind="background_consolidation_confirmation_required",
                    entity_refs=(Reference("belief", str(targets[0].id)),),
                )
                output_refs.append(BackgroundSourceRef("atomic_belief", str(targets[0].id)))
        elif operation == "supersede":
            new_belief = self._atomic_belief_from_draft(
                drafts[0],
                authority=validated.authority,
                requires_confirmation=requires_confirmation,
                context=context,
                run_id=run_id,
                now=now,
                supersedes=targets[0].id,
            )
            if requires_confirmation:
                self.write_atomic_belief(
                    new_belief,
                    source_kind=context.source_kind,
                    audit=_background_operation_audit(
                        validated,
                        window_id=window_id,
                        run_id=run_id,
                    ),
                    conn=conn,
                )
            else:
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
            if not requires_confirmation:
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
            else:
                self._write_optional_audit(
                    conn,
                    _background_operation_audit(
                        validated,
                        window_id=window_id,
                        run_id=run_id,
                    ),
                    default_kind="background_consolidation_confirmation_required",
                    entity_refs=(Reference("belief", str(targets[0].id)),),
                )
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
        requires_confirmation: bool,
        context: Any,
        run_id: str | None,
        now: str,
        supersedes: BeliefId | str | None = None,
    ) -> AtomicBelief:
        about = self._materialized_about(draft, context)
        return AtomicBelief(
            id=BeliefId(new_id("belief")),
            subject=unknown_subject_ref(),
            about=about,
            object=draft.object or draft.content,
            content=NLStatement(draft.content),
            memory_kind=MemoryKind(draft.memory_kind),
            derivation_stage=DerivationStage(context.derivation_stage),
            scope=BeliefScope(draft.scope),
            authority=authority,
            lifecycle=(
                BeliefLifecycle.PENDING_CONFIRMATION
                if requires_confirmation
                else BeliefLifecycle.ACTIVE
            ),
            structure=draft.structure,
            sources=_program_attached_sources(context, run_id=run_id),
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
        requires_confirmation: bool,
        context: Any,
        run_id: str | None,
        now: str,
    ) -> SummaryBelief:
        about = self._materialized_about(draft, context)
        return SummaryBelief(
            id=BeliefId(new_id("belief")),
            subject=unknown_subject_ref(),
            about=about,
            object=draft.object or draft.content,
            content=NLStatement(draft.content),
            summary_kind=SummaryKind(draft.summary_kind),
            derivation_stage=DerivationStage(context.derivation_stage),
            scope=BeliefScope(draft.scope),
            authority=authority,
            lifecycle=(
                BeliefLifecycle.PENDING_CONFIRMATION
                if requires_confirmation
                else BeliefLifecycle.ACTIVE
            ),
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


def _program_attached_sources(context: Any, *, run_id: str | None) -> list[Reference]:
    refs = [Reference("background_source_window", context.source_window.window_id)]
    refs.extend(
        Reference(item.source_type, item.source_id)
        for item in context.source_window.source_refs
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
) -> dict[str, Any]:
    return {
        "kind": "background_consolidation_operation",
        "payload": {
            "operation": validated.operation,
            "window_id": window_id,
            "run_id": run_id,
            "requires_confirmation": validated.requires_confirmation,
            "source_span_note": validated.source_span_note,
        },
    }
