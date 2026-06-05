"""Shared write boundary for current cognition state."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_agent.cognition.authority import (
    CognitionSourceKind,
    require_authority_within_ceiling,
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
        source: Reference,
        observed_at: str,
        audit: Mapping[str, Any] | None = None,
        conn: sqlite3.Connection | None = None,
    ) -> AtomicBelief | None:
        def op(db: sqlite3.Connection) -> AtomicBelief | None:
            updated = self.beliefs.reaffirm(
                belief_id,
                source=source,
                observed_at=observed_at,
                conn=db,
            )
            if updated is not None:
                self._write_optional_audit(
                    db,
                    audit,
                    default_kind="atomic_belief_reaffirm",
                    entity_refs=(Reference("belief", str(updated.id)), source),
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
        from alpha_agent.cognition.background_llm_contract import (  # noqa: PLC0415
            BackgroundLLMValidationError,
            ValidatedAtomicBeliefDraft,
            ValidatedBeliefUpdate,
            ValidatedSummaryBeliefDraft,
            validate_background_llm_json,
        )

        try:
            validated = validate_background_llm_json(raw_output, context)
        except BackgroundLLMValidationError as exc:
            self._mark_background_validation_failed(
                context,
                window_id=window_id,
                run_id=run_id,
                error=str(exc),
            )
            raise

        if any(isinstance(payload, ValidatedBeliefUpdate) for payload in validated.payloads):
            error = "belief_update persistence is reserved for consolidation phase"
            self._mark_background_validation_failed(
                context,
                window_id=window_id,
                run_id=run_id,
                error=error,
            )
            raise BackgroundLLMValidationError(error)

        written: list[BeliefRecord] = []
        target_unit = _target_unit_for_context(context)
        source_refs = tuple(context.source_window.source_refs)
        output_refs: list[BackgroundSourceRef] = []
        now = utc_now_iso()

        with self.store.immediate_transaction() as conn:
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
                    output_refs.append(
                        BackgroundSourceRef("summary_belief", str(summary_belief.id))
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

    def _atomic_belief_from_draft(
        self,
        draft: Any,
        *,
        authority: Authority,
        requires_confirmation: bool,
        context: Any,
        run_id: str | None,
        now: str,
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
