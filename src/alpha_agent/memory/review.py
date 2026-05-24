"""Memory review helpers for previewing extracted candidates before storage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace

from alpha_agent.memory.controller import MemoryController
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import (
    ConversationMessage,
    ExtractedMemoryCandidate,
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    proposed_layer_for_candidate,
)
from alpha_agent.memory.persistence import PersistedMemory
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class MemoryCandidateAudit:
    """Source transcript and decision history for a stored candidate."""

    candidate: MemoryCandidate
    source_messages: list[ConversationMessage]
    decisions: list[MemoryDecision]


class MemoryReviewService:
    """Preview, edit, and explicitly persist extracted memory candidates."""

    def __init__(self, store: MemoryStore, extractor: MemoryExtractor | None = None):
        self.store = store
        self.extractor = extractor or MemoryExtractor()
        self.controller = MemoryController(
            store,
            retriever=MemoryRetriever(store),
            extractor=self.extractor,
        )

    def preview(self, message: str) -> list[ExtractedMemoryCandidate]:
        """Extract candidates without writing events or durable memories."""

        return self.extractor.extract(
            user_message=message,
            assistant_response="",
            source_event_ids=[],
        )

    def approve(
        self,
        *,
        message: str,
        session_id: str,
        candidates: list[ExtractedMemoryCandidate],
        scope: MemoryScope | None = None,
    ) -> list[PersistedMemory]:
        """Store approved candidates after writing a review source message."""

        if not candidates:
            return []
        memory_scope = scope or MemoryScope.default()
        persisted: list[PersistedMemory] = []
        with self.store.immediate_transaction() as conn:
            source_message = self.store.append_conversation_message(
                session_id=session_id,
                role="user",
                raw_content=message,
                metadata={"review_mode": True},
                conn=conn,
            )
            approved = [
                candidate
                if candidate.source_event_ids
                else replace(candidate, source_event_ids=[source_message.id])
                for candidate in candidates
            ]
            for candidate in approved:
                stored = self.store.insert_memory_candidate(
                    _stored_review_candidate(candidate, scope=memory_scope),
                    conn=conn,
                )
                approved_candidate = self.store.update_memory_candidate_status(
                    stored.id,
                    "approved",
                    reviewer_metadata={"reviewer": "cli"},
                    conn=conn,
                )
                _record_decision(
                    self.store,
                    candidate_id=approved_candidate.id,
                    action="approve",
                    reviewer="cli",
                    rationale="approved from one-shot review workflow",
                    conn=conn,
                )
                persisted.extend(
                    self.controller.promote_candidate(
                        approved_candidate,
                        action="promote",
                        reviewer="cli",
                        rationale="promoted after one-shot review approval",
                        conn=conn,
                    )
                )
        return persisted

    def reject(
        self,
        *,
        message: str,
        session_id: str,
        candidates: list[ExtractedMemoryCandidate],
        scope: MemoryScope | None = None,
        reviewer: str = "cli",
    ) -> list[MemoryCandidate]:
        """Store rejected one-shot review candidates with source evidence and audit rows."""

        if not candidates:
            return []
        memory_scope = scope or MemoryScope.default()
        rejected_candidates: list[MemoryCandidate] = []
        with self.store.immediate_transaction() as conn:
            source_message = self.store.append_conversation_message(
                session_id=session_id,
                role="user",
                raw_content=message,
                metadata={"review_mode": True},
                conn=conn,
            )
            rejected = [
                candidate
                if candidate.source_event_ids
                else replace(candidate, source_event_ids=[source_message.id])
                for candidate in candidates
            ]
            for candidate in rejected:
                stored = self.store.insert_memory_candidate(
                    _stored_review_candidate(candidate, scope=memory_scope),
                    conn=conn,
                )
                rejected_candidate = self.store.update_memory_candidate_status(
                    stored.id,
                    "rejected",
                    reviewer_metadata={"reviewer": reviewer},
                    conn=conn,
                )
                _record_decision(
                    self.store,
                    candidate_id=rejected_candidate.id,
                    action="reject",
                    reviewer=reviewer,
                    rationale="rejected from one-shot review workflow",
                    conn=conn,
                )
                rejected_candidates.append(rejected_candidate)
        return rejected_candidates

    def list_candidates(
        self,
        *,
        status: str | None = "pending",
        scope: MemoryScope | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """List stored candidates from previous turns."""

        scopes = (scope or MemoryScope.default()).allowed_read_scopes()
        return self.store.list_memory_candidates(status=status, scopes=scopes, limit=limit)

    def list_reviewable_candidates(
        self,
        *,
        scope: MemoryScope | None = None,
        limit: int = 50,
    ) -> list[MemoryCandidate]:
        """List stored candidates that can still be reviewed."""

        scopes = (scope or MemoryScope.default()).allowed_read_scopes()
        return self.store.list_memory_candidates(
            statuses=["pending", "edited"],
            scopes=scopes,
            limit=limit,
        )

    def approve_stored(
        self,
        candidate_id: str,
        *,
        reviewer: str = "cli",
    ) -> list[PersistedMemory]:
        """Approve and promote a stored candidate without re-extracting its source."""

        with self.store.immediate_transaction() as conn:
            candidate = self.store.get_memory_candidate(candidate_id, conn=conn)
            if candidate is None:
                raise KeyError(f"memory candidate not found: {candidate_id}")
            _require_reviewable_visible(candidate, scope=MemoryScope.default())
            approved = self.store.update_memory_candidate_status(
                candidate_id,
                "approved",
                reviewer_metadata={"reviewer": reviewer},
                conn=conn,
            )
            _record_decision(
                self.store,
                candidate_id=approved.id,
                action="approve",
                reviewer=reviewer,
                rationale="approved from stored review workflow",
                conn=conn,
            )
            return self.controller.promote_candidate(
                approved,
                action="promote",
                reviewer=reviewer,
                rationale="promoted after stored review approval",
                conn=conn,
            )

    def reject_stored(
        self,
        candidate_id: str,
        *,
        reviewer: str = "cli",
    ) -> MemoryCandidate:
        """Reject a stored candidate while preserving it for audit."""

        with self.store.immediate_transaction() as conn:
            existing = self.store.get_memory_candidate(candidate_id, conn=conn)
            if existing is None:
                raise KeyError(f"memory candidate not found: {candidate_id}")
            _require_reviewable_visible(existing, scope=MemoryScope.default())
            candidate = self.store.update_memory_candidate_status(
                candidate_id,
                "rejected",
                reviewer_metadata={"reviewer": reviewer},
                conn=conn,
            )
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action="reject",
                    memory_type=None,
                    memory_id=None,
                    reviewer=reviewer,
                    rationale="rejected from stored review workflow",
                    created_at=utc_now_iso(),
                    metadata={},
                ),
                conn=conn,
            )
        return candidate

    def edit_stored(
        self,
        candidate_id: str,
        *,
        content: str | None = None,
        subject: str | None = None,
        predicate: str | None = None,
        object_value: str | None = None,
        reviewer: str = "cli",
    ) -> MemoryCandidate:
        """Edit a stored candidate while preserving source ids and audit evidence."""

        with self.store.immediate_transaction() as conn:
            existing = self.store.get_memory_candidate(candidate_id, conn=conn)
            if existing is None:
                raise KeyError(f"memory candidate not found: {candidate_id}")
            _require_reviewable_visible(existing, scope=MemoryScope.default())
            edited_weak_structure = dict(existing.weak_structure)
            if subject is not None:
                edited_weak_structure["subject"] = subject
            if predicate is not None:
                edited_weak_structure["predicate"] = predicate
            if object_value is not None:
                edited_weak_structure["object"] = object_value
            edited_content = content if content is not None else existing.content
            edited = self.store.update_memory_candidate_review(
                candidate_id,
                content=edited_content,
                weak_structure=edited_weak_structure,
                status="edited",
                reviewer_metadata={"reviewer": reviewer},
                conn=conn,
            )
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=edited.id,
                    action="edit",
                    memory_type=None,
                    memory_id=None,
                    reviewer=reviewer,
                    rationale="edited from stored review workflow",
                    created_at=utc_now_iso(),
                    metadata={
                        "original_content": existing.content,
                        "edited_content": edited.content,
                        "original_weak_structure": dict(existing.weak_structure),
                        "edited_weak_structure": dict(edited.weak_structure),
                        "source_message_ids": list(existing.source_message_ids),
                    },
                ),
                conn=conn,
            )
        return edited

    def inspect_stored(self, candidate_id: str) -> MemoryCandidateAudit:
        """Recover source transcript evidence and decisions for one stored candidate."""

        candidate = self.store.get_memory_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"memory candidate not found: {candidate_id}")
        _require_visible(candidate, scope=MemoryScope.default())
        return MemoryCandidateAudit(
            candidate=candidate,
            source_messages=self.store.list_conversation_messages_by_ids(
                candidate.source_message_ids
            ),
            decisions=self.store.list_memory_decisions(candidate.id),
        )


def edit_candidate(
    candidate: ExtractedMemoryCandidate,
    *,
    content: str | None = None,
    subject: str | None = None,
    predicate: str | None = None,
    object_value: str | None = None,
) -> ExtractedMemoryCandidate:
    """Return an edited copy of a candidate."""

    return replace(
        candidate,
        content=content if content is not None else candidate.content,
        subject=subject if subject is not None else candidate.subject,
        predicate=predicate if predicate is not None else candidate.predicate,
        object=object_value if object_value is not None else candidate.object,
    )


def _stored_review_candidate(
    candidate: ExtractedMemoryCandidate,
    *,
    scope: MemoryScope,
) -> MemoryCandidate:
    now = utc_now_iso()
    return MemoryCandidate(
        id=new_id("cand"),
        candidate_type=candidate.type,
        proposed_layer=proposed_layer_for_candidate(candidate.type),
        content=candidate.content,
        weak_structure={
            "subject": candidate.subject,
            "predicate": candidate.predicate,
            "object": candidate.object,
        },
        salience=candidate.salience,
        confidence=candidate.confidence,
        scope=scope,
        source_message_ids=list(candidate.source_event_ids),
        status="pending",
        created_at=now,
        updated_at=now,
        metadata=dict(candidate.metadata),
    )


def _record_decision(
    store: MemoryStore,
    *,
    candidate_id: str,
    action: str,
    reviewer: str,
    rationale: str,
    conn: sqlite3.Connection | None = None,
) -> None:
    store.insert_memory_decision(
        MemoryDecision(
            id=new_id("decision"),
            candidate_id=candidate_id,
            action=action,
            memory_type=None,
            memory_id=None,
            reviewer=reviewer,
            rationale=rationale,
            created_at=utc_now_iso(),
            metadata={},
        ),
        conn=conn,
    )


def _require_visible(candidate: MemoryCandidate, *, scope: MemoryScope) -> None:
    visible_scope_keys = {item.scope_key for item in scope.allowed_read_scopes()}
    if candidate.scope.scope_key not in visible_scope_keys:
        raise PermissionError(f"memory candidate is outside visible scope: {candidate.id}")


def _require_reviewable_visible(candidate: MemoryCandidate, *, scope: MemoryScope) -> None:
    _require_visible(candidate, scope=scope)
    if candidate.status not in {"pending", "edited"}:
        raise ValueError(
            f"memory candidate {candidate.id} must be pending or edited, got {candidate.status}"
        )
