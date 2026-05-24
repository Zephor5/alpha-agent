"""Memory review helpers for previewing extracted candidates before storage."""

from __future__ import annotations

from dataclasses import replace

from alpha_agent.memory.controller import MemoryController
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import (
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
        source_message = self.store.append_conversation_message(
            session_id=session_id,
            role="user",
            raw_content=message,
            metadata={"review_mode": True},
        )
        approved = [
            candidate
            if candidate.source_event_ids
            else replace(candidate, source_event_ids=[source_message.id])
            for candidate in candidates
        ]
        persisted: list[PersistedMemory] = []
        for candidate in approved:
            stored = self.store.insert_memory_candidate(
                _stored_review_candidate(candidate, scope=memory_scope)
            )
            approved_candidate = self.store.update_memory_candidate_status(
                stored.id,
                "approved",
                reviewer_metadata={"reviewer": "cli"},
            )
            persisted.extend(
                self.controller.promote_candidate(
                    approved_candidate,
                    action="approve",
                    reviewer="cli",
                    rationale="approved from one-shot review workflow",
                )
            )
        return persisted

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

    def approve_stored(
        self,
        candidate_id: str,
        *,
        reviewer: str = "cli",
    ) -> list[PersistedMemory]:
        """Approve and promote a stored candidate without re-extracting its source."""

        candidate = self.store.get_memory_candidate(candidate_id)
        if candidate is None:
            raise KeyError(f"memory candidate not found: {candidate_id}")
        _require_pending_visible(candidate, scope=MemoryScope.default())
        approved = self.store.update_memory_candidate_status(
            candidate_id,
            "approved",
            reviewer_metadata={"reviewer": reviewer},
        )
        return self.controller.promote_candidate(
            approved,
            action="approve",
            reviewer=reviewer,
            rationale="approved from stored review workflow",
        )

    def reject_stored(
        self,
        candidate_id: str,
        *,
        reviewer: str = "cli",
    ) -> MemoryCandidate:
        """Reject a stored candidate while preserving it for audit."""

        existing = self.store.get_memory_candidate(candidate_id)
        if existing is None:
            raise KeyError(f"memory candidate not found: {candidate_id}")
        _require_pending_visible(existing, scope=MemoryScope.default())
        candidate = self.store.update_memory_candidate_status(
            candidate_id,
            "rejected",
            reviewer_metadata={"reviewer": reviewer},
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
            )
        )
        return candidate


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


def _require_pending_visible(candidate: MemoryCandidate, *, scope: MemoryScope) -> None:
    visible_scope_keys = {item.scope_key for item in scope.allowed_read_scopes()}
    if candidate.scope.scope_key not in visible_scope_keys:
        raise PermissionError(f"memory candidate is outside visible scope: {candidate.id}")
    if candidate.status != "pending":
        raise ValueError(
            f"memory candidate {candidate.id} must be pending, got {candidate.status}"
        )
