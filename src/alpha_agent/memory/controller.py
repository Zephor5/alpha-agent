"""Policy boundary for memory scope, candidates, promotion, and retrieval."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import (
    ExtractedMemoryCandidate,
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    RetrievedContext,
    proposed_layer_for_candidate,
)
from alpha_agent.memory.persistence import PersistedMemory, persist_candidates
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


class MemoryController:
    """Own memory policy decisions outside the runtime orchestration body."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        retriever: MemoryRetriever,
        extractor: MemoryExtractor | None = None,
    ):
        self.store = store
        self.retriever = retriever
        self.extractor = extractor or MemoryExtractor()

    def scope_for_turn(
        self,
        *,
        session_id: str,
        source_metadata: dict[str, Any] | None,
    ) -> MemoryScope:
        """Derive the current write scope from caller source metadata."""

        return MemoryScope.from_source_metadata(
            session_id=session_id,
            source_metadata=source_metadata,
        )

    def retrieve_context(
        self,
        *,
        query: str,
        session_id: str,
        scope: MemoryScope,
        limit: int,
        record_access: bool = True,
    ) -> RetrievedContext:
        """Retrieve only memories visible to the current scope."""

        return self.retriever.retrieve_context(
            query,
            session_id,
            limit=limit,
            scopes=scope.allowed_read_scopes(),
            record_access=record_access,
            access_scope=scope,
        )

    def extract_candidates(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        source_message_ids: list[str],
        scope: MemoryScope,
    ) -> list[MemoryCandidate]:
        """Extract and store candidates before any durable promotion decision."""

        extracted = self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
            source_event_ids=source_message_ids,
        )
        now = utc_now_iso()
        stored: list[MemoryCandidate] = []
        for candidate in extracted:
            memory_candidate = MemoryCandidate(
                id=new_id("cand"),
                candidate_type=candidate.type,
                proposed_layer=proposed_layer_for_candidate(candidate.type),
                content=candidate.content,
                weak_structure=_candidate_weak_structure(candidate),
                salience=candidate.salience,
                confidence=candidate.confidence,
                scope=scope,
                source_message_ids=list(candidate.source_event_ids),
                status="pending",
                created_at=now,
                updated_at=now,
                metadata=dict(candidate.metadata),
            )
            stored.append(self.store.insert_memory_candidate(memory_candidate))
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.candidates.created",
            content=str(len(stored)),
            metadata={
                "candidate_ids": [candidate.id for candidate in stored],
                "scope": scope.to_record(),
            },
        )
        return stored

    def decide_runtime_candidates(
        self,
        *,
        session_id: str,
        candidates: list[MemoryCandidate],
        trusted_scope: bool = True,
    ) -> list[PersistedMemory]:
        """Apply runtime auto-approval policy and persist approved candidates."""

        persisted: list[PersistedMemory] = []
        explicit_batch = any(
            candidate.metadata.get("extractor") == "explicit_or_correction"
            for candidate in candidates
        )
        for candidate in candidates:
            if trusted_scope and self._should_auto_approve(candidate, explicit_batch=explicit_batch):
                approved = self.store.update_memory_candidate_status(
                    candidate.id,
                    "auto_approved",
                    reviewer_metadata={"reviewer": "memory_controller"},
                )
                persisted.extend(
                    self.promote_candidate(
                        approved,
                        action="auto_approve",
                        reviewer="memory_controller",
                        rationale="explicit high-confidence runtime memory",
                    )
                )
                continue
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action="pending",
                    memory_type=None,
                    memory_id=None,
                    reviewer="memory_controller",
                    rationale="candidate requires review",
                    created_at=utc_now_iso(),
                    metadata={},
                )
            )
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.decisions",
            content=str(len(persisted)),
            metadata={
                "persisted": [item.__dict__ for item in persisted],
                "candidate_count": len(candidates),
            },
        )
        return persisted

    def promote_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        action: str,
        reviewer: str | None,
        rationale: str,
    ) -> list[PersistedMemory]:
        """Persist one approved candidate and record the decision."""

        extracted = _stored_candidate_to_extracted(candidate)
        persisted = persist_candidates(self.store, [extracted], scope=candidate.scope)
        for item in persisted:
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action=action,
                    memory_type=item.memory_type,
                    memory_id=item.memory_id,
                    reviewer=reviewer,
                    rationale=rationale,
                    created_at=utc_now_iso(),
                    metadata={"candidate_type": item.candidate_type},
                )
            )
        if not persisted:
            self.store.insert_memory_decision(
                MemoryDecision(
                    id=new_id("decision"),
                    candidate_id=candidate.id,
                    action="skip",
                    memory_type=None,
                    memory_id=None,
                    reviewer=reviewer,
                    rationale="candidate did not map to a durable memory",
                    created_at=utc_now_iso(),
                    metadata={},
                )
            )
        return persisted

    def _should_auto_approve(
        self,
        candidate: MemoryCandidate,
        *,
        explicit_batch: bool = False,
    ) -> bool:
        if explicit_batch and candidate.confidence >= 0.65:
            return True
        extractor = str(candidate.metadata.get("extractor") or "")
        if extractor == "explicit_or_correction":
            return candidate.confidence >= 0.7
        source_text = candidate.content.casefold()
        explicit = "remember" in source_text or "user said:" in source_text
        return explicit and candidate.confidence >= 0.65 and candidate.salience >= 0.75


def _candidate_weak_structure(candidate: ExtractedMemoryCandidate) -> dict[str, Any]:
    return {
        "subject": candidate.subject,
        "predicate": candidate.predicate,
        "object": candidate.object,
    }


def _stored_candidate_to_extracted(candidate: MemoryCandidate) -> ExtractedMemoryCandidate:
    weak = candidate.weak_structure
    return ExtractedMemoryCandidate(
        type=candidate.candidate_type,
        content=candidate.content,
        salience=candidate.salience,
        confidence=candidate.confidence,
        subject=_optional_str(weak.get("subject")),
        predicate=_optional_str(weak.get("predicate")),
        object=_optional_str(weak.get("object")),
        source_event_ids=list(candidate.source_message_ids),
        metadata=dict(candidate.metadata),
    )


def edited_candidate(
    candidate: ExtractedMemoryCandidate,
    *,
    source_message_ids: list[str],
) -> ExtractedMemoryCandidate:
    """Return a candidate with stored source ids preserved after review edits."""

    return replace(candidate, source_event_ids=source_message_ids)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
