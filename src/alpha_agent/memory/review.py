"""Memory review helpers for previewing extracted candidates before storage."""

from __future__ import annotations

from dataclasses import replace

from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import ExtractedMemoryCandidate
from alpha_agent.memory.persistence import PersistedMemory, persist_candidates
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.events import create_event


class MemoryReviewService:
    """Preview, edit, and explicitly persist extracted memory candidates."""

    def __init__(self, store: MemoryStore, extractor: MemoryExtractor | None = None):
        self.store = store
        self.extractor = extractor or MemoryExtractor()

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
    ) -> list[PersistedMemory]:
        """Store approved candidates after writing a review source event."""

        if not candidates:
            return []
        source_event = self.store.insert_event(
            create_event(
                session_id=session_id,
                role="user",
                content=message,
                metadata={"review_mode": True},
            )
        )
        approved = [
            candidate
            if candidate.source_event_ids
            else replace(candidate, source_event_ids=[source_event.id])
            for candidate in candidates
        ]
        return persist_candidates(self.store, approved)


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
