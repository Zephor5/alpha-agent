"""Manual memory consolidation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.memory.controller import MemoryController, MemoryPromotionPolicyError
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import MemoryCandidate, MemoryDecision
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class ConsolidationReport:
    """Summary of a manual consolidation run."""

    scanned_episodes: int
    promoted_facts: int
    ignored_low_salience: int
    promoted_count: int = 0
    merged_count: int = 0
    skipped_count: int = 0
    superseded_count: int = 0
    conflict_count: int = 0
    scanned_candidates: int = 0
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Return a short report suitable for CLI output."""

        lines = [
            "Consolidation complete",
            f"- scanned episodes: {self.scanned_episodes}",
            f"- scanned candidates: {self.scanned_candidates}",
            f"- promoted facts: {self.promoted_facts}",
            f"- promoted: {self.promoted_count}",
            f"- merged: {self.merged_count}",
            f"- skipped: {self.skipped_count}",
            f"- superseded: {self.superseded_count}",
            f"- conflicts queued: {self.conflict_count}",
            f"- ignored low-salience episodes: {self.ignored_low_salience}",
        ]
        lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)


class ConsolidationService:
    """Promote high-salience episodic memories into stable semantic facts."""

    def __init__(
        self,
        store: MemoryStore,
        semantic_manager: SemanticMemoryManager | None = None,
        extractor: MemoryExtractor | None = None,
    ):
        self.store = store
        self.semantic_manager = semantic_manager or SemanticMemoryManager(store)
        self.extractor = extractor or MemoryExtractor()
        self.controller = MemoryController(
            store,
            retriever=MemoryRetriever(store),
            extractor=self.extractor,
        )

    def consolidate(self, limit: int = 100) -> ConsolidationReport:
        """Run deterministic manual consolidation."""

        episodes = self.store.list_episodic_memories(limit=limit)
        stored_candidates = self._consolidatable_candidates(limit=limit)
        promoted = 0
        ignored = 0
        action_counts: dict[str, int] = defaultdict(int)
        notes: list[str] = []
        generated_candidate_count = 0
        for candidate in stored_candidates:
            if self._already_promoted(candidate):
                action_counts["skip"] += 1
                continue
            approved = candidate
            if candidate.status == "pending":
                if not self._is_stable_repeated_candidate(candidate, stored_candidates):
                    action_counts["skip"] += 1
                    continue
                approved = self._auto_approve_candidate(candidate)
            promoted_items = self._promote_candidate_or_reject(
                approved,
                rationale="promoted during consolidation",
            )
            if not promoted_items:
                action_counts["skip"] += 1
            for item in promoted_items:
                action_counts[item.action] += 1

        for episode in episodes:
            if episode.salience < 0.65:
                ignored += 1
                continue
            generated = self.controller.extract_candidates(
                session_id="memory-consolidation",
                user_message=episode.content,
                assistant_response="",
                source_message_ids=episode.source_event_ids,
                scope=episode.scope,
            )
            generated_candidate_count += len(generated)
            for candidate in generated:
                if candidate.candidate_type != "semantic":
                    rejected = self.store.update_memory_candidate_status(
                        candidate.id,
                        "rejected",
                        reviewer_metadata={"reviewer": "consolidation_service"},
                    )
                    self._record_candidate_decision(
                        rejected,
                        action="reject",
                        memory_type=None,
                        memory_id=None,
                        rationale="non-semantic consolidation candidate skipped",
                    )
                    action_counts["skip"] += 1
                    continue
                approved = self._auto_approve_candidate(candidate)
                promoted_items = self._promote_candidate_or_reject(
                    approved,
                    rationale="promoted from episodic consolidation candidate",
                )
                if not promoted_items:
                    action_counts["skip"] += 1
                for item in promoted_items:
                    action_counts[item.action] += 1
                    if item.action in {"store", "update", "supersede", "conflict-review"}:
                        promoted += 1
        if promoted or action_counts:
            notes.append("Duplicate and conflicting semantic facts used lifecycle policy.")
        return ConsolidationReport(
            scanned_episodes=len(episodes),
            promoted_facts=promoted,
            ignored_low_salience=ignored,
            promoted_count=action_counts["store"] + action_counts["update"],
            merged_count=action_counts["merge"],
            skipped_count=action_counts["skip"],
            superseded_count=action_counts["supersede"],
            conflict_count=action_counts["conflict-review"],
            scanned_candidates=len(stored_candidates) + generated_candidate_count,
            notes=notes,
        )

    def _consolidatable_candidates(self, *, limit: int) -> list[MemoryCandidate]:
        return self.store.list_memory_candidates(
            statuses=["approved", "auto_approved", "pending"],
            limit=limit,
        )

    def _already_promoted(self, candidate: MemoryCandidate) -> bool:
        return any(
            decision.memory_id for decision in self.store.list_memory_decisions(candidate.id)
        )

    def _is_stable_repeated_candidate(
        self,
        candidate: MemoryCandidate,
        candidates: list[MemoryCandidate],
    ) -> bool:
        if candidate.candidate_type != "semantic":
            return False
        if candidate.confidence < 0.65 or _metadata_float(candidate.metadata, "stability") < 0.7:
            return False
        if _sensitivity_flags(candidate):
            return False
        key = _candidate_group_key(candidate)
        if key is None:
            return False
        peers = [
            peer
            for peer in candidates
            if peer.status == "pending"
            and peer.scope.scope_key == candidate.scope.scope_key
            and _candidate_group_key(peer) == key
            and not _sensitivity_flags(peer)
        ]
        return len(peers) >= 2

    def _auto_approve_candidate(self, candidate: MemoryCandidate) -> MemoryCandidate:
        approved = self.store.update_memory_candidate_status(
            candidate.id,
            "auto_approved",
            reviewer_metadata={"reviewer": "consolidation_service"},
        )
        self._record_candidate_decision(
            approved,
            action="auto_approve",
            memory_type=None,
            memory_id=None,
            rationale="auto-approved by consolidation policy",
        )
        return approved

    def _record_candidate_decision(
        self,
        candidate: MemoryCandidate,
        *,
        action: str,
        memory_type: str | None,
        memory_id: str | None,
        rationale: str,
    ) -> None:
        self.store.insert_memory_decision(
            MemoryDecision(
                id=new_id("decision"),
                candidate_id=candidate.id,
                action=action,
                memory_type=memory_type,
                memory_id=memory_id,
                reviewer="consolidation_service",
                rationale=rationale,
                created_at=utc_now_iso(),
                metadata={},
            )
        )

    def _promote_candidate_or_reject(
        self,
        candidate: MemoryCandidate,
        *,
        rationale: str,
    ) -> list[Any]:
        try:
            return self.controller.promote_candidate(
                candidate,
                action="promote",
                reviewer="consolidation_service",
                rationale=rationale,
            )
        except MemoryPromotionPolicyError as exc:
            rejected = self.store.update_memory_candidate_status(
                candidate.id,
                "rejected",
                reviewer_metadata={
                    "reviewer": "consolidation_service",
                    "policy_denial": exc.reason,
                },
            )
            self._record_candidate_decision(
                rejected,
                action="reject",
                memory_type=None,
                memory_id=None,
                rationale=f"blocked by extraction policy: {exc.reason}",
            )
            return []


def _candidate_group_key(candidate: MemoryCandidate) -> tuple[str, str, str, str] | None:
    weak = candidate.weak_structure
    subject = _optional_key(weak.get("subject"))
    predicate = _optional_key(weak.get("predicate"))
    object_value = _optional_key(weak.get("object"))
    if subject and predicate and object_value:
        return (candidate.scope.scope_key, subject, predicate, object_value)
    content = " ".join(candidate.content.casefold().split())
    return (candidate.scope.scope_key, "content", content, "") if content else None


def _optional_key(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).casefold().split())
    return normalized or None


def _metadata_float(metadata: dict[str, Any], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _sensitivity_flags(candidate: MemoryCandidate) -> list[str]:
    flags = candidate.metadata.get("sensitivity_flags")
    if not isinstance(flags, list):
        return []
    return [str(flag) for flag in flags if str(flag).strip()]
