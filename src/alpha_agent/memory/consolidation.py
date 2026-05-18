"""Manual memory consolidation."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class ConsolidationReport:
    """Summary of a manual consolidation run."""

    scanned_episodes: int
    promoted_facts: int
    ignored_low_salience: int
    pruned_working_memory: int
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        """Return a short report suitable for CLI output."""

        lines = [
            "Consolidation complete",
            f"- scanned episodes: {self.scanned_episodes}",
            f"- promoted facts: {self.promoted_facts}",
            f"- ignored low-salience episodes: {self.ignored_low_salience}",
            f"- pruned low-priority working memory: {self.pruned_working_memory}",
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

    def consolidate(self, limit: int = 100) -> ConsolidationReport:
        """Run deterministic manual consolidation."""

        episodes = self.store.list_episodic_memories(limit=limit)
        promoted = 0
        ignored = 0
        pruned_working_memory = self.store.prune_low_priority_working_memory()
        notes: list[str] = []
        for episode in episodes:
            if episode.salience < 0.65:
                ignored += 1
                continue
            candidates = self.extractor.extract(
                user_message=episode.content,
                assistant_response="",
                source_event_ids=episode.source_event_ids,
            )
            for candidate in candidates:
                if candidate.type != "semantic":
                    continue
                if not candidate.subject or not candidate.predicate or not candidate.object:
                    continue
                self.semantic_manager.upsert_fact(
                    subject=candidate.subject,
                    predicate=candidate.predicate,
                    object_value=candidate.object,
                    content=candidate.content,
                    confidence=max(candidate.confidence, episode.confidence),
                    salience=max(candidate.salience, episode.salience),
                    source_memory_ids=[episode.id],
                )
                promoted += 1
        if promoted:
            notes.append("Duplicate semantic facts were merged by subject/predicate/object.")
        return ConsolidationReport(
            scanned_episodes=len(episodes),
            promoted_facts=promoted,
            ignored_low_salience=ignored,
            pruned_working_memory=pruned_working_memory,
            notes=notes,
        )
