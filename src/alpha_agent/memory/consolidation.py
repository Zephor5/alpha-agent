"""Manual memory consolidation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.graph.store import GraphStore
from alpha_agent.memory.controller import MemoryController, MemoryPromotionPolicyError
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import MemoryCandidate, MemoryDecision, MemoryScope, SemanticMemory
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
    projected_scene_count: int = 0
    projected_persona_count: int = 0
    graph_edge_count: int = 0
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
            f"- scene projections: {self.projected_scene_count}",
            f"- persona projections: {self.projected_persona_count}",
            f"- graph edges updated: {self.graph_edge_count}",
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
        scene_count = self.update_scene_memories()
        persona_count = self.update_persona_memory()
        graph_edge_count = self.update_graph_index()
        if promoted or action_counts:
            notes.append("Duplicate and conflicting semantic facts used lifecycle policy.")
        if scene_count or persona_count:
            notes.append("Scene and persona projections are source-backed semantic memories.")
        if graph_edge_count:
            notes.append("Graph edges were updated only for source-backed non-user facts.")
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
            projected_scene_count=scene_count,
            projected_persona_count=persona_count,
            graph_edge_count=graph_edge_count,
            notes=notes,
        )

    def update_scene_memories(self, *, limit: int = 100) -> int:
        """Build source-backed topic/project summaries from active atomic memories."""

        active = self._active_reviewed_atomic_semantic(limit=limit)
        grouped: dict[tuple[str, str], list[SemanticMemory]] = defaultdict(list)
        for memory in active:
            if not _has_source_messages(self.store, memory):
                continue
            topic = _scene_topic(memory)
            if topic is None:
                continue
            grouped[(memory.scope.scope_key, topic)].append(memory)

        projected = 0
        active_projection_keys: set[tuple[str, str, str, str, str]] = set()
        for (_, topic), memories in grouped.items():
            scope = memories[0].scope
            key = _projection_key(
                scope=scope,
                memory_type="scene",
                subject=f"scene:{topic.casefold()}",
                predicate="summarizes",
                object_value=topic,
            )
            active_projection_keys.add(key)
            source_memory_ids = [memory.id for memory in memories]
            source_message_ids = _source_message_ids(self.store, memories)
            if not source_message_ids:
                continue
            if self._projection_sources_unchanged(
                memory_type="scene",
                scope=scope,
                source_memory_ids=source_memory_ids,
                source_signature=_source_signature(memories),
            ):
                continue
            content = _scene_summary(topic, memories)
            source_signature = _source_signature(memories)
            stored = self._upsert_projection_memory(
                content=content,
                memory_type="scene",
                subject=f"scene:{topic.casefold()}",
                predicate="summarizes",
                object_value=topic,
                entities=_dedupe(
                    [topic, *[entity for memory in memories for entity in memory.entities]]
                ),
                confidence=min(memory.confidence for memory in memories),
                salience=max(memory.salience for memory in memories),
                stability=min(memory.stability for memory in memories),
                source_memory_ids=source_memory_ids,
                scope=scope,
                metadata={
                    "projection_layer": "scene",
                    "source_message_ids": source_message_ids,
                    "source_count": len(source_memory_ids),
                    "source_signature": source_signature,
                },
            )
            if stored is not None:
                projected += 1
        self._retire_stale_projections(
            memory_type="scene",
            active_keys=active_projection_keys,
        )
        return projected

    def update_persona_memory(self, *, limit: int = 100) -> int:
        """Build low-frequency profile projection from reviewed active memories."""

        active = self._active_reviewed_atomic_semantic(limit=limit)
        profile_sources = [
            memory
            for memory in active
            if _is_persona_source(memory) and _has_source_messages(self.store, memory)
        ]
        if not profile_sources:
            self._retire_stale_projections(memory_type="persona", active_keys=set())
            return 0
        scene_context = [
            memory
            for memory in self.store.list_semantic_memories(
                limit=limit,
                statuses=["active"],
            )
            if memory.memory_type == "scene"
            and memory.scope.scope_key in {source.scope.scope_key for source in profile_sources}
        ]
        by_scope: dict[str, list[SemanticMemory]] = defaultdict(list)
        scene_context_by_scope: dict[str, list[SemanticMemory]] = defaultdict(list)
        for memory in profile_sources:
            by_scope[memory.scope.scope_key].append(memory)
        for memory in scene_context:
            scene_context_by_scope[memory.scope.scope_key].append(memory)

        projected = 0
        active_projection_keys: set[tuple[str, str, str, str, str]] = set()
        for memories in by_scope.values():
            scope = memories[0].scope
            scenes = scene_context_by_scope.get(scope.scope_key, [])
            key = _projection_key(
                scope=scope,
                memory_type="persona",
                subject="user.profile",
                predicate="summarizes",
                object_value="stable reviewed preferences",
            )
            active_projection_keys.add(key)
            source_memory_ids = [memory.id for memory in memories]
            source_message_ids = _source_message_ids(self.store, memories)
            if not source_message_ids:
                continue
            if self._projection_sources_unchanged(
                memory_type="persona",
                scope=scope,
                source_memory_ids=source_memory_ids,
                source_signature=_source_signature(memories),
            ):
                continue
            content = _persona_summary(memories, scene_context=scenes)
            source_signature = _source_signature(memories)
            stored = self._upsert_projection_memory(
                content=content,
                memory_type="persona",
                subject="user.profile",
                predicate="summarizes",
                object_value="stable reviewed preferences",
                entities=_dedupe([entity for memory in memories for entity in memory.entities]),
                confidence=min(memory.confidence for memory in memories),
                salience=max(memory.salience for memory in memories),
                stability=min(memory.stability for memory in memories),
                source_memory_ids=source_memory_ids,
                scope=scope,
                metadata={
                    "projection_layer": "persona",
                    "source_message_ids": source_message_ids,
                    "source_count": len(source_memory_ids),
                    "source_signature": source_signature,
                    "current_request_priority": "explicit_current_user_request_wins",
                },
            )
            if stored is not None:
                projected += 1
        self._retire_stale_projections(
            memory_type="persona",
            active_keys=active_projection_keys,
        )
        return projected

    def update_graph_index(self, *, limit: int = 100) -> int:
        """Index source-backed non-user fact relations for audit."""

        graph = GraphStore(self.store.db_path)
        count = 0
        for memory in self._active_reviewed_atomic_semantic(limit=limit):
            if not _graph_worthy(memory) or not _has_source_messages(self.store, memory):
                continue
            source = graph.upsert_node(
                memory.subject or "",
                kind="entity",
                aliases=[],
                salience=memory.salience,
                metadata={"evidence_memory_ids": [memory.id]},
            )
            target = graph.upsert_node(
                memory.object or "",
                kind="entity",
                aliases=[],
                salience=memory.salience,
                metadata={"evidence_memory_ids": [memory.id]},
            )
            graph.add_edge(
                source.id,
                target.id,
                relation_type=memory.predicate or "related_to",
                evidence_memory_ids=[memory.id],
                confidence=memory.confidence,
                metadata={
                    "purpose": "audit",
                    "source_message_ids": list(memory.source_memory_ids),
                },
            )
            count += 1
        return count

    def _active_reviewed_atomic_semantic(self, *, limit: int) -> list[SemanticMemory]:
        reviewed_ids = self.store.list_reviewed_semantic_memory_ids()
        return [
            memory
            for memory in self.store.list_semantic_memories(
                limit=limit,
                statuses=["active"],
            )
            if memory.memory_type not in {"scene", "persona"}
            and memory.id in reviewed_ids
        ]

    def _upsert_projection_memory(
        self,
        *,
        content: str,
        memory_type: str,
        subject: str,
        predicate: str,
        object_value: str,
        entities: list[str],
        confidence: float,
        salience: float,
        stability: float,
        source_memory_ids: list[str],
        scope: MemoryScope,
        metadata: dict[str, Any],
    ) -> SemanticMemory | None:
        existing = self._active_projection(
            memory_type=memory_type,
            scope=scope,
            subject=subject,
            predicate=predicate,
            object_value=object_value,
        )
        source_signature = metadata.get("source_signature")
        if (
            existing is not None
            and existing.source_memory_ids == source_memory_ids
            and existing.metadata.get("source_signature") == source_signature
        ):
            return None
        supersedes_id = existing.id if existing is not None else None
        projection_id = new_id("sem")
        if existing is not None:
            self._retire_projection(
                existing,
                reason="projection source evidence changed",
                superseded_by_id=projection_id,
            )
        now = utc_now_iso()
        return self.store.upsert_semantic_memory(
            SemanticMemory(
                id=projection_id,
                content=content,
                memory_type=memory_type,
                subject=subject,
                predicate=predicate,
                object=object_value,
                entities=entities,
                confidence=confidence,
                salience=salience,
                stability=stability,
                source_memory_ids=source_memory_ids,
                created_at=now,
                updated_at=now,
                metadata=metadata,
                status="active",
                valid_from=now,
                supersedes_id=supersedes_id,
                scope=scope,
            )
        )

    def _active_projection(
        self,
        *,
        memory_type: str,
        scope: MemoryScope,
        subject: str,
        predicate: str,
        object_value: str,
    ) -> SemanticMemory | None:
        for memory in self.store.list_semantic_memories(
            limit=100,
            scopes=[scope],
            statuses=["active"],
        ):
            if (
                memory.memory_type == memory_type
                and memory.subject == subject
                and memory.predicate == predicate
                and memory.object == object_value
            ):
                return memory
        return None

    def _retire_stale_projections(
        self,
        *,
        memory_type: str,
        active_keys: set[tuple[str, str, str, str, str]],
    ) -> None:
        for memory in self.store.list_semantic_memories(
            limit=200,
            statuses=["active"],
        ):
            if memory.memory_type != memory_type:
                continue
            key = _projection_key(
                scope=memory.scope,
                memory_type=memory.memory_type,
                subject=memory.subject or "",
                predicate=memory.predicate or "",
                object_value=memory.object or "",
            )
            if key not in active_keys:
                self._retire_projection(memory, reason="projection has no active reviewed evidence")

    def _retire_projection(
        self,
        memory: SemanticMemory,
        *,
        reason: str,
        superseded_by_id: str | None = None,
    ) -> SemanticMemory:
        now = utc_now_iso()
        metadata = dict(memory.metadata)
        metadata["retire_reason"] = reason
        return self.store.upsert_semantic_memory(
            SemanticMemory(
                id=memory.id,
                content=memory.content,
                memory_type=memory.memory_type,
                subject=memory.subject,
                predicate=memory.predicate,
                object=memory.object,
                entities=list(memory.entities),
                confidence=memory.confidence,
                salience=memory.salience,
                stability=memory.stability,
                source_memory_ids=list(memory.source_memory_ids),
                created_at=memory.created_at,
                updated_at=now,
                metadata=metadata,
                status="superseded",
                valid_from=memory.valid_from,
                valid_until=now,
                supersedes_id=memory.supersedes_id,
                superseded_by_id=superseded_by_id or memory.superseded_by_id,
                scope=memory.scope,
            )
        )

    def _projection_sources_unchanged(
        self,
        *,
        memory_type: str,
        scope: MemoryScope,
        source_memory_ids: list[str],
        source_signature: list[dict[str, str]],
    ) -> bool:
        wanted = set(source_memory_ids)
        for memory in self.store.list_semantic_memories(
            limit=20,
            scopes=[scope],
            statuses=["active"],
        ):
            if (
                memory.memory_type == memory_type
                and set(memory.source_memory_ids) == wanted
                and memory.metadata.get("source_signature") == source_signature
            ):
                return True
        return False

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


def _scene_topic(memory: SemanticMemory) -> str | None:
    if memory.subject and memory.subject != "user":
        return _display_value(memory.subject)
    for entity in memory.entities:
        if entity.casefold() != "user":
            return _display_value(entity)
    return None


def _scene_summary(topic: str, memories: list[SemanticMemory]) -> str:
    facts = "; ".join(_display_value(memory.content) for memory in memories[:5])
    return f"{topic}: {facts}"


def _persona_summary(
    memories: list[SemanticMemory],
    *,
    scene_context: list[SemanticMemory] | None = None,
) -> str:
    facts = "; ".join(_display_value(memory.content) for memory in memories[:6])
    scenes = "; ".join(
        _display_value(memory.content) for memory in (scene_context or [])[:3]
    )
    if scenes:
        return f"Stable user profile projection: {facts}. Related scene context: {scenes}"
    return f"Stable user profile projection: {facts}"


def _is_persona_source(memory: SemanticMemory) -> bool:
    if memory.memory_type not in {"preference", "profile"}:
        return False
    if memory.status != "active":
        return False
    return memory.confidence >= 0.85 and memory.stability >= 0.85


def _graph_worthy(memory: SemanticMemory) -> bool:
    if not memory.subject or not memory.predicate or not memory.object:
        return False
    if memory.subject.casefold() == "user":
        return False
    return memory.confidence >= 0.7


def _has_source_messages(store: MemoryStore, memory: SemanticMemory) -> bool:
    return bool(store.list_conversation_messages_by_ids(_candidate_source_message_ids(memory)))


def _source_message_ids(
    store: MemoryStore,
    memories: list[SemanticMemory],
) -> list[str]:
    ids: list[str] = []
    for memory in memories:
        if memory.memory_type in {"scene", "persona"}:
            for message_id in _metadata_source_message_ids(memory):
                _append_id(ids, message_id)
            for drill_memory in store.drill_down_semantic_memory(memory.id).source_messages:
                _append_id(ids, drill_memory.id)
            continue
        for message_id in _candidate_source_message_ids(memory):
            _append_id(ids, message_id)
    existing = store.list_conversation_messages_by_ids(ids)
    return [message.id for message in existing]


def _source_signature(memories: list[SemanticMemory]) -> list[dict[str, str]]:
    return [
        {
            "id": memory.id,
            "updated_at": memory.updated_at,
            "content": memory.content,
        }
        for memory in memories
    ]


def _projection_key(
    *,
    scope: MemoryScope,
    memory_type: str,
    subject: str,
    predicate: str,
    object_value: str,
) -> tuple[str, str, str, str, str]:
    return (
        scope.scope_key,
        memory_type,
        _display_value(subject).casefold(),
        _display_value(predicate).casefold(),
        _display_value(object_value).casefold(),
    )


def _candidate_source_message_ids(memory: SemanticMemory) -> list[str]:
    return _dedupe([*memory.source_memory_ids, *_metadata_source_message_ids(memory)])


def _metadata_source_message_ids(memory: SemanticMemory) -> list[str]:
    value = memory.metadata.get("source_message_ids")
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        _append_id(result, value)
    return result


def _append_id(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _display_value(value: str) -> str:
    return " ".join(value.strip().split())
