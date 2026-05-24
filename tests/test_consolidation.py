from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import MemoryCandidate, MemoryScope
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.time import utc_now_iso


def _insert_approved_candidate(
    store: MemoryStore,
    *,
    candidate_id: str,
    content: str,
    subject: str,
    predicate: str,
    object_value: str,
    source_message_id: str,
    entities: list[str] | None = None,
    confidence: float = 0.9,
    salience: float = 0.85,
    stability: float = 0.9,
) -> None:
    now = utc_now_iso()
    weak_structure: dict[str, object] = {
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
    }
    if entities:
        weak_structure["entities"] = entities
    store.insert_memory_candidate(
        MemoryCandidate(
            id=candidate_id,
            candidate_type="semantic",
            proposed_layer="semantic",
            content=content,
            weak_structure=weak_structure,
            salience=salience,
            confidence=confidence,
            scope=MemoryScope.default(),
            source_message_ids=[source_message_id],
            status="approved",
            created_at=now,
            updated_at=now,
            metadata={"stability": stability},
        )
    )


def test_consolidation_promotes_explicit_durable_facts(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    episodic = EpisodicMemoryManager(store)
    episodic.create(
        content="User said: remember that my favorite color is blue",
        source_event_ids=["evt1"],
        salience=0.9,
        confidence=0.8,
    )

    report = ConsolidationService(store).consolidate()
    semantic = store.list_semantic_memories()

    assert report.promoted_facts == 1
    assert report.promoted_count == 1
    assert len(semantic) == 1
    assert semantic[0].subject == "user.favorite_color"
    assert semantic[0].object == "blue"


def test_consolidation_builds_source_backed_scene_persona_and_graph(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Remember that Project Atlas uses SQLite retrieval and I prefer terse diffs.",
    )
    _insert_approved_candidate(
        store,
        candidate_id="cand-project",
        content="Project Atlas uses SQLite retrieval",
        subject="Project Atlas",
        predicate="uses",
        object_value="SQLite retrieval",
        entities=["Project Atlas", "SQLite retrieval"],
        source_message_id=source.id,
    )
    _insert_approved_candidate(
        store,
        candidate_id="cand-preference",
        content="User prefers terse diffs",
        subject="user",
        predicate="prefers",
        object_value="terse diffs",
        entities=["user"],
        source_message_id=source.id,
        confidence=0.92,
        salience=0.82,
        stability=0.9,
    )

    report = ConsolidationService(store).consolidate()
    memories = store.list_semantic_memories(statuses=["active"], limit=20)
    project_fact = next(memory for memory in memories if memory.subject == "project atlas")
    preference = next(memory for memory in memories if memory.subject == "user")
    scenes = [memory for memory in memories if memory.memory_type == "scene"]
    personas = [memory for memory in memories if memory.memory_type == "persona"]
    scene_audit = store.drill_down_semantic_memory(scenes[0].id)
    persona_audit = store.drill_down_semantic_memory(personas[0].id)
    graph_edges = store.list_relation_edges()

    assert report.projected_scene_count >= 1
    assert report.projected_persona_count == 1
    assert scenes[0].source_memory_ids == [project_fact.id]
    assert personas[0].source_memory_ids == [preference.id]
    assert scenes[0].id not in personas[0].source_memory_ids
    assert [memory.id for memory in scene_audit.atomic_memories] == [project_fact.id]
    assert [message.id for message in scene_audit.source_messages] == [source.id]
    assert [memory.id for memory in persona_audit.atomic_memories] == [preference.id]
    assert [message.id for message in persona_audit.source_messages] == [source.id]
    assert graph_edges
    assert project_fact.id in graph_edges[0].evidence_memory_ids
    relation_audit = store.audit_relation_edges(
        source_name="Project Atlas",
        relation_type="uses",
    )
    assert [memory.id for memory in relation_audit[0].evidence_memories] == [
        project_fact.id
    ]
    assert [message.id for message in relation_audit[0].source_messages] == [source.id]


def test_consolidation_requires_reviewed_lineage_for_scene_persona_and_graph(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Project Atlas uses SQLite retrieval and I prefer terse diffs.",
    )
    semantic = SemanticMemoryManager(store)
    semantic.remember_atomic(
        content="Project Atlas uses SQLite retrieval",
        memory_type="fact",
        subject="Project Atlas",
        predicate="uses",
        object_value="SQLite retrieval",
        entities=["Project Atlas", "SQLite retrieval"],
        confidence=0.9,
        salience=0.85,
        stability=0.9,
        source_memory_ids=[source.id],
    )
    semantic.remember_atomic(
        content="User prefers terse diffs",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="terse diffs",
        entities=["user"],
        confidence=0.92,
        salience=0.82,
        stability=0.9,
        source_memory_ids=[source.id],
    )

    report = ConsolidationService(store).consolidate()

    assert report.projected_scene_count == 0
    assert report.projected_persona_count == 0
    assert report.graph_edge_count == 0
    assert [
        memory.memory_type
        for memory in store.list_semantic_memories(statuses=["active"], limit=10)
    ] == ["fact", "preference"]
    assert store.list_relation_edges() == []


def test_scene_projection_replaces_inactive_source_evidence(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    first_source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Remember that Project Atlas uses SQLite retrieval.",
    )
    second_source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Remember that Project Atlas uses FTS fallback.",
    )
    _insert_approved_candidate(
        store,
        candidate_id="cand-project-sqlite",
        content="Project Atlas uses SQLite retrieval",
        subject="Project Atlas",
        predicate="uses",
        object_value="SQLite retrieval",
        entities=["Project Atlas", "SQLite retrieval"],
        source_message_id=first_source.id,
    )
    _insert_approved_candidate(
        store,
        candidate_id="cand-project-fts",
        content="Project Atlas has FTS fallback",
        subject="Project Atlas",
        predicate="has",
        object_value="FTS fallback",
        entities=["Project Atlas", "FTS fallback"],
        source_message_id=second_source.id,
    )
    ConsolidationService(store).consolidate()
    active_before = store.list_semantic_memories(statuses=["active"], limit=20)
    sqlite_memory = next(memory for memory in active_before if memory.object == "sqlite retrieval")
    fts_memory = next(memory for memory in active_before if memory.object == "fts fallback")

    store.forget_semantic_memory(sqlite_memory.id, reason="test stale evidence")
    ConsolidationService(store).consolidate()
    active_after = store.list_semantic_memories(statuses=["active"], limit=20)
    scenes = [memory for memory in active_after if memory.memory_type == "scene"]
    superseded_scenes = [
        memory
        for memory in store.list_semantic_memories(statuses=["superseded"], limit=20)
        if memory.memory_type == "scene"
    ]
    scene_audit = store.drill_down_semantic_memory(scenes[0].id)
    relation_audit = store.audit_relation_edges(source_name="Project Atlas")

    assert scenes[0].source_memory_ids == [fts_memory.id]
    assert set(superseded_scenes[0].source_memory_ids) == {
        sqlite_memory.id,
        fts_memory.id,
    }
    assert superseded_scenes[0].superseded_by_id == scenes[0].id
    assert [memory.id for memory in scene_audit.atomic_memories] == [fts_memory.id]
    assert [message.id for message in scene_audit.source_messages] == [second_source.id]
    assert sqlite_memory.id not in scenes[0].source_memory_ids
    assert sqlite_memory.id not in [
        memory.id for memory in relation_audit[0].evidence_memories
    ]


def test_consolidation_projections_ignore_pending_and_rejected_candidates(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    for status in ["pending", "rejected"]:
        store.insert_memory_candidate(
            MemoryCandidate(
                id=f"cand-{status}",
                candidate_type="semantic",
                proposed_layer="semantic",
                content=f"User prefers {status} candidate only",
                weak_structure={
                    "subject": "user",
                    "predicate": "prefers",
                    "object": f"{status} candidate only",
                },
                salience=0.9,
                confidence=0.9,
                scope=MemoryScope.default(),
                source_message_ids=[f"msg-{status}"],
                status=status,
                created_at=now,
                updated_at=now,
                metadata={"stability": 0.95},
            )
        )

    report = ConsolidationService(store).consolidate()

    assert report.projected_persona_count == 0
    assert report.projected_scene_count == 0
    assert store.list_semantic_memories() == []


def test_consolidation_report_omits_working_memory_pruning(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    report = ConsolidationService(store).consolidate()

    assert "working memory" not in report.render()


def test_consolidation_promotes_stable_repeated_pending_candidates(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    scope = MemoryScope.default()
    for index in range(2):
        store.insert_memory_candidate(
            MemoryCandidate(
                id=f"cand-stable-{index}",
                candidate_type="semantic",
                proposed_layer="semantic",
                content="User prefers: tea",
                weak_structure={
                    "subject": "user",
                    "predicate": "prefers",
                    "object": "tea",
                },
                salience=0.82,
                confidence=0.78,
                scope=scope,
                source_message_ids=[f"msg-{index}"],
                status="pending",
                created_at=now,
                updated_at=now,
                metadata={"stability": 0.86},
            )
        )

    report = ConsolidationService(store).consolidate()
    semantic = store.list_semantic_memories()

    assert report.promoted_count == 1
    assert report.merged_count >= 1
    assert report.scanned_candidates == 2
    assert len(semantic) == 1
    assert set(semantic[0].source_memory_ids) == {"msg-0", "msg-1"}
    assert {
        candidate.status for candidate in store.list_memory_candidates(statuses=["auto_approved"])
    } == {"auto_approved"}


def test_consolidation_queues_conflict_review_for_low_confidence_candidate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    scope = MemoryScope.default()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approved-conflict",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers: coffee",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "coffee"},
            salience=0.72,
            confidence=0.58,
            scope=scope,
            source_message_ids=["msg-coffee"],
            status="approved",
            created_at=now,
            updated_at=now,
            metadata={"stability": 0.5},
        )
    )
    from alpha_agent.memory.semantic import SemanticMemoryManager

    SemanticMemoryManager(store).upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers: tea",
        scope=scope,
    )

    report = ConsolidationService(store).consolidate()

    assert report.conflict_count == 1
    assert [memory.object for memory in store.list_semantic_memories(statuses=["active"])] == [
        "tea"
    ]
    assert [
        memory.object
        for memory in store.list_semantic_memories(statuses=["conflict_review"])
    ] == ["coffee"]


def test_consolidation_episode_promotion_respects_sensitive_policy(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    EpisodicMemoryManager(store).create(
        content="User said: remember that my password is swordfish",
        source_event_ids=["evt-secret"],
        salience=0.9,
        confidence=0.8,
    )

    report = ConsolidationService(store).consolidate()

    assert report.promoted_facts == 0
    assert store.list_semantic_memories() == []
    assert store.list_memory_candidates() == []


def test_consolidation_episode_promotion_respects_do_not_remember_policy(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    EpisodicMemoryManager(store).create(
        content="User said: do not remember that I prefer tea",
        source_event_ids=["evt-private"],
        salience=0.9,
        confidence=0.8,
    )

    ConsolidationService(store).consolidate()

    assert store.list_semantic_memories() == []
    assert store.list_memory_candidates() == []


def test_consolidation_episode_promotion_respects_group_write_policy(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    group_scope = MemoryScope.from_source_metadata(
        session_id="s1",
        source_metadata={
            "platform": "slack",
            "chat_id": "team",
            "chat_type": "group",
            "user_id": "u1",
        },
    )
    EpisodicMemoryManager(store).create(
        content="User said: I prefer tea",
        source_event_ids=["evt-group"],
        salience=0.9,
        confidence=0.8,
        scope=group_scope,
    )

    ConsolidationService(store).consolidate()

    assert store.list_semantic_memories() == []
    assert store.list_memory_candidates() == []


def test_consolidation_episode_promotion_respects_system_source_policy(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="remember that I prefer tea",
        source_metadata={"is_system_message": True},
    )
    EpisodicMemoryManager(store).create(
        content="User said: remember that I prefer tea",
        source_event_ids=[source.id],
        salience=0.9,
        confidence=0.8,
    )

    ConsolidationService(store).consolidate()

    assert store.list_semantic_memories() == []
    assert store.list_memory_candidates() == []


def test_consolidation_refuses_approved_sensitive_candidate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approved-secret",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User password is swordfish",
            weak_structure={"subject": "user", "predicate": "password", "object": "swordfish"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[],
            status="approved",
            created_at=now,
            updated_at=now,
        )
    )

    ConsolidationService(store).consolidate()

    rejected = store.get_memory_candidate("cand-approved-secret")
    assert rejected is not None
    assert rejected.status == "rejected"
    assert store.list_semantic_memories() == []
    assert [decision.action for decision in store.list_memory_decisions(rejected.id)] == [
        "reject"
    ]


def test_consolidation_refuses_auto_approved_system_candidate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="remember that I prefer tea",
        source_metadata={"is_system_message": True},
    )
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approved-system",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers: tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[source.id],
            status="auto_approved",
            created_at=now,
            updated_at=now,
        )
    )

    ConsolidationService(store).consolidate()

    rejected = store.get_memory_candidate("cand-approved-system")
    assert rejected is not None
    assert rejected.status == "rejected"
    assert store.list_semantic_memories() == []


def test_consolidation_refuses_approved_group_ambient_candidate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    group_scope = MemoryScope.from_source_metadata(
        session_id="s1",
        source_metadata={
            "platform": "slack",
            "chat_id": "team",
            "chat_type": "group",
            "user_id": "u1",
        },
    )
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="I prefer tea",
    )
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approved-group",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers: tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=group_scope,
            source_message_ids=[source.id],
            status="approved",
            created_at=now,
            updated_at=now,
        )
    )

    ConsolidationService(store).consolidate()

    rejected = store.get_memory_candidate("cand-approved-group")
    assert rejected is not None
    assert rejected.status == "rejected"
    assert store.list_semantic_memories() == []


def test_consolidation_refuses_approved_do_not_remember_candidate(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="do not remember that I prefer tea",
    )
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approved-private",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers: tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[source.id],
            status="approved",
            created_at=now,
            updated_at=now,
        )
    )

    ConsolidationService(store).consolidate()

    rejected = store.get_memory_candidate("cand-approved-private")
    assert rejected is not None
    assert rejected.status == "rejected"
    assert store.list_semantic_memories() == []
