from __future__ import annotations

from pathlib import Path

from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.models import MemoryCandidate, MemoryScope
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.time import utc_now_iso


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
