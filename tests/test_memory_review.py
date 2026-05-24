from __future__ import annotations

from pathlib import Path

import pytest

import alpha_agent.memory.controller as controller_module
from alpha_agent.memory.controller import MemoryController
from alpha_agent.memory.models import MemoryCandidate, MemoryScope
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.review import MemoryReviewService
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.time import utc_now_iso


def test_one_shot_reject_stores_auditable_candidates(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    service = MemoryReviewService(store)
    candidates = service.preview("remember that I prefer tea")

    rejected = service.reject(
        message="remember that I prefer tea",
        session_id="session-review",
        candidates=candidates,
        reviewer="cli",
    )

    assert len(rejected) == 2
    assert [candidate.status for candidate in rejected] == ["rejected", "rejected"]
    assert store.list_semantic_memories() == []
    assert store.list_episodic_memories() == []
    for candidate in rejected:
        audit = service.inspect_stored(candidate.id)
        assert [decision.action for decision in audit.decisions] == ["reject"]
        assert [message.raw_content for message in audit.source_messages] == [
            "remember that I prefer tea"
        ]


def test_one_shot_approve_rolls_back_candidate_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    service = MemoryReviewService(store)
    candidates = service.preview("remember that I prefer tea")

    def fail_persist(*args: object, **kwargs: object) -> object:
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(controller_module, "persist_candidates", fail_persist)

    with pytest.raises(RuntimeError, match="promotion failed"):
        service.approve(
            message="remember that I prefer tea",
            session_id="session-review",
            candidates=candidates[:1],
        )

    assert store.list_memory_candidates() == []
    assert store.list_conversation_messages("session-review") == []
    assert store.stats()["memory_decisions"] == 0
    assert store.list_semantic_memories() == []


def test_stored_approve_rolls_back_status_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-approve-rollback",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=["msg-source"],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    service = MemoryReviewService(store)

    def fail_persist(*args: object, **kwargs: object) -> object:
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(controller_module, "persist_candidates", fail_persist)

    with pytest.raises(RuntimeError, match="promotion failed"):
        service.approve_stored("cand-approve-rollback")

    candidate = store.get_memory_candidate("cand-approve-rollback")
    assert candidate is not None
    assert candidate.status == "pending"
    assert store.list_memory_decisions("cand-approve-rollback") == []
    assert store.list_semantic_memories() == []


def test_stored_approve_rejects_unpromotable_candidate_without_terminal_status(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-unpromotable",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers tea",
            weak_structure={"subject": None, "predicate": None, "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=["msg-source"],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    service = MemoryReviewService(store)

    with pytest.raises(ValueError, match="cannot be promoted"):
        service.approve_stored("cand-unpromotable")

    candidate = store.get_memory_candidate("cand-unpromotable")
    assert candidate is not None
    assert candidate.status == "pending"
    assert store.list_memory_decisions("cand-unpromotable") == []
    assert store.list_semantic_memories() == []


def test_runtime_auto_approve_rolls_back_status_when_promotion_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-auto-rollback",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers tea",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=["msg-source"],
            status="pending",
            created_at=now,
            updated_at=now,
            metadata={"extractor": "explicit_or_correction"},
        )
    )
    candidate = store.get_memory_candidate("cand-auto-rollback")
    assert candidate is not None
    controller = MemoryController(store, retriever=MemoryRetriever(store))

    def fail_persist(*args: object, **kwargs: object) -> object:
        raise RuntimeError("promotion failed")

    monkeypatch.setattr(controller_module, "persist_candidates", fail_persist)

    with pytest.raises(RuntimeError, match="promotion failed"):
        controller.decide_runtime_candidates(
            session_id="session-runtime",
            candidates=[candidate],
        )

    rolled_back = store.get_memory_candidate("cand-auto-rollback")
    assert rolled_back is not None
    assert rolled_back.status == "pending"
    assert store.list_memory_decisions("cand-auto-rollback") == []
    assert store.list_semantic_memories() == []


def test_stored_candidate_edit_preserves_sources_and_records_decision(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Remember that I prefer coffee",
    )
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-edit",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers coffee",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "coffee"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[source.id],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    service = MemoryReviewService(store)

    edited = service.edit_stored(
        "cand-edit",
        content="User prefers tea",
        object_value="tea",
        reviewer="cli",
    )
    persisted = service.approve_stored("cand-edit", reviewer="cli")
    audit = service.inspect_stored("cand-edit")

    assert edited.status == "edited"
    assert edited.source_message_ids == [source.id]
    assert edited.content == "User prefers tea"
    assert edited.weak_structure["object"] == "tea"
    assert persisted[0].memory_type == "semantic"
    assert audit.candidate.source_message_ids == [source.id]
    assert [message.raw_content for message in audit.source_messages] == [
        "Remember that I prefer coffee"
    ]
    assert [decision.action for decision in audit.decisions] == ["edit", "approve", "store"]
    edit_decision = audit.decisions[0]
    assert edit_decision.metadata["original_content"] == "User prefers coffee"
    assert edit_decision.metadata["edited_content"] == "User prefers tea"
    assert edit_decision.metadata["source_message_ids"] == [source.id]
    semantic = store.list_semantic_memories()[0]
    assert semantic.content == "User prefers tea"
    assert semantic.source_memory_ids == [source.id]


def test_candidate_correction_supersedes_active_memory_and_audit_links_sources(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    new_source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Actually I prefer coffee",
    )
    service = MemoryReviewService(store)
    first = service.approve(
        message="Remember that I prefer tea",
        session_id="review-1",
        candidates=service.preview("remember that I prefer tea")[:1],
    )
    assert first[0].memory_type == "semantic"
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-correction",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers coffee",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "coffee"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=[new_source.id],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )

    persisted = service.approve_stored("cand-correction", reviewer="cli")
    audit = service.inspect_memory(persisted[0].memory_id)

    active = store.list_semantic_memories(statuses=["active"])
    superseded = store.list_semantic_memories(statuses=["superseded"])
    assert persisted[0].action == "supersede"
    assert [memory.object for memory in active] == ["coffee"]
    assert [memory.object for memory in superseded] == ["tea"]
    assert audit.memory.id == active[0].id
    assert [memory.id for memory in audit.supersession_chain] == [superseded[0].id]
    assert new_source.id in audit.source_message_ids


def test_audit_old_superseded_memory_shows_active_replacement(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    old = semantic.upsert_fact(
        "user",
        "prefers",
        "tea",
        "User prefers tea",
        source_memory_ids=["msg-old"],
    )
    new = semantic.upsert_fact(
        "user",
        "prefers",
        "coffee",
        "User prefers coffee",
        source_memory_ids=["msg-new"],
    )
    service = MemoryReviewService(store)

    audit = service.inspect_memory(old.id)

    assert audit.memory.id == old.id
    assert audit.memory.status == "superseded"
    assert [memory.id for memory in audit.supersession_chain] == [new.id]
    assert audit.source_message_ids == ["msg-old", "msg-new"]


def test_reject_edited_candidate_keeps_edit_audit_without_promotion(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    store.insert_memory_candidate(
        MemoryCandidate(
            id="cand-reject-edited",
            candidate_type="semantic",
            proposed_layer="semantic",
            content="User prefers coffee",
            weak_structure={"subject": "user", "predicate": "prefers", "object": "coffee"},
            salience=0.9,
            confidence=0.8,
            scope=MemoryScope.default(),
            source_message_ids=["msg-source"],
            status="pending",
            created_at=now,
            updated_at=now,
        )
    )
    service = MemoryReviewService(store)

    service.edit_stored("cand-reject-edited", content="User prefers tea", reviewer="cli")
    service.reject_stored("cand-reject-edited", reviewer="cli")
    audit = service.inspect_stored("cand-reject-edited")

    assert audit.candidate.status == "rejected"
    assert [decision.action for decision in audit.decisions] == ["edit", "reject"]
    assert store.list_semantic_memories() == []
