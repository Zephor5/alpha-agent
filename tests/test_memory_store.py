from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alpha_agent.memory.models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryScope,
    ProceduralMemory,
    SessionContextState,
)
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.utils.time import utc_now_iso


def test_database_initialization(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    stats = store.stats()

    assert stats["conversation_messages"] == 0
    assert stats["session_context_states"] == 0
    assert stats["runtime_traces"] == 0
    assert stats["episodic"] == 0
    assert stats["semantic"] == 0
    assert "working_memory" not in stats
    with store.connect() as conn:
        table_names = {
            str(row["name"])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "events" not in table_names
    assert "working_memory" not in table_names


def test_append_and_list_conversation_messages_by_ordinal(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    first = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="raw hello",
        model_content="expanded hello",
        source_metadata={"platform": "telegram", "message_id": "m1"},
        metadata={"source": "test"},
    )
    second = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="I will call a tool.",
        tool_calls=[{"id": "call_1", "name": "lookup", "arguments": {"q": "alpha"}}],
        provider_metadata={"provider": "mock", "model": "mock-1"},
    )
    third = store.append_conversation_message(
        session_id="s1",
        role="tool",
        raw_content='{"answer": 42}',
        tool_call_id="call_1",
        tool_result_id="result_1",
    )

    before_third = store.list_conversation_messages("s1", before_ordinal=third.ordinal)
    after_first = store.list_conversation_messages("s1", after_ordinal=first.ordinal)

    assert [message.ordinal for message in [first, second, third]] == [1, 2, 3]
    assert [message.id for message in before_third] == [first.id, second.id]
    assert [message.id for message in after_first] == [second.id, third.id]
    assert before_third[0].model_content == "expanded hello"
    assert before_third[0].source_metadata == {"platform": "telegram", "message_id": "m1"}
    assert before_third[1].tool_calls == [
        {"id": "call_1", "name": "lookup", "arguments": {"q": "alpha"}}
    ]
    assert before_third[1].provider_metadata == {"provider": "mock", "model": "mock-1"}
    assert after_first[1].tool_call_id == "call_1"
    assert after_first[1].tool_result_id == "result_1"


def test_concurrent_conversation_appends_assign_unique_ordinals(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    def append_message(index: int) -> int:
        message = store.append_conversation_message(
            session_id="s1",
            role="user",
            raw_content=f"message {index}",
        )
        return message.ordinal

    with ThreadPoolExecutor(max_workers=8) as executor:
        ordinals = list(executor.map(append_message, range(20)))

    messages = store.list_conversation_messages("s1")

    assert sorted(ordinals) == list(range(1, 21))
    assert [message.ordinal for message in messages] == list(range(1, 21))


def test_upsert_session_context_state_keeps_one_active_state(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    created_at = utc_now_iso()

    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=2,
            summary="Earlier context.",
            summary_source_message_ids=["msg_1", "msg_2"],
            compression_version="v1",
            created_at=created_at,
            updated_at=created_at,
            metadata={"token_count": 128},
        )
    )
    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=4,
            summary="Updated context.",
            summary_source_message_ids=["msg_1", "msg_2", "msg_3", "msg_4"],
            compression_version="v2",
            created_at=utc_now_iso(),
            updated_at=utc_now_iso(),
            metadata={"reason": "threshold"},
        )
    )

    state = store.get_session_context_state("s1")
    stats = store.stats()

    assert state is not None
    assert state.compressed_until_ordinal == 4
    assert state.summary == "Updated context."
    assert state.summary_source_message_ids == ["msg_1", "msg_2", "msg_3", "msg_4"]
    assert state.compression_version == "v2"
    assert state.created_at == created_at
    assert state.metadata == {"reason": "threshold"}
    assert stats["session_context_states"] == 1


def test_upsert_session_context_state_rejects_backward_compression(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    created_at = utc_now_iso()
    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=4,
            summary="Updated context.",
            summary_source_message_ids=["msg_1", "msg_2", "msg_3", "msg_4"],
            compression_version="v2",
            created_at=created_at,
            updated_at=created_at,
        )
    )

    with pytest.raises(ValueError, match="cannot move backward"):
        store.upsert_session_context_state(
            SessionContextState(
                session_id="s1",
                compressed_until_ordinal=3,
                summary="Older context.",
                summary_source_message_ids=["msg_1", "msg_2", "msg_3"],
                compression_version="v1",
                created_at=utc_now_iso(),
                updated_at=utc_now_iso(),
            )
        )

    state = store.get_session_context_state("s1")
    assert state is not None
    assert state.compressed_until_ordinal == 4
    assert state.summary == "Updated context."


def test_append_and_list_runtime_traces(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()

    trace = store.append_runtime_trace(
        session_id="s1",
        event_type="llm.completed",
        content="LLM call completed.",
        metadata={"provider": "mock", "retry_count": 0},
    )

    traces = store.list_runtime_traces(session_id="s1")

    assert traces == [trace]
    assert traces[0].event_type == "llm.completed"
    assert traces[0].metadata == {"provider": "mock", "retry_count": 0}


def test_semantic_upsert_merges_same_fact(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    first = semantic.upsert_fact(
        "user",
        "prefers",
        "concise answers",
        "User prefers concise answers",
        source_memory_ids=["msg-1"],
    )
    second = semantic.upsert_fact(
        "User",
        "Prefers",
        "Concise Answers",
        "Updated content",
        source_memory_ids=["msg-2"],
    )

    memories = store.list_semantic_memories()
    assert len(memories) == 1
    assert first.id == second.id
    assert memories[0].content == "Updated content"
    assert memories[0].source_memory_ids == ["msg-1", "msg-2"]


def test_semantic_lifecycle_reports_update_and_skip_actions(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    first = semantic.remember_atomic(
        content="User prefers tea",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="tea",
        source_memory_ids=["msg-1"],
    )
    updated = semantic.remember_atomic(
        content="User strongly prefers tea",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="tea",
        source_memory_ids=["msg-2"],
    )
    skipped = semantic.remember_atomic(
        content="User strongly prefers tea",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="tea",
        source_memory_ids=["msg-1", "msg-2"],
    )

    assert first.action == "store"
    assert updated.action == "update"
    assert skipped.action == "skip"
    assert updated.memory.id == skipped.memory.id


def test_semantic_normalized_content_duplicate_merges_sources(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    first = semantic.remember_atomic(
        content="User prefers green tea.",
        memory_type="preference",
        confidence=0.8,
        salience=0.9,
        source_memory_ids=["msg-1"],
    )
    second = semantic.remember_atomic(
        content=" user prefers GREEN tea ",
        memory_type="preference",
        confidence=0.7,
        salience=0.6,
        source_memory_ids=["msg-2"],
    )

    memories = store.list_semantic_memories(statuses=["active"])
    assert len(memories) == 1
    assert second.action == "merge"
    assert first.memory.id == second.memory.id
    assert memories[0].source_memory_ids == ["msg-1", "msg-2"]


def test_semantic_changed_object_supersedes_old_active_memory(tmp_path: Path) -> None:
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
        "User now prefers coffee",
        source_memory_ids=["msg-new"],
    )

    active = store.list_semantic_memories(statuses=["active"])
    superseded = store.list_semantic_memories(statuses=["superseded"])

    assert [memory.id for memory in active] == [new.id]
    assert [memory.id for memory in superseded] == [old.id]
    assert active[0].supersedes_id == old.id
    assert superseded[0].superseded_by_id == new.id
    assert active[0].source_memory_ids == ["msg-new", "msg-old"]


def test_semantic_similar_changed_object_supersedes_instead_of_merging_stale_object(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    old = semantic.remember_atomic(
        content="User strongly prefers dark roast tea every morning",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="dark roast tea",
        confidence=0.8,
        salience=0.9,
        source_memory_ids=["msg-old"],
    )
    new = semantic.remember_atomic(
        content="User strongly prefers dark roast coffee every morning",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="dark roast coffee",
        confidence=0.8,
        salience=0.9,
        source_memory_ids=["msg-new"],
    )

    active = store.list_semantic_memories(statuses=["active"])
    superseded = store.list_semantic_memories(statuses=["superseded"])

    assert new.action == "supersede"
    assert [memory.object for memory in active] == ["dark roast coffee"]
    assert [memory.id for memory in superseded] == [old.memory.id]
    assert active[0].id == new.memory.id


def test_semantic_low_confidence_changed_object_queues_conflict_review(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    old = semantic.upsert_fact("user", "prefers", "tea", "User prefers tea")
    review = semantic.remember_atomic(
        content="User might prefer coffee",
        memory_type="preference",
        subject="user",
        predicate="prefers",
        object_value="coffee",
        confidence=0.4,
        salience=0.8,
        source_memory_ids=["msg-review"],
    )

    active = store.list_semantic_memories(statuses=["active"])
    conflicts = store.list_semantic_memories(statuses=["conflict_review"])

    assert review.action == "conflict-review"
    assert [memory.id for memory in active] == [old.id]
    assert [memory.id for memory in conflicts] == [review.memory.id]
    assert conflicts[0].metadata["conflicts_with"] == [old.id]


def test_semantic_same_transaction_similarity_duplicate_merges(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)

    with store.immediate_transaction() as conn:
        first = semantic.remember_atomic(
            content="Alpha Project uses SQLite memory storage for local runtime",
            memory_type="fact",
            entities=["Alpha Project"],
            confidence=0.8,
            salience=0.7,
            source_memory_ids=["msg-1"],
            conn=conn,
        )
        second = semantic.remember_atomic(
            content="Alpha Project uses SQLite memory store for local runtime",
            memory_type="fact",
            entities=["Alpha Project"],
            confidence=0.8,
            salience=0.7,
            source_memory_ids=["msg-2"],
            conn=conn,
        )

    memories = store.list_semantic_memories(statuses=["active"])
    assert first.action == "store"
    assert second.action == "merge"
    assert len(memories) == 1
    assert memories[0].source_memory_ids == ["msg-1", "msg-2"]


def test_forget_semantic_memory_marks_deleted_without_physical_delete(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    semantic = SemanticMemoryManager(store)
    memory = semantic.upsert_fact("user", "prefers", "tea", "User prefers tea")

    deleted = store.forget_semantic_memory(memory.id, reason="test forget")

    assert deleted.status == "deleted"
    assert deleted.metadata["forget_reason"] == "test forget"
    assert store.list_semantic_memories(statuses=["active"]) == []
    assert store.list_semantic_memories(statuses=["deleted"])[0].id == memory.id


def test_procedural_upsert_is_scope_aware(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    default_scope = MemoryScope.default()
    other_scope = MemoryScope(
        kind="platform_user",
        scope_key="platform:telegram:user:other",
        platform="telegram",
        user_id="other",
    )

    first = store.upsert_procedural_memory(
        ProceduralMemory(
            id="proc-default",
            name="Debug Loop",
            description="Default scope procedure",
            trigger="debug",
            procedure_markdown="Default steps",
            success_count=0,
            failure_count=0,
            confidence=0.7,
            created_at=now,
            updated_at=now,
            scope=default_scope,
        )
    )
    second = store.upsert_procedural_memory(
        ProceduralMemory(
            id="proc-other",
            name="Debug Loop",
            description="Other scope procedure",
            trigger="debug",
            procedure_markdown="Other steps",
            success_count=0,
            failure_count=0,
            confidence=0.8,
            created_at=now,
            updated_at=now,
            scope=other_scope,
        )
    )
    updated_default = store.upsert_procedural_memory(
        ProceduralMemory(
            id="proc-default-replacement",
            name="Debug Loop",
            description="Updated default scope procedure",
            trigger="debug",
            procedure_markdown="Updated default steps",
            success_count=0,
            failure_count=0,
            confidence=0.9,
            created_at=now,
            updated_at=now,
            scope=default_scope,
        )
    )

    default_memories = store.list_procedural_memories(scopes=[default_scope])
    other_memories = store.list_procedural_memories(scopes=[other_scope])
    assert first.id == updated_default.id
    assert first.id != second.id
    assert [memory.description for memory in default_memories] == [
        "Updated default scope procedure"
    ]
    assert [memory.description for memory in other_memories] == ["Other scope procedure"]


def test_negative_preference_is_stored_as_dislike() -> None:
    from alpha_agent.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor()

    candidates = extractor.extract("I don't like verbose answers", "", ["evt1"])

    semantic = [candidate for candidate in candidates if candidate.type == "semantic"]
    assert semantic[0].predicate == "dislikes"
    assert semantic[0].object == "verbose answers"


def test_memory_candidates_and_decisions_are_auditable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    candidate = MemoryCandidate(
        id="cand-1",
        candidate_type="semantic",
        proposed_layer="semantic",
        content="User prefers tea",
        weak_structure={"subject": "user", "predicate": "prefers", "object": "tea"},
        salience=0.9,
        confidence=0.8,
        scope=MemoryScope.default(),
        source_message_ids=["msg-1"],
        status="pending",
        created_at=now,
        updated_at=now,
    )

    store.insert_memory_candidate(candidate)
    updated = store.update_memory_candidate_status(
        "cand-1",
        "rejected",
        reviewer_metadata={"reviewer": "cli"},
    )
    store.insert_memory_decision(
        MemoryDecision(
            id="decision-1",
            candidate_id="cand-1",
            action="reject",
            memory_type=None,
            memory_id=None,
            reviewer="cli",
            rationale="test rejection",
            created_at=now,
        )
    )

    assert updated.status == "rejected"
    assert updated.reviewer_metadata == {"reviewer": "cli"}
    assert store.list_memory_candidates(status="pending") == []
    assert store.list_memory_candidates(status="rejected")[0].source_message_ids == ["msg-1"]
    assert store.stats()["memory_decisions"] == 1


def test_memory_candidate_audit_recovers_sources_and_decision_history(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    now = utc_now_iso()
    source = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="Remember that I prefer green tea",
        source_metadata={"message_id": "platform-1"},
    )
    candidate = MemoryCandidate(
        id="cand-audit",
        candidate_type="semantic",
        proposed_layer="semantic",
        content="User prefers green tea",
        weak_structure={"subject": "user", "predicate": "prefers", "object": "green tea"},
        salience=0.9,
        confidence=0.8,
        scope=MemoryScope.default(),
        source_message_ids=[source.id],
        status="pending",
        created_at=now,
        updated_at=now,
    )

    store.insert_memory_candidate(candidate)
    store.insert_memory_decision(
        MemoryDecision(
            id="decision-1",
            candidate_id="cand-audit",
            action="pending",
            memory_type=None,
            memory_id=None,
            reviewer="memory_controller",
            rationale="needs review",
            created_at=now,
        )
    )

    sources = store.list_conversation_messages_by_ids([source.id, "missing-msg"])
    decisions = store.list_memory_decisions("cand-audit")

    assert [message.id for message in sources] == [source.id]
    assert sources[0].raw_content == "Remember that I prefer green tea"
    assert decisions[0].action == "pending"
    assert decisions[0].rationale == "needs review"
