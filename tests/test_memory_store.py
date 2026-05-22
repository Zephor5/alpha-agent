from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from alpha_agent.memory.models import SessionContextState
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
    )
    second = semantic.upsert_fact("User", "Prefers", "Concise Answers", "Updated content")

    memories = store.list_semantic_memories()
    assert len(memories) == 1
    assert first.id == second.id
    assert memories[0].content == "Updated content"


def test_negative_preference_is_stored_as_dislike() -> None:
    from alpha_agent.memory.extractor import MemoryExtractor

    extractor = MemoryExtractor()

    candidates = extractor.extract("I don't like verbose answers", "", ["evt1"])

    semantic = [candidate for candidate in candidates if candidate.type == "semantic"]
    assert semantic[0].predicate == "dislikes"
    assert semantic[0].object == "verbose answers"
