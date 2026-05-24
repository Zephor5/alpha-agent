from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from alpha_agent.llm.base import ChatMessage, LLMToolDefinition
from alpha_agent.memory.models import (
    ConversationMessage,
    EpisodicMemory,
    ProceduralMemory,
    RetrievedContext,
    SemanticMemory,
    SessionContextState,
)
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.prompt_builder import PromptBuilder
from alpha_agent.runtime.session_context import SessionContextManager, SessionContextProjection


def test_prompt_includes_memory_sections() -> None:
    context = RetrievedContext(
        semantic_memories=[
            SemanticMemory(
                id="sem1",
                content="User prefers concise answers",
                memory_type="preference",
                subject="user",
                predicate="prefers",
                object="concise answers",
                entities=["user", "concise answers"],
                confidence=0.9,
                salience=0.8,
                stability=0.8,
                source_memory_ids=[],
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        episodic_memories=[
            EpisodicMemory(
                id="epi1",
                content="User asked for memory MVP",
                summary="User asked for memory MVP",
                source_event_ids=[],
                people=[],
                places=[],
                topics=[],
                salience=0.8,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Debug failures",
                trigger="debug",
                procedure_markdown="1. Reproduce\n2. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        entity_hints=["Alpha"],
    )

    messages = PromptBuilder().build(
        "What should we do?",
        context,
        runtime_reminders=["Tool lookup: current build status is green"],
    )
    context_prompt = cast(str, messages[1].get("content"))

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert messages[-1]["content"] == "What should we do?"
    assert context_prompt.startswith("<system-reminder>\n")
    assert context_prompt.endswith("\n</system-reminder>")
    assert "## Retrieved Context (Reference Only)" in context_prompt
    assert "### Runtime Reminders" in context_prompt
    assert "Tool lookup: current build status is green" in context_prompt
    assert "### User Facts" in context_prompt
    assert "status=active" in context_prompt
    assert "scope=user:default" in context_prompt
    assert "source=" in context_prompt
    assert "### Prior Episodes" in context_prompt
    assert "### Relevant Procedures" in context_prompt
    assert "### Entity Hints" in context_prompt
    assert "## Current User Message" not in context_prompt
    assert sum(1 for message in messages if message["role"] == "system") == 1


def test_prompt_omits_empty_context_message() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )

    messages = PromptBuilder().build("hello", context)

    assert [message["role"] for message in messages] == ["system", "user"]
    assert messages[-1]["content"] == "hello"
    assert sum(1 for message in messages if message["role"] == "system") == 1


def test_prompt_keeps_irrelevant_procedure_body_out_of_context() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Diagnose failures",
                trigger="debug failure regression",
                procedure_markdown="1. Reproduce\n2. Inspect\n3. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )

    messages = PromptBuilder().build("What tools do you have?", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert "Debug Loop: Diagnose failures" in context_prompt
    assert "1. Reproduce" not in context_prompt


def test_prompt_includes_matching_procedure_body() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Diagnose failures",
                trigger="debug failure regression",
                procedure_markdown="1. Reproduce\n2. Inspect\n3. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )

    messages = PromptBuilder().build("Debug this failing regression", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert "1. Reproduce" in context_prompt


def test_prompt_applies_independent_layer_budgets() -> None:
    context = RetrievedContext(
        semantic_memories=[
            SemanticMemory(
                id="sem1",
                content="User prefers " + "very detailed " * 20 + "answers",
                memory_type="preference",
                subject="user",
                predicate="prefers",
                object="detailed answers",
                entities=["user"],
                confidence=0.92,
                salience=0.9,
                stability=0.8,
                source_memory_ids=["msg-sem"],
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        episodic_memories=[
            EpisodicMemory(
                id="epi1",
                content="",
                summary="Prior decision " + "with extensive detail " * 20,
                source_event_ids=["msg-epi"],
                people=[],
                places=[],
                topics=[],
                salience=0.7,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
        procedural_memories=[
            ProceduralMemory(
                id="proc1",
                name="Debug Loop",
                description="Debug " + "complicated failures " * 20,
                trigger="debug",
                procedure_markdown="1. Reproduce\n2. Inspect\n3. Fix",
                success_count=0,
                failure_count=0,
                confidence=0.8,
                created_at="2026-01-01T00:00:00+00:00",
                updated_at="2026-01-01T00:00:00+00:00",
                metadata={},
            )
        ],
    )
    builder = PromptBuilder(
        semantic_memory_tokens=18,
        episodic_memory_tokens=16,
        procedural_memory_tokens=16,
        session_context_tokens=12,
    )

    messages = builder.build("Debug this", context)
    context_prompt = cast(str, messages[1].get("content"))

    assert "### User Facts" in context_prompt
    assert "### Prior Episodes" in context_prompt
    assert "### Relevant Procedures" in context_prompt
    assert "..." in context_prompt
    assert context_prompt.count("very detailed") < 5
    assert context_prompt.count("extensive detail") < 5
    assert context_prompt.count("complicated failures") < 5


def test_prompt_applies_session_context_budget() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )
    projection = SessionContextProjection(
        state=SessionContextState(
            session_id="s1",
            compressed_until_ordinal=1,
            summary="Summary " + "long detail " * 30,
            summary_source_message_ids=["msg_1"],
            compression_version="test",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
        uncompressed_messages=[
            ConversationMessage(
                id="msg_2",
                session_id="s1",
                ordinal=2,
                role="user",
                raw_content="Prior " + "message detail " * 30,
                model_content=None,
                tool_call_id=None,
                tool_calls=[],
                tool_result_id=None,
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:01+00:00",
            )
        ],
        before_ordinal=3,
    )

    messages = PromptBuilder(session_context_tokens=16).build(
        "current question",
        context,
        session_context=projection,
    )
    rendered = "\n".join(str(message.get("content", "")) for message in messages)

    assert "## Compressed Session Context" in rendered
    assert "..." in rendered
    assert rendered.count("long detail") < 5
    assert rendered.count("message detail") < 5


def test_session_context_budget_includes_preamble_and_omits_overflow_messages() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )
    projection = SessionContextProjection(
        state=SessionContextState(
            session_id="s1",
            compressed_until_ordinal=1,
            summary="Summary " + "long detail " * 30,
            summary_source_message_ids=["msg_1"],
            compression_version="test",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
        uncompressed_messages=[
            ConversationMessage(
                id="msg_2",
                session_id="s1",
                ordinal=2,
                role="user",
                raw_content="Prior message that must not fit tiny session budget.",
                model_content=None,
                tool_call_id=None,
                tool_calls=[],
                tool_result_id=None,
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:01+00:00",
            )
        ],
        before_ordinal=3,
    )

    messages = PromptBuilder(session_context_tokens=8).build(
        "current question",
        context,
        session_context=projection,
    )
    session_messages = messages[1:-1]
    rendered_session = "\n".join(str(message.get("content", "")) for message in session_messages)

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert len(rendered_session) <= 8 * 4
    assert "Prior message" not in rendered_session


def test_session_context_budget_counts_tool_call_metadata() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )
    projection = SessionContextProjection(
        state=None,
        uncompressed_messages=[
            ConversationMessage(
                id="msg_2",
                session_id="s1",
                ordinal=2,
                role="assistant",
                raw_content="",
                model_content=None,
                tool_call_id=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query":"' + ("large " * 200) + '"}',
                        },
                    }
                ],
                tool_result_id=None,
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:01+00:00",
            ),
            ConversationMessage(
                id="msg_3",
                session_id="s1",
                ordinal=3,
                role="tool",
                raw_content="large result",
                model_content=None,
                tool_call_id="call_1" + ("x" * 200),
                tool_calls=[],
                tool_result_id="trace_1",
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:02+00:00",
            ),
        ],
        before_ordinal=4,
    )

    messages = PromptBuilder(session_context_tokens=8).build(
        "current question",
        context,
        session_context=projection,
    )

    assert [message["role"] for message in messages] == ["system", "user"]


def test_prompt_projects_session_summary_and_prior_messages_before_current_user() -> None:
    context = RetrievedContext(
        semantic_memories=[],
        episodic_memories=[],
        procedural_memories=[],
    )
    projection = SessionContextProjection(
        state=SessionContextState(
            session_id="s1",
            compressed_until_ordinal=2,
            summary="Earlier turn summary.",
            summary_source_message_ids=["msg_1", "msg_2"],
            compression_version="stub",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        ),
        uncompressed_messages=[
            ConversationMessage(
                id="msg_3",
                session_id="s1",
                ordinal=3,
                role="user",
                raw_content="prior question",
                model_content=None,
                tool_call_id=None,
                tool_calls=[],
                tool_result_id=None,
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:00+00:00",
            ),
            ConversationMessage(
                id="msg_4",
                session_id="s1",
                ordinal=4,
                role="assistant",
                raw_content="prior answer",
                model_content=None,
                tool_call_id=None,
                tool_calls=[],
                tool_result_id=None,
                provider_metadata={},
                source_metadata={},
                created_at="2026-01-01T00:00:01+00:00",
            ),
        ],
        before_ordinal=5,
    )

    messages = PromptBuilder().build(
        "current question",
        context,
        session_context=projection,
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert "## Compressed Session Context (Reference Only)" in cast(
        str,
        messages[1]["content"],
    )
    assert "Earlier turn summary." in cast(str, messages[1]["content"])
    assert messages[2] == {"role": "user", "content": "prior question"}
    assert messages[3] == {"role": "assistant", "content": "prior answer"}
    assert messages[-1] == {"role": "user", "content": "current question"}


def test_session_context_rewinds_boundary_inside_tool_replay_sequence(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="look this up",
    )
    assistant_tool_call = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
            }
        ],
    )
    tool_result = store.append_conversation_message(
        session_id="s1",
        role="tool",
        raw_content='{"content":"found alpha","metadata":{},"name":"lookup"}',
        tool_call_id="call_1",
        tool_result_id="trace_1",
    )
    current_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="continue",
    )
    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=assistant_tool_call.ordinal,
            summary="Compressed earlier user request.",
            summary_source_message_ids=["msg_1", assistant_tool_call.id],
            compression_version="test",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    projection = SessionContextManager(store).load(
        "s1",
        before_ordinal=current_user.ordinal,
    )

    assert [message.id for message in projection.uncompressed_messages] == [
        assistant_tool_call.id,
        tool_result.id,
    ]


def test_session_context_drops_incomplete_trailing_tool_call_sequence(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="look this up",
    )
    assistant_tool_call = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
            }
        ],
    )
    current_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="continue",
    )

    projection = SessionContextManager(store).load(
        "s1",
        before_ordinal=current_user.ordinal,
    )

    assert assistant_tool_call.id not in {
        message.id for message in projection.uncompressed_messages
    }
    assert [message.role for message in projection.uncompressed_messages] == ["user"]


def test_session_context_rejects_summary_that_reaches_current_user(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="prior",
    )
    current_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="current",
    )
    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=current_user.ordinal,
            summary="Unsafe summary.",
            summary_source_message_ids=["msg_1", "msg_2"],
            compression_version="test",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    with pytest.raises(ValueError, match="compressed_until_ordinal"):
        SessionContextManager(store).load("s1", before_ordinal=current_user.ordinal)


def test_prompt_token_estimate_includes_tool_call_payloads_and_tool_schemas() -> None:
    messages: list[ChatMessage] = [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_memory", "arguments": '{"query":"hello"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "result"},
    ]
    tools = [
        LLMToolDefinition(
            name="lookup_memory",
            description="Lookup memory.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    text_only_estimate = len("helloresult") // 4
    message_estimate = PromptBuilder().estimate_prompt_tokens(messages)
    estimate_with_tools = PromptBuilder().estimate_prompt_tokens(messages, tools=tools)

    assert message_estimate > text_only_estimate
    assert estimate_with_tools > message_estimate
    assert PromptBuilder().rough_token_estimate(messages) == message_estimate
