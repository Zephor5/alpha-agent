from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import pytest

from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinition,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.context_handover import (
    DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION,
    DEFAULT_HANDOVER_COMPRESSION_VERSION,
    DEFAULT_MEMORY_EXTRACTION_VERSION,
    build_handover_compression_prompt,
    build_handover_compression_prompt_from_projection_with_prefix,
    compress_session_context,
    handover_prompt_prefix_hash,
    handover_tools_schema_hash,
)
from alpha_agent.runtime.session_context import SessionContextAssembler, wrap_system_reminder
from alpha_agent.state.store import StateStore


class _ProviderCall(TypedDict):
    messages: list[ChatMessage]
    tools: Sequence[LLMToolDefinitionInput] | None
    tool_choice: LLMToolChoice | None


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _runtime_messages(assembler: SessionContextAssembler, session_id: str) -> list[ChatMessage]:
    return [
        {"role": "system", "content": "Identity: Alpha Agent.\nTest runtime prefix."},
        *assembler.load(session_id).chat_messages,
    ]


class _RecordingProvider:
    name = "recording"

    def __init__(self, response: str):
        self.response = response
        self.calls: list[_ProviderCall] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
            }
        )
        return LLMResponse(content=self.response, model="test-model", provider=self.name)


class _FailingProvider:
    name = "failing"

    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice
        self.calls += 1
        raise RuntimeError("provider failed")


def test_build_prompt_preserves_explicit_runtime_messages_and_appends_instruction() -> None:
    runtime_messages: list[ChatMessage] = [
        {"role": "system", "content": "Identity: Alpha Agent.\nStable prompt prefix."},
        {"role": "user", "content": wrap_system_reminder("Cognition reminder.")},
        {"role": "user", "content": "source-visible user message"},
    ]

    prompt = build_handover_compression_prompt(
        runtime_messages,
        compression_point_ordinal=7,
    )

    assert prompt.compression_point_ordinal == 7
    assert prompt.messages[:-1] == runtime_messages
    assert prompt.messages[0]["role"] == "system"
    assert "Stable prompt prefix." in prompt.messages[0]["content"]
    assert prompt.messages[-1]["role"] == "user"
    instruction = prompt.messages[-1]["content"]
    assert isinstance(instruction, str)
    assert instruction.startswith("<system-reminder>")
    assert instruction.endswith("</system-reminder>")
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in instruction


def test_projection_prompt_helper_requires_explicit_runtime_prefix(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="source-visible user message",
    )
    projection = SessionContextAssembler(store).load("s1")

    with pytest.raises(ValueError, match="runtime prefix"):
        build_handover_compression_prompt_from_projection_with_prefix(
            projection,
            prefix_messages=[],
        )

    prompt = build_handover_compression_prompt_from_projection_with_prefix(
        projection,
        prefix_messages=[
            {"role": "system", "content": "Identity: Alpha Agent.\nStable prefix."}
        ],
    )

    assert prompt.messages[0]["role"] == "system"
    assert "Stable prefix." in str(prompt.messages[0]["content"])
    assert prompt.messages[1]["content"] == "source-visible user message"
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in str(prompt.messages[-1]["content"])


def test_compression_call_preserves_runtime_prefix_and_passes_tools(tmp_path) -> None:
    store = _store(tmp_path)
    user = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="prior answer",
    )
    provider = _RecordingProvider("continuity body")
    tools = [
        LLMToolDefinition(
            name="lookup",
            description="Lookup context.",
            parameters={"type": "object", "properties": {}},
        )
    ]
    tool_choice: LLMToolChoice = {"type": "function", "function": {"name": "lookup"}}
    runtime_messages: list[ChatMessage] = [
        {"role": "system", "content": "Identity: Alpha Agent.\nNormal runtime prefix."},
        {"role": "user", "content": wrap_system_reminder("Current cognition reminder.")},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "prior answer"},
    ]

    result = compress_session_context(
        session_id="s1",
        assembler=SessionContextAssembler(store),
        llm_provider=provider,
        llm_messages=runtime_messages,
        tools=tools,
        tool_choice=tool_choice,
    )

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["tools"] == tools
    assert call["tool_choice"] == tool_choice
    messages = call["messages"]
    assert messages[:-1] == runtime_messages
    assert messages[0]["role"] == "system"
    assert "Normal runtime prefix." in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    instruction = messages[-1]["content"]
    assert isinstance(instruction, str)
    assert instruction.startswith("<system-reminder>")
    assert instruction.endswith("</system-reminder>")
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in instruction
    assert "future user text that must not be compressed" not in str(messages)

    persisted = store.list_session_messages("s1")
    assert [message.kind for message in persisted] == [
        "user_message",
        "assistant_message",
        "compressed_message",
    ]
    assert all(
        DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in message.raw_content
        for message in persisted
    )
    compressed = result.message
    assert compressed == persisted[-1]
    assert compressed.kind == "compressed_message"
    assert compressed.llm_role == "user"
    assert compressed.raw_content == wrap_system_reminder("continuity body")
    assert compressed.compression_point_ordinal == assistant.ordinal
    assert compressed.compression_point_ordinal != result.message.ordinal
    assert compressed.compression_point_ordinal == user.ordinal + 1
    assert compressed.compression_version == DEFAULT_HANDOVER_COMPRESSION_VERSION

    traces = store.list_runtime_traces("s1")
    assert [trace.event_type for trace in traces] == [
        "handover_compression.started",
        "handover_compression.completed",
    ]
    assert traces[0].metadata["compression_point_ordinal"] == assistant.ordinal
    assert traces[0].metadata["prompt_message_count"] == len(runtime_messages) + 1
    assert traces[1].metadata["compressed_message_id"] == compressed.id
    assert traces[1].metadata["provider"] == provider.name
    assert traces[1].metadata["model"] == "test-model"
    assert traces[1].metadata["extraction_version"] == DEFAULT_MEMORY_EXTRACTION_VERSION
    assert traces[1].metadata["prompt_prefix_hash"] == handover_prompt_prefix_hash(
        runtime_messages
    )
    assert traces[1].metadata["tools_schema_hash"] == handover_tools_schema_hash(tools)
    assert traces[1].metadata["covered_source_message_ids"] == [user.id, assistant.id]
    assert traces[1].metadata["covered_source_message_refs"] == [
        {
            "source_type": "session_message",
            "source_id": user.id,
            "ordinal": user.ordinal,
            "kind": user.kind,
        },
        {
            "source_type": "session_message",
            "source_id": assistant.id,
            "ordinal": assistant.ordinal,
            "kind": assistant.kind,
        },
    ]
    assert traces[1].metadata["covered_ordinal_start"] == user.ordinal
    assert traces[1].metadata["covered_ordinal_end"] == assistant.ordinal
    assert all(
        DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in trace.content
        and DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in str(trace.metadata)
        for trace in traces
    )


def test_latest_compressed_wins_after_compression(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="old context",
    )
    store.append_compressed_message(
        session_id="s1",
        raw_content="old handover",
        compression_point_ordinal=1,
        compression_version="old-v1",
    )
    covered_after_old_handover = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="covered after old handover",
    )
    provider = _RecordingProvider("new continuity")
    assembler = SessionContextAssembler(store)

    result = compress_session_context(
        session_id="s1",
        assembler=assembler,
        llm_provider=provider,
        llm_messages=_runtime_messages(assembler, "s1"),
    )
    fresh_after_new_handover = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="fresh after new handover",
    )

    assert result.message.compression_point_ordinal == covered_after_old_handover.ordinal
    projection = SessionContextAssembler(store).load("s1")
    assert projection.compressed_message == result.message
    assert [message.id for message in projection.source_messages] == [
        result.message.id,
        fresh_after_new_handover.id,
    ]
    assert projection.chat_messages == [
        {"role": "user", "content": wrap_system_reminder("new continuity")},
        {"role": "user", "content": "fresh after new handover"},
    ]


def test_provider_failure_does_not_write_or_mutate_source_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    before = store.list_session_messages("s1")
    provider = _FailingProvider()
    assembler = SessionContextAssembler(store)

    with pytest.raises(RuntimeError, match="provider failed"):
        compress_session_context(
            session_id="s1",
            assembler=assembler,
            llm_provider=provider,
            llm_messages=_runtime_messages(assembler, "s1"),
        )

    assert provider.calls == 1
    assert store.list_session_messages("s1") == before
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "handover_compression.started",
        "handover_compression.failed",
    ]


def test_compression_requires_existing_source_message_as_compression_point(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _RecordingProvider("continuity body")

    with pytest.raises(ValueError, match="compression point"):
        compress_session_context(
            session_id="empty",
            assembler=SessionContextAssembler(store),
            llm_provider=provider,
            llm_messages=[{"role": "system", "content": "Identity: Alpha Agent."}],
        )

    assert provider.calls == []
    assert store.list_session_messages("empty") == []


def test_runtime_compression_requires_full_llm_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="source message",
    )
    provider = _RecordingProvider("continuity body")

    with pytest.raises(ValueError, match="requires explicit llm_messages"):
        compress_session_context(
            session_id="s1",
            assembler=SessionContextAssembler(store),
            llm_provider=provider,
            llm_messages=None,
        )

    assert provider.calls == []
    assert store.list_runtime_traces("s1") == []
    assert [message.kind for message in store.list_session_messages("s1")] == [
        "user_message"
    ]
