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
    LLMUsage,
)
from alpha_agent.runtime.chat_messages import wrap_system_reminder
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
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.state.store import StateStore
from alpha_agent.utils.system_reminder import SYSTEM_REMINDER_CLOSE, SYSTEM_REMINDER_OPEN


class _ProviderCall(TypedDict):
    messages: list[ChatMessage]
    tools: Sequence[LLMToolDefinitionInput] | None
    tool_choice: LLMToolChoice | None
    response_format: object | None


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

    def __init__(
        self,
        response: str,
        *,
        usage: LLMUsage | None = None,
        raw_usage: dict[str, object] | None = None,
    ):
        self.response = response
        self.usage = usage
        self.raw_usage = raw_usage
        self.calls: list[_ProviderCall] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: object | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        metadata = (
            {"response_payload": {"usage": self.raw_usage}}
            if self.raw_usage is not None
            else {}
        )
        return LLMResponse(
            content=self.response,
            model="test-model",
            provider=self.name,
            metadata=metadata,
            usage=self.usage,
        )


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
        response_format: object | None = None,
    ) -> LLMResponse:
        del messages, tools, tool_choice, response_format
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
    assert instruction.startswith(SYSTEM_REMINDER_OPEN)
    assert instruction.endswith(SYSTEM_REMINDER_CLOSE)
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
    assert instruction.startswith(SYSTEM_REMINDER_OPEN)
    assert instruction.endswith(SYSTEM_REMINDER_CLOSE)
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
    assert compressed.provider_metadata["llm_call_id"] == result.llm_call_id
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
    assert traces[0].metadata["llm_call_id"] == result.llm_call_id
    assert traces[0].metadata["compression_point_ordinal"] == assistant.ordinal
    assert traces[0].metadata["prompt_message_count"] == len(runtime_messages) + 1
    assert traces[1].metadata["llm_call_id"] == result.llm_call_id
    assert traces[1].metadata["started_trace_id"] == traces[0].id
    assert traces[1].metadata["compressed_message_id"] == compressed.id
    assert traces[1].metadata["provider"] == provider.name
    assert traces[1].metadata["model"] == "test-model"
    assert traces[1].metadata["extraction_version"] == DEFAULT_MEMORY_EXTRACTION_VERSION
    assert traces[1].metadata["prompt_prefix_hash"] == handover_prompt_prefix_hash(
        runtime_messages
    )
    assert traces[1].metadata["tools_schema_hash"] == handover_tools_schema_hash(tools)
    assert "covered_source_message_ids" not in traces[1].metadata
    assert "covered_source_message_refs" not in traces[1].metadata
    assert "context_source_message_refs" not in traces[1].metadata
    assert traces[1].metadata["covered_ordinal_start"] == user.ordinal
    assert traces[1].metadata["covered_ordinal_end"] == assistant.ordinal
    assert all(
        DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in trace.content
        and DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in str(trace.metadata)
        for trace in traces
    )


def test_compression_records_successful_llm_call_and_session_usage(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    raw_usage = {
        "total_tokens": 240,
        "prompt_tokens": 190,
        "prompt_tokens_details": {"cached_tokens": 70},
        "completion_tokens": 50,
        "completion_tokens_details": {"reasoning_tokens": 9},
    }
    usage = LLMUsage(
        total_tokens=240,
        cached_tokens=70,
        prompt_cache_miss_tokens=120,
        reasoning_tokens=9,
        completion_tokens=50,
    )
    provider = _RecordingProvider(
        "continuity body",
        usage=usage,
        raw_usage=raw_usage,
    )
    assembler = SessionContextAssembler(store)

    result = compress_session_context(
        session_id="s1",
        assembler=assembler,
        llm_provider=provider,
        llm_messages=_runtime_messages(assembler, "s1"),
    )

    traces = store.list_runtime_traces("s1")
    assert [trace.event_type for trace in traces] == [
        "handover_compression.started",
        "handover_compression.completed",
    ]
    assert traces[0].metadata["llm_call_id"] == result.llm_call_id
    assert traces[1].metadata["llm_call_id"] == result.llm_call_id
    assert result.started_trace_id == traces[0].id
    assert result.completed_trace_id == traces[1].id
    assert result.message.provider_metadata["llm_call_id"] == result.llm_call_id

    calls = store.list_llm_calls(session_id="s1")
    assert len(calls) == 1
    call = calls[0]
    assert call.id == result.llm_call_id
    assert call.provider == provider.name
    assert call.model == "test-model"
    assert call.total_tokens == 240
    assert call.cached_tokens == 70
    assert call.prompt_cache_miss_tokens == 120
    assert call.reasoning_tokens == 9
    assert call.completion_tokens == 50
    assert call.raw_usage == raw_usage
    assert call.started_trace_id == traces[0].id
    assert call.completed_trace_id == traces[1].id

    session = store.get_session_record("s1")
    assert session is not None
    assert session.total_tokens == 240
    assert session.cached_tokens == 70
    assert session.prompt_cache_miss_tokens == 120
    assert session.reasoning_tokens == 9
    assert session.completion_tokens == 50


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
    assert store.list_llm_calls(session_id="s1") == []


def test_compression_accounting_failure_does_not_fail_successful_compression(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    provider = _RecordingProvider("continuity body")

    def fail_append_llm_call(**_: object) -> None:
        raise RuntimeError("ledger unavailable")

    monkeypatch.setattr(store, "append_llm_call", fail_append_llm_call)
    assembler = SessionContextAssembler(store)

    result = compress_session_context(
        session_id="s1",
        assembler=assembler,
        llm_provider=provider,
        llm_messages=_runtime_messages(assembler, "s1"),
    )

    assert result.message.raw_content == wrap_system_reminder("continuity body")
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "handover_compression.started",
        "handover_compression.completed",
        "handover_compression.accounting_failed",
    ]
    failure_trace = store.list_runtime_traces("s1")[-1]
    assert failure_trace.metadata["llm_call_id"] == result.llm_call_id
    assert failure_trace.metadata["error_type"] == "RuntimeError"
    assert failure_trace.metadata["error"] == "ledger unavailable"


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
