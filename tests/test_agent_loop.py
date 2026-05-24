from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from alpha_agent.llm.base import ChatMessage, LLMResponse, LLMToolCall
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.memory.models import (
    ConversationMessage,
    MemoryScope,
    RetrievedContext,
    SessionContextState,
)
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.agent import (
    AgentCanceledError,
    AlphaAgent,
    LLMCallError,
    ToolLoopLimitExceeded,
    ToolProtocolError,
)
from alpha_agent.runtime.context_compression import (
    CompressionBudget,
    CompressionContext,
    CompressionFocus,
    CompressionResult,
)
from alpha_agent.tools.base import ToolResult
from alpha_agent.tools.registry import ToolRegistry


def test_mock_agent_loop_stores_user_and_assistant_messages(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    ProceduralMemoryManager(store).load_builtin_skills()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
    )

    result = agent.respond("remember that I prefer concise answers", session_id="s1")

    messages = store.list_conversation_messages("s1")
    traces = store.list_runtime_traces(session_id="s1", limit=20)
    semantic = store.list_semantic_memories()
    candidates = store.list_memory_candidates(status="auto_approved")
    assert "Mock response" in result.response
    assert [message.role for message in messages] == [
        "user",
        "assistant",
    ]
    assert messages[0].raw_content == "remember that I prefer concise answers"
    assert "Mock response" in messages[1].raw_content
    assert {
        trace.event_type
        for trace in traces
    } >= {
        "llm.started",
        "llm.completed",
        "memory.extracted",
    }
    assert len(semantic) == 1
    assert candidates
    assert result.debug["persisted_memory_count"] >= 1
    assert result.debug["memory_scope"]["scope_key"] == "user:default"
    assert result.debug["extracted_memory_count"] >= 1


def test_agent_memory_capture_mode_disabled_skips_candidates_and_persistence(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
        memory_capture_mode="disabled",
    )

    result = agent.respond("remember that I prefer concise answers", session_id="s1")

    assert result.debug["memory_capture_mode"] == "disabled"
    assert result.debug["extracted_memory_count"] == 0
    assert result.debug["persisted_memory_count"] == 0
    assert store.list_memory_candidates() == []
    assert store.list_semantic_memories() == []


def test_agent_memory_capture_mode_candidate_only_does_not_promote(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
        memory_capture_mode="candidate_only",
    )

    result = agent.respond("remember that I prefer concise answers", session_id="s1")

    pending = store.list_memory_candidates(status="pending")
    assert result.debug["memory_capture_mode"] == "candidate_only"
    assert result.debug["extracted_memory_count"] >= 1
    assert result.debug["persisted_memory_count"] == 0
    assert pending
    assert store.list_semantic_memories() == []


def test_shared_gateway_scope_writes_candidates_to_channel_scope(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    shared_scope = MemoryScope(
        kind="chat_thread",
        scope_key="platform:telegram:chat:chat-1:thread:main",
        platform="telegram",
        chat_id="chat-1",
        user_id=None,
        metadata={"session_mode": "group_shared", "source_user_id": "user-1"},
    )
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
        memory_capture_mode="candidate_only",
    )

    result = agent.respond(
        "remember that the channel prefers weekly summaries",
        session_id="s1",
        source_metadata={"channel": "gateway", "memory_scope": shared_scope.to_record()},
    )

    candidate = store.list_memory_candidates()[0]
    assert result.debug["memory_scope"]["scope_key"] == shared_scope.scope_key
    assert candidate.scope.scope_key == shared_scope.scope_key
    assert candidate.scope.user_id is None


def test_agent_turn_persists_user_source_metadata(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
    )

    agent.respond(
        "hello",
        session_id="s1",
        source_metadata={"channel": "cli", "command": "ask"},
    )

    messages = store.list_conversation_messages("s1")
    assert messages[0].source_metadata == {"channel": "cli", "command": "ask"}
    assert messages[1].source_metadata == {}


def test_agent_honors_configured_retrieval_limit(tmp_path: Path) -> None:
    class RecordingRetriever(MemoryRetriever):
        def __init__(self, store: MemoryStore):
            super().__init__(store)
            self.seen_limit: int | None = None

        def retrieve_context(
            self,
            query: str,
            session_id: str,
            limit: int = 8,
            **kwargs: Any,
        ) -> RetrievedContext:
            self.seen_limit = limit
            return super().retrieve_context(query, session_id, limit, **kwargs)

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    retriever = RecordingRetriever(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=retriever,
        retrieval_limit=2,
    )

    agent.respond("hello", session_id="s1")

    assert retriever.seen_limit == 2


def test_agent_does_not_inject_current_user_message_as_retrieved_context(
    tmp_path: Path,
) -> None:
    class RecordingProvider:
        name = "recording-provider"

        def __init__(self) -> None:
            self.requests: list[list[ChatMessage]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.requests.append(messages)
            return LLMResponse(
                content="recorded response",
                model="mock",
                provider=self.name,
                metadata={},
                finish_reason="stop",
            )

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
    )

    agent.respond("first question", session_id="s1")
    agent.respond("second question", session_id="s1")

    second_request = provider.requests[-1]
    assert second_request[-1] == {"role": "user", "content": "second question"}
    assert sum(1 for message in second_request if message["role"] == "system") == 1
    assert second_request[1] == {"role": "user", "content": "first question"}
    assert second_request[2] == {"role": "assistant", "content": "recorded response"}
    prior_context = "\n\n".join(
        str(message.get("content", "")) for message in second_request[:-1]
    )
    assert "second question" not in prior_context
    assert "## Current User Message" not in prior_context


def test_agent_compresses_older_context_preserves_tail_and_transcript(
    tmp_path: Path,
) -> None:
    class RecordingProvider:
        name = "recording-provider"

        def __init__(self) -> None:
            self.requests: list[list[ChatMessage]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.requests.append(messages)
            return LLMResponse(
                content="compressed response",
                model="mock",
                provider=self.name,
                metadata={},
                finish_reason="stop",
            )

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    older_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="older user context " + ("alpha " * 80),
    )
    older_assistant = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="older assistant context " + ("beta " * 80),
    )
    tail_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="recent tail question",
    )
    tail_assistant = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="recent tail answer",
    )
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        context_max_prompt_tokens=80,
        context_compression_threshold_ratio=0.5,
        context_recent_tail_messages=2,
    )

    result = agent.respond("CURRENT UNIQUE REQUEST", session_id="s1")

    request = provider.requests[0]
    state = store.get_session_context_state("s1")
    transcript = store.list_conversation_messages("s1")
    assert state is not None
    assert state.compressed_until_ordinal == older_assistant.ordinal
    assert state.summary_source_message_ids == [older_user.id, older_assistant.id]
    assert state.compression_version
    assert state.metadata["prompt_token_estimate_before"] > state.metadata[
        "prompt_token_estimate_after"
    ]
    assert state.metadata["threshold_tokens"] == 40
    assert result.debug["context_compression_status"] == "completed"
    assert result.debug["prompt_token_estimate"] == result.debug[
        "prompt_token_estimate_after_rebuild"
    ]
    assert result.debug["prompt_token_estimate_before_compression"] > result.debug[
        "prompt_token_estimate_after_rebuild"
    ]
    assert request[-1] == {"role": "user", "content": "CURRENT UNIQUE REQUEST"}
    assert request[-3:-1] == [
        {"role": "user", "content": tail_user.raw_content},
        {"role": "assistant", "content": tail_assistant.raw_content},
    ]
    summary_messages = [
        message
        for message in request
        if "## Compressed Session Context" in str(message.get("content", ""))
    ]
    assert len(summary_messages) == 1
    summary_content = str(summary_messages[0]["content"])
    assert "older user context" in summary_content
    assert "older assistant context" in summary_content
    assert "CURRENT UNIQUE REQUEST" not in summary_content
    assert {"role": "user", "content": older_user.raw_content} not in request
    assert {"role": "assistant", "content": older_assistant.raw_content} not in request
    assert [message.id for message in transcript[:4]] == [
        older_user.id,
        older_assistant.id,
        tail_user.id,
        tail_assistant.id,
    ]
    assert [message.raw_content for message in transcript[:4]] == [
        older_user.raw_content,
        older_assistant.raw_content,
        tail_user.raw_content,
        tail_assistant.raw_content,
    ]
    assert transcript[-2].raw_content == "CURRENT UNIQUE REQUEST"

    events = store.list_runtime_traces(session_id="s1", limit=20)
    assert {event.event_type for event in events} >= {
        "context_compression.started",
        "context_compression.completed",
    }


def test_agent_keeps_tool_replay_intact_when_compression_boundary_would_split_it(
    tmp_path: Path,
) -> None:
    class RecordingProvider:
        name = "recording-provider"

        def __init__(self) -> None:
            self.requests: list[list[ChatMessage]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.requests.append(messages)
            return LLMResponse(
                content="tool replay preserved",
                model="mock",
                provider=self.name,
                metadata={},
                finish_reason="stop",
            )

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    prior_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="look up alpha " + ("detail " * 80),
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
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        context_max_prompt_tokens=80,
        context_compression_threshold_ratio=0.5,
        context_recent_tail_messages=1,
    )

    agent.respond("continue from the tool result", session_id="s1")

    request = provider.requests[0]
    state = store.get_session_context_state("s1")
    assert state is not None
    assert state.compressed_until_ordinal == prior_user.ordinal
    assert state.summary_source_message_ids == [prior_user.id]
    assert request[-3:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": tool_result.raw_content,
        },
        {"role": "user", "content": "continue from the tool result"},
    ]
    assert {"role": "user", "content": prior_user.raw_content} not in request
    assert [message.raw_content for message in store.list_conversation_messages("s1")[:3]] == [
        prior_user.raw_content,
        assistant_tool_call.raw_content,
        tool_result.raw_content,
    ]


@pytest.mark.parametrize(
    ("invalid_output", "error_match"),
    [
        ("future_boundary", "compressed_until_ordinal"),
        ("future_source_id", "summary_source_message_ids"),
    ],
)
def test_agent_rejects_invalid_compressor_output_without_changing_context_state(
    tmp_path: Path,
    invalid_output: str,
    error_match: str,
) -> None:
    class BadCompressor:
        compression_version = "bad-compressor"

        def should_compress(
            self,
            context: CompressionContext,
            budget: CompressionBudget,
        ) -> bool:
            return True

        def compress(
            self,
            messages: Sequence[ConversationMessage],
            previous_summary: str,
            focus: CompressionFocus,
        ) -> CompressionResult:
            compressed_until_ordinal = (
                999
                if invalid_output == "future_boundary"
                else list(messages)[-1].ordinal
            )
            source_message_ids = [
                *focus.previous_summary_source_message_ids,
                *[message.id for message in messages],
            ]
            if invalid_output == "future_source_id":
                source_message_ids.append("msg_future")
            return CompressionResult(
                summary="unsafe summary",
                summary_source_message_ids=source_message_ids,
                compressed_until_ordinal=compressed_until_ordinal,
                compression_version=self.compression_version,
                input_token_estimate=10,
                output_token_estimate=2,
            )

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    store.upsert_session_context_state(
        SessionContextState(
            session_id="s1",
            compressed_until_ordinal=0,
            summary="unchanged summary",
            summary_source_message_ids=["msg_previous"],
            compression_version="previous",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
            metadata={"stable": True},
        )
    )
    prior_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="older user context " + ("alpha " * 80),
    )
    prior_assistant = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="older assistant context " + ("beta " * 80),
    )
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
        context_compressor=BadCompressor(),
        context_max_prompt_tokens=80,
        context_compression_threshold_ratio=0.5,
        context_recent_tail_messages=1,
    )

    with pytest.raises(ValueError, match=error_match):
        agent.respond("current user must stay raw", session_id="s1")

    state = store.get_session_context_state("s1")
    assert state is not None
    assert state.compressed_until_ordinal == 0
    assert state.summary == "unchanged summary"
    assert state.summary_source_message_ids == ["msg_previous"]
    assert state.compression_version == "previous"
    assert state.metadata == {"stable": True}
    transcript = store.list_conversation_messages("s1")
    assert [message.raw_content for message in transcript] == [
        prior_user.raw_content,
        prior_assistant.raw_content,
        "current user must stay raw",
    ]
    failed = next(
        trace
        for trace in store.list_runtime_traces(session_id="s1", limit=20)
        if trace.event_type == "context_compression.failed"
    )
    assert failed.metadata["stage"] == "validation"
    assert failed.metadata["error_type"] == "ValueError"
    assert not any(
        trace.event_type == "context_compression.completed"
        for trace in store.list_runtime_traces(session_id="s1", limit=20)
    )


@pytest.mark.parametrize("failure_stage", ["decision", "compress"])
def test_agent_compression_failures_emit_trace_and_do_not_write_context_state(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    class RaisingCompressor:
        compression_version = f"raising-{failure_stage}"

        def should_compress(
            self,
            context: CompressionContext,
            budget: CompressionBudget,
        ) -> bool:
            if failure_stage == "decision":
                raise RuntimeError("decision failed")
            return True

        def compress(
            self,
            messages: Sequence[ConversationMessage],
            previous_summary: str,
            focus: CompressionFocus,
        ) -> CompressionResult:
            raise RuntimeError("compress failed")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    prior_user = store.append_conversation_message(
        session_id="s1",
        role="user",
        raw_content="older user context " + ("alpha " * 80),
    )
    prior_assistant = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="older assistant context " + ("beta " * 80),
    )
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
        context_compressor=RaisingCompressor(),
        context_max_prompt_tokens=80,
        context_compression_threshold_ratio=0.5,
        context_recent_tail_messages=1,
    )

    with pytest.raises(RuntimeError, match=failure_stage):
        agent.respond("current user must remain in transcript", session_id="s1")

    assert store.get_session_context_state("s1") is None
    assert [
        message.raw_content for message in store.list_conversation_messages("s1")
    ] == [
        prior_user.raw_content,
        prior_assistant.raw_content,
        "current user must remain in transcript",
    ]
    failed = next(
        trace
        for trace in store.list_runtime_traces(session_id="s1", limit=20)
        if trace.event_type == "context_compression.failed"
    )
    assert failed.metadata["stage"] == failure_stage
    assert failed.metadata["error_type"] == "RuntimeError"
    assert not any(
        trace.event_type == "context_compression.completed"
        for trace in store.list_runtime_traces(session_id="s1", limit=20)
    )


def test_agent_logs_raw_llm_request_and_response(tmp_path: Path) -> None:
    class MetadataProvider:
        name = "metadata-provider"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="logged response",
                model="mock",
                provider=self.name,
                metadata={
                    "request_payload": {"messages": messages},
                    "response_payload": {"id": "resp-1"},
                },
                finish_reason="stop",
            )

    trace_path = tmp_path / "logs" / "llm.jsonl"
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MetadataProvider(),
        retriever=MemoryRetriever(store),
        llm_debug_logging=True,
        llm_trace_log_path=trace_path,
    )

    agent.respond("hello", session_id="s1")

    traces = store.list_runtime_traces(session_id="s1", limit=20)
    started = next(trace for trace in traces if trace.event_type == "llm.started")
    completed = next(
        trace for trace in traces if trace.event_type == "llm.completed"
    )
    assert started.metadata["request"]["messages"][0]["role"] == "system"
    assert started.metadata["request"]["tool_choice"] is None
    assert completed.metadata["response"]["content"] == "logged response"
    assert completed.metadata["response"]["metadata"] == {"response_payload": {"id": "resp-1"}}
    assert completed.metadata["response"]["tool_calls"] == []

    entries = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [entry["event"] for entry in entries] == ["llm.request", "llm.response"]
    assert entries[0]["metadata"]["request"] == started.metadata["request"]
    assert entries[1]["metadata"]["response"] == completed.metadata["response"]
    assert set(entries[1]["metadata"]) == {"llm_call_id", "retry_count", "response"}
    assert "request_payload" not in entries[1]["metadata"]["response"]["metadata"]


def test_agent_omits_raw_llm_logs_unless_debug_logging_is_enabled(tmp_path: Path) -> None:
    class MetadataProvider:
        name = "metadata-provider"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="quiet response",
                model="mock",
                provider=self.name,
                metadata={"raw_payload": {"id": "resp-1"}},
                finish_reason="stop",
            )

    trace_path = tmp_path / "logs" / "llm.jsonl"
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MetadataProvider(),
        retriever=MemoryRetriever(store),
        llm_trace_log_path=trace_path,
    )

    agent.respond("hello", session_id="s1")

    traces = store.list_runtime_traces(session_id="s1", limit=20)
    started = next(trace for trace in traces if trace.event_type == "llm.started")
    completed = next(
        trace for trace in traces if trace.event_type == "llm.completed"
    )
    assert "request" not in started.metadata
    assert "response" not in completed.metadata
    assert not trace_path.exists()


def test_tool_registry_exports_provider_neutral_llm_tool_definitions() -> None:
    class SearchTool:
        name = "search"
        description = "Search indexed notes."
        parameters = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        strict = True

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="done")

    class PingTool:
        name = "ping"
        description = "Ping with no parameters."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="pong")

    registry = ToolRegistry()
    registry.register(SearchTool())
    registry.register(PingTool())

    definitions = registry.to_llm_tool_definitions()

    assert [definition.name for definition in definitions] == ["ping", "search"]
    assert definitions[0].parameters == {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }
    assert definitions[0].strict is None
    assert definitions[1].parameters == SearchTool.parameters
    assert definitions[1].strict is True


def test_agent_does_not_pass_tool_kwargs_when_registry_is_empty(tmp_path: Path) -> None:
    class RecordingProvider:
        name = "recording"

        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.kwargs.append(kwargs)
            return LLMResponse(content="plain response", model="mock", provider=self.name)

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
    )

    result = agent.respond("hello", session_id="s1")

    assert result.response == "plain response"
    assert provider.kwargs == [{}]
    assert result.debug["llm_round_count"] == 1
    assert result.debug["final_finish_reason"] is None


def test_agent_passes_registered_tools_and_auto_tool_choice(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."
        parameters = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="echoed")

    class RecordingProvider:
        name = "recording"

        def __init__(self) -> None:
            self.kwargs: list[dict[str, Any]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.kwargs.append(kwargs)
            return LLMResponse(content="done", model="mock", provider=self.name)

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    agent.respond("hello", session_id="s1")

    assert provider.kwargs[0]["tool_choice"] == "auto"
    assert [tool.name for tool in provider.kwargs[0]["tools"]] == ["echo"]
    assert provider.kwargs[0]["tools"][0].parameters == EchoTool.parameters


def test_agent_records_transcript_and_runtime_trace_sequence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
    )

    result = agent.respond("hello", session_id="s1")

    traces = list(reversed(store.list_runtime_traces(session_id="s1", limit=20)))
    event_types = [trace.event_type for trace in traces]
    assert event_types == [
        "context_compression.skipped",
        "llm.started",
        "llm.completed",
        "memory.candidates.created",
        "memory.extracted",
        "memory.decisions",
    ]
    messages = store.list_conversation_messages("s1")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].id == result.debug["assistant_message_id"]


def test_provider_tool_calls_can_run_multiple_bounded_iterations(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."
        parameters = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(
                name=self.name,
                content="echoed",
                metadata={"arguments": arguments},
            )

    class ToolCallingProvider:
        name = "tool-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            if len(self.calls) == 1:
                return LLMResponse(
                    content="",
                    model="mock",
                    provider=self.name,
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            id="call_1",
                            name="echo",
                            arguments={"text": "alpha"},
                            raw_arguments='{"text":"alpha"}',
                        )
                    ],
                )
            if len(self.calls) == 2:
                return LLMResponse(
                    content="",
                    model="mock",
                    provider=self.name,
                    finish_reason="tool_calls",
                    tool_calls=[
                        LLMToolCall(
                            id="call_2",
                            name="echo",
                            arguments={"text": "beta"},
                            raw_arguments='{"text":"beta"}',
                        )
                    ],
                )
            return LLMResponse(
                content="final answer",
                model="mock",
                provider=self.name,
                finish_reason="stop",
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = ToolCallingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    result = agent.respond("use echo", session_id="s1")

    assert result.response == "final answer"
    assert len(provider.calls) == 3
    assert provider.calls[0][1]["tool_choice"] == "auto"
    assert provider.calls[1][1]["tool_choice"] == "auto"
    assert provider.calls[2][1]["tool_choice"] == "auto"
    follow_up_messages = provider.calls[1][0]
    assert follow_up_messages[-2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": '{"text":"alpha"}'},
            }
        ],
    }
    assert follow_up_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": (
            '{"content":"echoed","metadata":{"arguments":{"text":"alpha"}},"name":"echo"}'
        ),
    }
    second_follow_up_messages = provider.calls[2][0]
    assert second_follow_up_messages[-4:] == [
        follow_up_messages[-2],
        follow_up_messages[-1],
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"beta"}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": (
                '{"content":"echoed","metadata":{"arguments":{"text":"beta"}},"name":"echo"}'
            ),
        },
    ]
    events = store.list_runtime_traces(session_id="s1", limit=30)
    llm_completed_rounds = [
        event.metadata["round"]
        for event in events
        if event.event_type == "llm.completed"
    ]
    assert llm_completed_rounds == ["tool_result_2", "tool_result_1", "initial"]
    assert result.debug["provider_tool_call_count"] == 2
    assert result.debug["tool_iteration_count"] == 2
    assert result.debug["llm_round_count"] == 3
    assert result.debug["initial_provider"] == "tool-provider"
    assert result.debug["final_provider"] == "tool-provider"
    assert result.debug["final_finish_reason"] == "stop"


def test_max_tool_iterations_triggers_no_tools_finalization(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="echoed")

    class RepeatingToolProvider:
        name = "looping-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            if len(self.calls) == 4:
                return LLMResponse(
                    content="summary after limit",
                    model="mock",
                    provider=self.name,
                    finish_reason="stop",
                )
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{len(self.calls)}",
                        name="echo",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = RepeatingToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
        max_tool_iterations=2,
    )

    result = agent.respond("use echo", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    finalizing = next(
        event for event in events if event.event_type == "tool_loop.finalizing"
    )
    assert result.response == "summary after limit"
    assert len(provider.calls) == 4
    assert provider.calls[-1][1]["tool_choice"] == "none"
    assert [tool.name for tool in provider.calls[-1][1]["tools"]] == ["echo"]
    assert provider.calls[-1][0][-1]["role"] == "user"
    assert provider.calls[-1][0][-1]["content"].startswith("<system-reminder>\n")
    assert provider.calls[-1][0][-1]["content"].endswith("\n</system-reminder>")
    assert "Do not call tools" in provider.calls[-1][0][-1]["content"]
    assert result.debug["llm_round_count"] == 4
    assert result.debug["tool_iteration_count"] == 2
    assert result.debug["final_finish_reason"] == "stop"
    assert finalizing.metadata["reason"] == "max_tool_iterations"
    assert finalizing.metadata["tool_iteration_count"] == 2


def test_max_llm_rounds_triggers_no_tools_finalization(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="echoed")

    class RoundLimitedProvider:
        name = "round-limited-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            if len(self.calls) == 3:
                return LLMResponse(
                    content="final answer after round limit",
                    model="mock",
                    provider=self.name,
                    finish_reason="stop",
                )
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{len(self.calls)}",
                        name="echo",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = RoundLimitedProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
        max_llm_rounds=2,
    )

    result = agent.respond("use echo", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=40)
    finalizing = next(
        event for event in events if event.event_type == "tool_loop.finalizing"
    )
    assert result.response == "final answer after round limit"
    assert len(provider.calls) == 3
    assert provider.calls[-1][1]["tool_choice"] == "none"
    assert [tool.name for tool in provider.calls[-1][1]["tools"]] == ["echo"]
    assert provider.calls[-1][0][-1]["role"] == "user"
    assert provider.calls[-1][0][-1]["content"].startswith("<system-reminder>\n")
    assert provider.calls[-1][0][-1]["content"].endswith("\n</system-reminder>")
    assert "Do not call tools" in provider.calls[-1][0][-1]["content"]
    assert result.debug["llm_round_count"] == 3
    assert result.debug["tool_iteration_count"] == 1
    assert result.debug["final_finish_reason"] == "stop"
    assert finalizing.metadata["reason"] == "max_llm_rounds"
    assert finalizing.metadata["llm_round_count"] == 2
    assert finalizing.metadata["tool_iteration_count"] == 1


def test_finalization_returning_tool_calls_fails_observably(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="echoed")

    class FinalizationToolCallingProvider:
        name = "finalization-tool-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id=f"call_{len(self.calls)}",
                        name="echo",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = FinalizationToolCallingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
        max_tool_iterations=1,
    )

    with pytest.raises(ToolLoopLimitExceeded, match="finalization returned tool calls"):
        agent.respond("use echo", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=40)
    failed = next(event for event in events if event.event_type == "turn.failed")
    finalization_failed = next(
        event
        for event in events
        if event.event_type == "tool_loop.finalization_failed"
    )
    assert len(provider.calls) == 3
    assert provider.calls[-1][1]["tool_choice"] == "none"
    assert [tool.name for tool in provider.calls[-1][1]["tools"]] == ["echo"]
    assert failed.metadata["stage"] == "tool_loop_limit_exceeded"
    assert failed.metadata["error_code"] == "tool_loop_limit_exceeded"
    assert failed.metadata["llm_round_count"] == 3
    assert failed.metadata["tool_iteration_count"] == 1
    assert failed.metadata["final_finish_reason"] == "tool_calls"
    assert finalization_failed.metadata["finish_reason"] == "tool_calls"


def test_provider_tool_call_missing_id_fails_before_running_tool(tmp_path: Path) -> None:
    class SideEffectTool:
        name = "write"
        description = "Side-effecting tool."

        def __init__(self) -> None:
            self.run_count = 0

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            self.run_count += 1
            return ToolResult(name=self.name, content="ran")

    class MissingIdProvider:
        name = "missing-id-provider"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id=None,
                        name="write",
                        arguments={"value": "danger"},
                        raw_arguments='{"value":"danger"}',
                    )
                ],
            )

    tool = SideEffectTool()
    registry = ToolRegistry()
    registry.register(tool)
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = MissingIdProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    with pytest.raises(ToolProtocolError, match="missing an id"):
        agent.respond("write", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    failed = next(event for event in events if event.event_type == "turn.failed")
    assert provider.calls == 1
    assert tool.run_count == 0
    assert not any(event.event_type == "tool.started" for event in events)
    assert failed.metadata["stage"] == "tool_protocol"
    assert failed.metadata["error_code"] == "tool_protocol_violation"
    assert failed.metadata["tool_call_count"] == 0


def test_tool_calls_finish_reason_with_empty_calls_fails_without_second_llm_round(
    tmp_path: Path,
) -> None:
    class EmptyToolCallsProvider:
        name = "empty-tool-calls-provider"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls += 1
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[],
            )

    class NoopTool:
        name = "noop"
        description = "Noop."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="noop")

    registry = ToolRegistry()
    registry.register(NoopTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = EmptyToolCallsProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    with pytest.raises(ToolProtocolError, match="finish_reason=tool_calls"):
        agent.respond("noop", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    failed = next(event for event in events if event.event_type == "turn.failed")
    llm_started = [
        event for event in events if event.event_type == "llm.started"
    ]
    assert provider.calls == 1
    assert len(llm_started) == 1
    assert not any(event.event_type == "tool.started" for event in events)
    assert failed.metadata["stage"] == "tool_protocol"
    assert failed.metadata["error_code"] == "tool_protocol_violation"
    assert failed.metadata["provider_tool_call_count"] == 0


def test_provider_tool_call_parse_error_becomes_tool_result_and_model_recovers(
    tmp_path: Path,
) -> None:
    class CountingTool:
        name = "lookup"
        description = "Lookup something."

        def __init__(self) -> None:
            self.run_count = 0

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            self.run_count += 1
            return ToolResult(name=self.name, content="should not run")

    class InvalidArgumentsProvider:
        name = "invalid-arguments"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls += 1
            if self.calls == 2:
                return LLMResponse(
                    content="recovered after tool error",
                    model="mock",
                    provider=self.name,
                    finish_reason="stop",
                )
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_bad",
                        name="lookup",
                        arguments={},
                        raw_arguments='{"query":',
                        metadata={
                            "arguments_parse_error": "Expecting value",
                            "raw_arguments": '{"query":',
                        },
                    )
                ],
            )

    tool = CountingTool()
    registry = ToolRegistry()
    registry.register(tool)
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = InvalidArgumentsProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    result = agent.respond("lookup", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.event_type == "tool.failed"
    )
    assert result.response == "recovered after tool error"
    assert provider.calls == 2
    assert tool.run_count == 0
    assert tool_failed.metadata["tool_name"] == "lookup"
    assert tool_failed.metadata["error_type"] == "ToolExecutionError"
    assert '"failed":true' in tool_failed.content
    assert '"error_type":"ToolExecutionError"' in tool_failed.content
    assert not any(event.event_type == "turn.failed" for event in events)


def test_provider_unknown_tool_becomes_tool_result_and_model_recovers(
    tmp_path: Path,
) -> None:
    class UnknownToolProvider:
        name = "unknown-tool-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            if len(self.calls) == 2:
                return LLMResponse(
                    content="recovered after unknown tool",
                    model="mock",
                    provider=self.name,
                    finish_reason="stop",
                )
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_missing",
                        name="missing_provider_tool",
                        arguments={"query": "alpha"},
                        raw_arguments='{"query":"alpha"}',
                    )
                ],
            )

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = UnknownToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=ToolRegistry(),
    )

    result = agent.respond("use missing tool", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.event_type == "tool.failed"
    )
    follow_up_messages = provider.calls[1][0]
    assert result.response == "recovered after unknown tool"
    assert len(provider.calls) == 2
    assert tool_failed.metadata["tool_name"] == "missing_provider_tool"
    assert tool_failed.metadata["error_type"] == "ToolExecutionError"
    assert '"failed":true' in tool_failed.content
    assert '"error":"Unknown tool: missing_provider_tool"' in tool_failed.content
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_call_id"] == "call_missing"
    assert follow_up_messages[-1]["content"] == tool_failed.content
    assert not any(event.event_type == "turn.failed" for event in events)


def test_provider_tool_runtime_exception_becomes_tool_result_and_model_recovers(
    tmp_path: Path,
) -> None:
    class ExplodingTool:
        name = "explode_provider"
        description = "Raise during provider-requested run."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            raise RuntimeError("provider boom")

    class ExplodingToolProvider:
        name = "exploding-tool-provider"

        def __init__(self) -> None:
            self.calls: list[tuple[list[ChatMessage], dict[str, Any]]] = []

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls.append((messages, kwargs))
            if len(self.calls) == 2:
                return LLMResponse(
                    content="recovered after runtime exception",
                    model="mock",
                    provider=self.name,
                    finish_reason="stop",
                )
            return LLMResponse(
                content="",
                model="mock",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_explode",
                        name="explode_provider",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

    registry = ToolRegistry()
    registry.register(ExplodingTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = ExplodingToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )

    result = agent.respond("use exploding tool", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.event_type == "tool.failed"
    )
    follow_up_messages = provider.calls[1][0]
    assert result.response == "recovered after runtime exception"
    assert len(provider.calls) == 2
    assert tool_failed.metadata["tool_name"] == "explode_provider"
    assert tool_failed.metadata["error_type"] == "RuntimeError"
    assert '"failed":true' in tool_failed.content
    assert '"error":"provider boom"' in tool_failed.content
    assert '"error_type":"RuntimeError"' in tool_failed.content
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_call_id"] == "call_explode"
    assert follow_up_messages[-1]["content"] == tool_failed.content
    assert not any(event.event_type == "turn.failed" for event in events)


def test_agent_retries_transient_llm_errors_with_bounded_count(tmp_path: Path) -> None:
    class FlakyProvider:
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            self.calls += 1
            if self.calls == 1:
                raise httpx.TimeoutException("temporary timeout")
            return LLMResponse(content="recovered", model="mock", provider=self.name)

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    provider = FlakyProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        retriever=MemoryRetriever(store),
        max_llm_retries=2,
    )

    result = agent.respond("hello", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    completed = next(
        event for event in events if event.event_type == "llm.completed"
    )
    assert result.response == "recovered"
    assert provider.calls == 2
    assert result.debug["llm_retry_count"] == 1
    assert completed.metadata["retry_count"] == 1


def test_agent_cancellation_writes_failed_event_and_clears_flag(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        retriever=MemoryRetriever(store),
    )
    agent.cancel("s1")

    with pytest.raises(AgentCanceledError):
        agent.respond("hello", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    failed = next(event for event in events if event.event_type == "turn.failed")
    assert failed.metadata["status"] == "canceled"
    assert failed.metadata["stage"] == "before_user_event"
    assert not agent.is_canceled("s1")


def test_provider_tool_cancellation_after_run_records_tool_failed(
    tmp_path: Path,
) -> None:
    class ToolCallingProvider:
        name = "tool-calling"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            return LLMResponse(
                content="",
                provider=self.name,
                model="mock",
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_cancel",
                        name="cancel_after_run",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )

    class CancelAfterRunTool:
        name = "cancel_after_run"
        description = "Cancel after successful run."

        def __init__(self, agent: AlphaAgent) -> None:
            self.agent = agent

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            self.agent.cancel("s1")
            return ToolResult(name=self.name, content="done")

    registry = ToolRegistry()
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=ToolCallingProvider(),
        retriever=MemoryRetriever(store),
        tool_registry=registry,
    )
    registry.register(CancelAfterRunTool(agent))

    with pytest.raises(AgentCanceledError):
        agent.respond("use the tool", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    tool_failed = next(
        event for event in events if event.event_type == "tool.failed"
    )
    turn_failed = next(
        event for event in events if event.event_type == "turn.failed"
    )
    assert tool_failed.metadata["tool_name"] == "cancel_after_run"
    assert tool_failed.metadata["error_type"] == "AgentCanceledError"
    assert turn_failed.metadata["status"] == "canceled"
    assert turn_failed.metadata["stage"] == "after_tool"


def test_agent_retry_exhaustion_records_llm_stage_and_retry_count(tmp_path: Path) -> None:
    class AlwaysTimeoutProvider:
        name = "always-timeout"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise httpx.TimeoutException("temporary timeout")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=AlwaysTimeoutProvider(),
        retriever=MemoryRetriever(store),
        max_llm_retries=2,
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    failed = next(event for event in events if event.event_type == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 2


def test_agent_zero_retry_transient_failure_records_llm_stage(tmp_path: Path) -> None:
    class TimeoutProvider:
        name = "timeout"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise httpx.TimeoutException("temporary timeout")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=TimeoutProvider(),
        retriever=MemoryRetriever(store),
        max_llm_retries=0,
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    failed = next(event for event in events if event.event_type == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 0


def test_agent_non_transient_provider_failure_records_llm_stage(tmp_path: Path) -> None:
    class BadProvider:
        name = "bad"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise ValueError("bad request")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    agent = AlphaAgent(
        store=store,
        llm_provider=BadProvider(),
        retriever=MemoryRetriever(store),
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_runtime_traces(session_id="s1", limit=20)
    failed = next(event for event in events if event.event_type == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 0
