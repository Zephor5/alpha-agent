from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import cast

import pytest

from alpha_agent.cognition.coordinator import LockBusy, LoopAcquireRequest, LoopCoordinator
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    SUBJECT_SELF,
    CognitiveEventKind,
    CounterpartId,
    Instant,
    counterpart_ref,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.config import LLMContextConfig
from alpha_agent.llm.base import (
    AssistantChatMessage,
    ChatMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinition,
    LLMToolDefinitionInput,
    ToolChatMessage,
)
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent, ContextWindowExceededError
from alpha_agent.runtime.context_handover import DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION
from alpha_agent.runtime.counterpart_router import DEFAULT_COUNTERPART_ID
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import Tool, ToolExecutionContext, ToolResult
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME
from alpha_agent.tools.registry import ToolRegistry
from tests.cognition.test_belief_projection_apply import belief


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _copy_chat_message(message: ChatMessage) -> ChatMessage:
    return cast(ChatMessage, dict(message))


def test_agent_responds_and_persists_session_messages(tmp_path) -> None:
    store = _store(tmp_path)
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())

    result = agent.respond("hello", session_id="s1")

    assert result.session_id == "s1"
    assert result.response == "Mock response: I heard you say: hello."
    assert result.debug["note"] == "runtime-owned foreground turn"
    messages = store.list_session_messages("s1")
    assert [message.kind for message in messages] == ["user_message", "assistant_message"]
    assert [message.llm_role for message in messages] == ["user", "assistant"]
    assert messages[0].raw_content == "hello"
    assert messages[1].raw_content == result.response
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "llm.started",
        "llm.completed",
    ]


def test_agent_replays_source_stream_without_foreground_duplicate_for_llm_input(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    provider = _QueuedRecordingProvider(
        [
            "Hello! How can I assist you today?",
            "Let's find something interesting to do.",
        ]
    )
    agent = AlphaAgent(store=store, llm_provider=provider)
    agent.respond("hello", session_id="s1")

    result = agent.respond("I'm bored", session_id="s1")

    assert result.response == "Let's find something interesting to do."
    second_call = provider.calls[-1]
    assert [message["role"] for message in second_call] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert str(second_call[0]["content"]).startswith("Identity: Alpha Agent")
    assert second_call[1:] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "I'm bored"},
    ]
    rendered_contents = "\n".join(str(message.get("content", "")) for message in second_call)
    assert "Foreground:" not in rendered_contents


def test_agent_two_turn_prompt_is_append_only_without_fixed_tail(tmp_path) -> None:
    store = _store(tmp_path)
    for index in range(1, 12):
        store.append_session_message(
            session_id="s1",
            kind="user_message" if index % 2 else "assistant_message",
            llm_role="user" if index % 2 else "assistant",
            raw_content=f"old {index}",
        )
    provider = _RecordingProvider("done")
    agent = AlphaAgent(store=store, llm_provider=provider)

    agent.respond("new turn", session_id="s1")

    first_call = provider.calls[0]
    contents = [message.get("content") for message in first_call]
    assert contents[1:] == [*(f"old {index}" for index in range(1, 12)), "new turn"]


def test_agent_reuses_session_profile_snapshot_before_history(tmp_path) -> None:
    store = _store(tmp_path)
    log = SQLiteEventLog(store)
    projection = _seed_active_digest(
        store,
        log,
        "belief:digest:v1",
        "Stable profile v1.",
    )
    provider = _QueuedRecordingProvider(["first answer", "second answer"])
    agent = AlphaAgent(store=store, llm_provider=provider, event_log=log)

    agent.respond("first turn", session_id="s1")

    _seed_active_digest(
        store,
        log,
        "belief:digest:v2",
        "Stable profile v2.",
        projection=projection,
        held_since="2026-01-01T00:00:01+00:00",
    )
    agent.respond("second turn", session_id="s1")

    first_call = provider.calls[0]
    assert [message["role"] for message in first_call] == ["system", "user", "user"]
    assert "Counterpart profile: Stable profile v1." in str(first_call[1]["content"])
    assert first_call[2]["content"] == "first turn"

    second_call = provider.calls[1]
    assert [message["role"] for message in second_call] == [
        "system",
        "user",
        "user",
        "assistant",
        "user",
    ]
    assert "Counterpart profile: Stable profile v1." in str(second_call[1]["content"])
    assert "Stable profile v2." not in str(second_call)
    assert [message.get("content") for message in second_call[2:]] == [
        "first turn",
        "first answer",
        "second turn",
    ]
    snapshot = store.get_session_profile_snapshot("s1")
    assert snapshot is not None
    assert snapshot.content == "Stable profile v1."


def test_session_profile_snapshots_are_keyed_by_session(tmp_path) -> None:
    store = _store(tmp_path)

    first = store.create_session_profile_snapshot(
        session_id="s1",
        counterpart_id="counterpart:a",
        source_belief_id="belief:digest:a",
        content="Profile A.",
    )
    second = store.create_session_profile_snapshot(
        session_id="s2",
        counterpart_id="counterpart:b",
        source_belief_id="belief:digest:b",
        content="Profile B.",
    )
    duplicate = store.create_session_profile_snapshot(
        session_id="s1",
        counterpart_id="counterpart:b",
        source_belief_id="belief:digest:a-new",
        content="Profile A new.",
    )

    assert first.content == "Profile A."
    assert second.content == "Profile B."
    assert duplicate.content == "Profile A."
    assert store.get_session_profile_snapshot("s1") == first
    assert store.get_session_profile_snapshot("s2") == second
    assert store.get_session_profile_snapshot("missing") is None


def test_agent_binds_session_counterpart_and_reuses_session_profile(tmp_path) -> None:
    store = _store(tmp_path)
    log = SQLiteEventLog(store)
    bob_counterpart_id = _routed_counterpart_id("local", "bob")
    _seed_active_digest(
        store,
        log,
        "belief:digest:alice",
        "Alice stable profile.",
    )
    _seed_active_digest(
        store,
        log,
        "belief:digest:bob",
        "Bob stable profile.",
        counterpart_id=bob_counterpart_id,
    )
    provider = _QueuedRecordingProvider(["alice answer", "bob answer"])
    agent = AlphaAgent(store=store, llm_provider=provider, event_log=log)

    agent.respond(
        "first turn",
        session_id="shared",
        source_metadata={"platform": "local", "user_id": "alice"},
    )
    agent.respond(
        "second turn",
        session_id="shared",
        source_metadata={"platform": "local", "user_id": "bob"},
    )

    assert "Counterpart profile: Alice stable profile." in str(provider.calls[0])
    assert "Bob stable profile." not in str(provider.calls[0])
    assert "Counterpart profile: Alice stable profile." in str(provider.calls[1])
    assert "Bob stable profile." not in str(provider.calls[1])
    binding = store.get_session_counterpart("shared")
    assert binding is not None
    assert binding.counterpart_id == str(DEFAULT_COUNTERPART_ID)
    snapshot = store.get_session_profile_snapshot("shared")
    assert snapshot is not None
    assert snapshot.content == "Alice stable profile."
    assert store.get_session_counterpart("missing") is None


def test_first_turn_profile_snapshot_participates_in_pre_user_budget(tmp_path) -> None:
    store = _store(tmp_path)
    log = SQLiteEventLog(store)
    _seed_active_digest(
        store,
        log,
        "belief:digest:large",
        " ".join(f"profileword{index}" for index in range(80)),
    )
    provider = _RecordingProvider("should not be called")
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        tool_registry=ToolRegistry(),
        llm_context_config=_zero_reserve_context(),
        max_context_tokens=55,
        event_log=log,
    )

    with pytest.raises(ContextWindowExceededError):
        agent.respond("short", session_id="s1")

    assert provider.calls == []
    assert store.list_session_messages("s1") == []
    snapshot = store.get_session_profile_snapshot("s1")
    assert snapshot is not None
    assert snapshot.content.startswith("profileword0")


def test_agent_executes_provider_tool_calls_and_stores_tool_round(tmp_path) -> None:
    store = _store(tmp_path)
    registry = build_tool_registry()
    registry.register(_EchoTool())
    provider = _ToolCallingProvider()
    agent = AlphaAgent(store=store, llm_provider=provider, tool_registry=registry)

    result = agent.respond("use tool", session_id="s1")

    assert result.response == "final answer"
    turn_id = result.debug["turn_id"]
    assert isinstance(turn_id, str) and turn_id.startswith("turn_")
    assert result.debug["tool_call_count"] == 1
    messages = store.list_session_messages("s1")
    assert [message.kind for message in messages] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    assert [message.metadata["turn_id"] for message in messages] == [
        turn_id,
        turn_id,
        turn_id,
        turn_id,
    ]
    traces = store.list_runtime_traces("s1")
    assert [trace.event_type for trace in traces] == [
        "llm.started",
        "llm.completed",
        "tool.started",
        "tool.completed",
        "llm.started",
        "llm.completed",
    ]
    assert [trace.metadata["turn_id"] for trace in traces] == [
        turn_id,
        turn_id,
        turn_id,
        turn_id,
        turn_id,
        turn_id,
    ]
    started_trace = traces[2]
    assert json.loads(started_trace.content)["arguments"] == {"text": "<trace-safe>"}
    assert started_trace.metadata["call"]["arguments"] == {"text": "<trace-safe>"}
    assert "raw_arguments" not in started_trace.metadata["call"]["metadata"]

    events = list(SQLiteEventLog(store).iter())
    foreground_events = [
        event
        for event in events
        if event.kind
        in {
            CognitiveEventKind.PERCEIVED,
            CognitiveEventKind.ACTED,
            CognitiveEventKind.TURN_SOURCES_RECORDED,
        }
    ]
    assert [event.kind for event in foreground_events] == [
        CognitiveEventKind.PERCEIVED,
        CognitiveEventKind.ACTED,
        CognitiveEventKind.TURN_SOURCES_RECORDED,
    ]
    assert [event.payload["turn_id"] for event in foreground_events] == [
        turn_id,
        turn_id,
        turn_id,
    ]
    assert [event.payload["session_id"] for event in foreground_events] == ["s1", "s1", "s1"]
    assert set(foreground_events[0].payload) == {
        "turn_id",
        "session_id",
        "stimulus_kind",
        "source",
        "from_counterpart",
        "source_refs",
        "content_digest",
        "content_length",
    }
    acted = foreground_events[1]
    source_recorded = foreground_events[2]
    assert acted.payload["llm_call_ids"] == result.debug["llm_call_ids"]
    assert acted.payload["llm_trace_ids"] == result.debug["llm_trace_ids"]
    assert acted.payload["tool_call_ids"] == ["call_1"]
    assert acted.payload["tool_result_trace_ids"] == [traces[3].id]
    assert source_recorded.payload["llm_call_ids"] == result.debug["llm_call_ids"]
    assert source_recorded.payload["llm_trace_ids"] == result.debug["llm_trace_ids"]
    assert source_recorded.payload["provider_tool_message_ids"] == [
        messages[1].id,
        messages[2].id,
    ]
    assert source_recorded.payload["provider_tool_trace_ids"] == [traces[3].id]


def test_busy_attempt_does_not_allocate_completed_turn_audit(tmp_path) -> None:
    store = _store(tmp_path)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        coordinator=_BusyCoordinator(),
    )

    result = agent.respond("hello", session_id="s1")

    assert result.debug["busy"] is True
    assert "turn_id" not in result.debug
    assert store.list_session_messages("s1") == []
    assert store.list_runtime_traces("s1") == []
    assert list(
        SQLiteEventLog(store).iter(
            kinds=[
                CognitiveEventKind.PERCEIVED,
                CognitiveEventKind.ACTED,
                CognitiveEventKind.TURN_SOURCES_RECORDED,
            ]
        )
    ) == []


def test_agent_sends_only_structured_tool_output_to_llm_context(tmp_path) -> None:
    store = _store(tmp_path)
    registry = build_tool_registry()
    registry.register(_StructuredTool())
    provider = _StructuredToolCallingProvider()
    agent = AlphaAgent(store=store, llm_provider=provider, tool_registry=registry)

    result = agent.respond("use structured tool", session_id="s1")

    assert result.response == "final answer"
    tool_message = cast(ToolChatMessage, provider.calls[1][-1])
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == {
        "items": [{"title": "Alpha"}],
        "ok": True,
    }
    assert "structured" not in tool_message["content"]
    assert "metadata" not in tool_message["content"]

    persisted = store.list_session_messages("s1")[2]
    assert json.loads(persisted.raw_content) == {
        "items": [{"title": "Alpha"}],
        "ok": True,
    }
    assert persisted.provider_metadata == {"tool_name": "structured"}
    assert persisted.metadata["result_metadata"] == {"source": "test"}
    assert persisted.metadata["tool_output_kind"] == "json"

    completed_trace = store.list_runtime_traces("s1")[3]
    assert completed_trace.event_type == "tool.completed"
    assert json.loads(completed_trace.content) == {
        "items": [{"title": "Alpha"}],
        "ok": True,
    }
    assert completed_trace.metadata["result"] == {
        "metadata": {"source": "test"},
        "name": "structured",
        "output": {"items": [{"title": "Alpha"}], "ok": True},
    }


def test_memory_recall_result_enters_follow_up_llm_and_persists(tmp_path) -> None:
    store = _store(tmp_path)
    log = SQLiteEventLog(store)
    _seed_active_belief(
        store,
        log,
        "belief:python",
        "User prefers Python examples.",
        object_="python",
    )
    provider = _MemoryRecallCallingProvider()
    agent = AlphaAgent(store=store, llm_provider=provider, event_log=log)

    result = agent.respond("What examples do I prefer?", session_id="s1")

    assert result.response == "You prefer Python examples."
    assert MEMORY_RECALL_TOOL_NAME in provider.tool_names_seen[0]
    assert len(provider.calls) == 2
    follow_up_tool_message = cast(ToolChatMessage, provider.calls[1][-1])
    assert follow_up_tool_message["role"] == "tool"
    assert follow_up_tool_message["tool_call_id"] == "call_recall"
    assert json.loads(follow_up_tool_message["content"]) == {
        "results": [
            {
                "id": "belief:python",
                "content": "User prefers Python examples.",
                "type": "preference",
                "scope": "counterpart",
                "status": "active",
                "held_since": "2026-01-01T00:00:00+00:00",
            }
        ]
    }

    messages = store.list_session_messages("s1")
    assert [message.kind for message in messages] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    turn_id = result.debug["turn_id"]
    assert messages[0].metadata == {"turn_id": turn_id}
    assert messages[1].metadata == {"turn_id": turn_id, "tool_call_ids": ["call_recall"]}
    assert messages[2].provider_metadata == {"tool_name": MEMORY_RECALL_TOOL_NAME}
    assert json.loads(messages[2].raw_content) == {
        "results": [
            {
                "id": "belief:python",
                "content": "User prefers Python examples.",
                "type": "preference",
                "scope": "counterpart",
                "status": "active",
                "held_since": "2026-01-01T00:00:00+00:00",
            }
        ]
    }
    assert messages[2].metadata["tool_output_kind"] == "json"
    assert messages[2].metadata["turn_id"] == turn_id
    assert messages[3].metadata == {"turn_id": turn_id}
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "llm.started",
        "llm.completed",
        "tool.started",
        "tool.completed",
        "llm.started",
        "llm.completed",
    ]


def test_llm_debug_payloads_are_written_to_jsonl_not_database(tmp_path) -> None:
    store = _store(tmp_path)
    trace_log = tmp_path / "llm.jsonl"
    agent = AlphaAgent(
        store=store,
        llm_provider=_RawMetadataProvider(),
        llm_debug_logging=True,
        llm_trace_log_path=trace_log,
    )

    agent.respond("secret input", session_id="s1")

    traces = store.list_runtime_traces("s1")
    trace_json = json.dumps([trace.metadata for trace in traces], sort_keys=True)
    assert "secret input" not in trace_json
    assert "secret prompt payload" not in trace_json
    assert "secret response payload" not in trace_json
    assert "request_payload" not in trace_json
    assert "response_payload" not in trace_json

    assistant = [
        message
        for message in store.list_session_messages("s1")
        if message.llm_role == "assistant"
    ][0]
    provider_metadata_json = json.dumps(assistant.provider_metadata, sort_keys=True)
    assert "secret response payload" not in provider_metadata_json
    assert "request_payload" not in provider_metadata_json
    assert "response_payload" not in provider_metadata_json

    log_entries = [json.loads(line) for line in trace_log.read_text(encoding="utf-8").splitlines()]
    assert [entry["event"] for entry in log_entries] == ["llm.request", "llm.response"]
    log_json = json.dumps(log_entries, sort_keys=True)
    assert "secret input" in log_json
    assert "secret response payload" in log_json
    response_log_json = json.dumps(log_entries[1], sort_keys=True)
    assert "secret prompt payload" not in response_log_json
    assert "request_payload" not in response_log_json
    assert "response_payload" not in response_log_json
    assert log_entries[1]["metadata"]["response"] == {
        "output_text": "secret response payload"
    }


def test_agent_cancel_before_turn_raises_and_clears_flag(tmp_path) -> None:
    store = _store(tmp_path)
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())
    agent.cancel("s1")

    with pytest.raises(Exception, match="canceled"):
        agent.respond("hello", session_id="s1")

    assert agent.is_canceled("s1") is False
    assert store.list_runtime_traces("s1", event_type="turn.failed")


def test_pending_user_too_large_rejected_without_persisting_message(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _QueuedRecordingProvider(["should not be called"])
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        llm_context_config=_zero_reserve_context(),
        max_context_tokens=4,
    )

    with pytest.raises(RuntimeError, match="pending user message exceeds"):
        agent.respond("one two three four five six", session_id="s1")

    assert provider.calls == []
    assert store.list_session_messages("s1") == []


def test_pre_user_compression_runs_before_pending_user_and_excludes_it(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="old source one two three four",
    )
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="old answer five six seven eight",
    )
    provider = _QueuedRecordingProvider(["pre-user handover", "final answer"])
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        llm_context_config=_compression_context(),
        max_context_tokens=340,
    )

    result = agent.respond("pending user must stay out of compression", session_id="s1")

    assert result.response == "final answer"
    assert len(provider.calls) == 2
    compression_call = provider.calls[0]
    assert compression_call[0]["role"] == "system"
    assert "Identity: Alpha Agent" in str(compression_call[0]["content"])
    assert "old source one two three four" in str(compression_call)
    assert "pending user must stay out of compression" not in str(compression_call)
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in str(compression_call[-1]["content"])

    persisted = store.list_session_messages("s1")
    assert [message.kind for message in persisted] == [
        "user_message",
        "assistant_message",
        "compressed_message",
        "user_message",
        "assistant_message",
    ]
    assert persisted[2].raw_content.startswith("<system-reminder>")
    assert persisted[2].ordinal < persisted[3].ordinal
    assert persisted[3].raw_content == "pending user must stay out of compression"
    assert all(
        DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in message.raw_content
        for message in persisted
    )


def test_failed_pre_user_compression_does_not_persist_pending_user(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="old source one two three four",
    )
    before = store.list_session_messages("s1")
    provider = _FailingOnceProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        llm_context_config=_compression_context(),
        max_context_tokens=340,
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        agent.respond("pending user must not be persisted", session_id="s1")

    assert len(provider.calls) == 1
    assert "pending user must not be persisted" not in str(provider.calls[0])
    assert store.list_session_messages("s1") == before
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "handover_compression.started",
        "handover_compression.failed",
        "turn.failed",
    ]


def test_tool_loop_compression_waits_for_tool_result_and_rebuilds_next_prompt(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    registry = build_tool_registry()
    registry.register(_EchoTool())
    provider = _ToolLoopCompressionProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        tool_registry=registry,
        llm_context_config=_compression_context(),
        max_context_tokens=340,
    )

    result = agent.respond("use tool", session_id="s1")

    assert result.response == "final answer"
    assert len(provider.calls) == 3
    first_call = provider.calls[0]
    assert [message["role"] for message in first_call] == ["system", "user"]
    assert "Identity: Alpha Agent" in str(first_call[0]["content"])
    assert first_call[1]["content"] == "use tool"

    compression_call = provider.calls[1]
    assert [message["role"] for message in compression_call] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert compression_call[0] == first_call[0]
    assert compression_call[1]["content"] == "use tool"
    compression_assistant = cast(AssistantChatMessage, compression_call[2])
    compression_tool = cast(ToolChatMessage, compression_call[3])
    assert compression_assistant["tool_calls"][0]["id"] == "call_1"
    assert compression_tool["tool_call_id"] == "call_1"
    assert compression_tool["content"] == "complete tool result: hello"
    assert compression_call[-1]["role"] == "user"
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in str(compression_call[-1]["content"])

    next_call = provider.calls[2]
    assert [message["role"] for message in next_call] == ["system", "user"]
    assert next_call[0] == first_call[0]
    assert next_call[1]["role"] == "user"
    assert "tool-loop handover" in str(next_call[1]["content"])
    assert DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in str(next_call)
    assert "use tool" not in str(next_call)
    assert "complete tool result: hello" not in str(next_call)
    assert "call_1" not in str(next_call)

    persisted = store.list_session_messages("s1")
    assert [message.kind for message in persisted] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "compressed_message",
        "assistant_message",
    ]
    assert persisted[1].tool_calls[0]["id"] == "call_1"
    assert persisted[2].tool_call_id == "call_1"
    assert persisted[2].raw_content == "complete tool result: hello"
    assert persisted[2].provider_metadata == {"tool_name": "echo"}
    assert persisted[2].metadata["result_metadata"] == {}
    assert persisted[2].metadata["tool_output_kind"] == "text"
    assert persisted[3].raw_content.startswith("<system-reminder>")
    assert persisted[3].raw_content.endswith("</system-reminder>")
    assert persisted[3].compression_point_ordinal == persisted[2].ordinal
    assert all(
        DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION not in message.raw_content
        for message in persisted
    )


class _RecordingProvider:
    name = "recording"

    def __init__(self, response: str):
        self.response = response
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(content=self.response, model="test", provider=self.name)


def _seed_active_digest(
    store: StateStore,
    log: SQLiteEventLog,
    belief_id: str,
    content: str,
    *,
    projection: BeliefProjection | None = None,
    counterpart_id: str = str(DEFAULT_COUNTERPART_ID),
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> BeliefProjection:
    projection = projection or BeliefProjection(store)
    counterpart = counterpart_ref(CounterpartId(counterpart_id))
    event = EventEmitter(log).emit(
        CognitiveEventKind.BELIEF_FORMED,
        payload={
            "belief": belief(
                belief_id,
                content,
                about=[counterpart],
                object_=f"counterpart_digest:{counterpart_id}",
                held_since=held_since,
            ).to_record()
        },
    )
    projection.apply(event)
    return projection


def _routed_counterpart_id(platform: str, user_id: str) -> str:
    digest = hashlib.sha1(f"{platform}:{user_id}".encode()).hexdigest()[:16]
    return f"counterpart:{digest}"


def _seed_active_belief(
    store: StateStore,
    log: SQLiteEventLog,
    belief_id: str,
    content: str,
    *,
    object_: str,
    held_since: str = "2026-01-01T00:00:00+00:00",
) -> BeliefProjection:
    projection = BeliefProjection(store)
    counterpart = counterpart_ref(CounterpartId(str(DEFAULT_COUNTERPART_ID)))
    event = EventEmitter(log).emit(
        CognitiveEventKind.BELIEF_FORMED,
        payload={
            "belief": belief(
                belief_id,
                content,
                about=[counterpart],
                object_=object_,
                held_since=held_since,
            ).to_record()
        },
    )
    projection.apply(event)
    return projection


class _QueuedRecordingProvider:
    name = "recording"

    def __init__(self, responses: Sequence[str]):
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del tools, tool_choice
        self.calls.append(messages)
        response = self.responses.pop(0)
        return LLMResponse(content=response, model="test", provider=self.name)


class _FailingOnceProvider:
    name = "failing"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del tools, tool_choice
        self.calls.append(messages)
        raise RuntimeError("provider failed")


class _BusyCoordinator(LoopCoordinator):
    def __init__(self) -> None:
        super().__init__(SUBJECT_SELF)

    @contextmanager
    def try_acquire(self, _req: LoopAcquireRequest) -> Iterator[None]:
        raise LockBusy("background", Instant("2026-01-01T00:00:00+00:00"))
        yield


class _RawMetadataProvider:
    name = "raw-provider"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content="final answer",
            model="test",
            provider=self.name,
            finish_reason="stop",
            metadata={
                "response_id": "resp-1",
                "finish_reason": "stop",
                "request_payload": {
                    "messages": [{"role": "user", "content": "secret prompt payload"}],
                },
                "response_payload": {"output_text": "secret response payload"},
            },
        )


class _ToolCallingProvider:
    name = "tool-provider"

    def __init__(self):
        self.calls = 0

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        name="echo",
                        arguments={"text": "hello"},
                        raw_arguments='{"text":"hello"}',
                    )
                ],
            )
        return LLMResponse(content="final answer", model="test", provider=self.name)


class _ToolLoopCompressionProvider:
    name = "tool-compression-provider"

    def __init__(self):
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del tools, tool_choice
        self.calls.append([_copy_chat_message(message) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        name="echo",
                        arguments={"text": "hello"},
                        raw_arguments='{"text":"hello"}',
                    )
                ],
            )
        if DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION in str(messages[-1].get("content")):
            return LLMResponse(content="tool-loop handover", model="test", provider=self.name)
        return LLMResponse(content="final answer", model="test", provider=self.name)


class _StructuredToolCallingProvider:
    name = "structured-tool-provider"

    def __init__(self):
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del tools, tool_choice
        self.calls.append([_copy_chat_message(message) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_1",
                        name="structured",
                        arguments={},
                        raw_arguments="{}",
                    )
                ],
            )
        return LLMResponse(content="final answer", model="test", provider=self.name)


class _MemoryRecallCallingProvider:
    name = "memory-recall-provider"

    def __init__(self) -> None:
        self.call_count = 0
        self.calls: list[list[ChatMessage]] = []
        self.tool_names_seen: list[list[str]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del tool_choice
        self.call_count += 1
        self.calls.append([_copy_chat_message(message) for message in messages])
        self.tool_names_seen.append([_tool_name(tool) for tool in tools or []])
        if self.call_count == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_recall",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={
                            "query": "what examples do I prefer?",
                            "scope": "counterpart",
                        },
                        raw_arguments=(
                            '{"query":"what examples do I prefer?",'
                            '"scope":"counterpart"}'
                        ),
                    )
                ],
            )
        return LLMResponse(
            content="You prefer Python examples.",
            model="test",
            provider=self.name,
        )


def _tool_name(tool: LLMToolDefinitionInput) -> str:
    if isinstance(tool, LLMToolDefinition):
        return tool.name
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")


class _EchoTool(Tool):
    name = "echo"
    description = "Echo input."

    def run(self, arguments, context: ToolExecutionContext):
        del context
        return ToolResult(
            name=self.name,
            output=f"complete tool result: {arguments['text']}",
            metadata={},
        )

    def trace_arguments(self, arguments):
        del arguments
        return {"text": "<trace-safe>"}


class _StructuredTool(Tool):
    name = "structured"
    description = "Return structured output."

    def run(self, arguments, context: ToolExecutionContext):
        del arguments, context
        return ToolResult(
            name=self.name,
            output={"ok": True, "items": [{"title": "Alpha"}]},
            metadata={"source": "test"},
        )


def _zero_reserve_context() -> LLMContextConfig:
    return LLMContextConfig(
        tool_truncate_threshold_ratio=1.0,
        handover_compress_threshold_ratio=1.0,
        minimum_remaining_tokens=0,
        expected_output_reserve_tokens=0,
        safety_margin_tokens=0,
    )


def _compression_context() -> LLMContextConfig:
    return LLMContextConfig(
        tool_truncate_threshold_ratio=1.0,
        handover_compress_threshold_ratio=0.01,
        minimum_remaining_tokens=0,
        expected_output_reserve_tokens=0,
        safety_margin_tokens=0,
    )
