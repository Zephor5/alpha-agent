from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.llm.base import ChatMessage, LLMResponse, LLMToolCall
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.memory.models import RetrievedContext
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.runtime.agent import (
    AgentCanceledError,
    AlphaAgent,
    LLMCallError,
    ToolLoopLimitExceeded,
    ToolProtocolError,
)
from alpha_agent.runtime.tools import ToolExecutionError
from alpha_agent.tools.base import ToolCall, ToolResult
from alpha_agent.tools.registry import ToolRegistry


def test_mock_agent_loop_stores_user_and_assistant_events(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    ProceduralMemoryManager(store).load_builtin_skills()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
    )

    result = agent.respond("remember that I prefer concise answers", session_id="s1")

    events = store.list_events(session_id="s1")
    semantic = store.list_semantic_memories()
    assert "Mock response" in result.response
    assert [event.role for event in events if event.role in {"user", "assistant"}] == [
        "assistant",
        "user",
    ]
    assert {
        event.metadata.get("event_type")
        for event in events
        if event.metadata.get("event_type")
    } >= {
        "turn.started",
        "memory.retrieved",
        "llm.started",
        "llm.completed",
        "memory.extracted",
        "turn.completed",
    }
    assert len(semantic) == 1
    assert result.debug["extracted_memory_count"] >= 1


def test_agent_honors_configured_retrieval_limit(tmp_path: Path) -> None:
    class RecordingRetriever(MemoryRetriever):
        def __init__(self, store: MemoryStore, working: WorkingMemoryManager):
            super().__init__(store, working)
            self.seen_limit: int | None = None

        def retrieve_context(
            self,
            query: str,
            session_id: str,
            limit: int = 8,
        ) -> RetrievedContext:
            self.seen_limit = limit
            return super().retrieve_context(query, session_id, limit)

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    retriever = RecordingRetriever(store, working)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=retriever,
        retrieval_limit=2,
    )

    agent.respond("hello", session_id="s1")

    assert retriever.seen_limit == 2


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
    working = WorkingMemoryManager(store)
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
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
    working = WorkingMemoryManager(store)
    provider = RecordingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    agent.respond("hello", session_id="s1")

    assert provider.kwargs[0]["tool_choice"] == "auto"
    assert [tool.name for tool in provider.kwargs[0]["tools"]] == ["echo"]
    assert provider.kwargs[0]["tools"][0].parameters == EchoTool.parameters


def test_agent_records_structured_event_sequence(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
    )

    result = agent.respond("hello", session_id="s1")

    events = list(reversed(store.list_events(session_id="s1", limit=20)))
    event_types = [
        str(event.metadata["event_type"])
        for event in events
        if "event_type" in event.metadata
    ]
    assert event_types == [
        "turn.started",
        "memory.retrieved",
        "llm.started",
        "llm.completed",
        "memory.extracted",
        "turn.completed",
    ]
    completed = next(
        event for event in events if event.metadata.get("event_type") == "turn.completed"
    )
    assert completed.metadata["assistant_response_event_id"]
    assert completed.metadata["retry_count"] == 0
    assert result.debug["delivery_event_id"] == completed.id


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
    working = WorkingMemoryManager(store)
    provider = ToolCallingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
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
    events = store.list_events(session_id="s1", limit=30)
    llm_completed_rounds = [
        event.metadata["round"]
        for event in events
        if event.metadata.get("event_type") == "llm.completed"
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
    working = WorkingMemoryManager(store)
    provider = RepeatingToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
        max_tool_iterations=2,
    )

    result = agent.respond("use echo", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    completed = next(
        event for event in events if event.metadata.get("event_type") == "turn.completed"
    )
    finalizing = next(
        event for event in events if event.metadata.get("event_type") == "tool_loop.finalizing"
    )
    assert result.response == "summary after limit"
    assert len(provider.calls) == 4
    assert provider.calls[-1][1] == {}
    assert provider.calls[-1][0][-1]["role"] == "user"
    assert "Do not call tools" in provider.calls[-1][0][-1]["content"]
    assert result.debug["llm_round_count"] == 4
    assert result.debug["tool_iteration_count"] == 2
    assert result.debug["final_finish_reason"] == "stop"
    assert completed.metadata["tool_iteration_count"] == 2
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
    working = WorkingMemoryManager(store)
    provider = RoundLimitedProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
        max_llm_rounds=2,
    )

    result = agent.respond("use echo", session_id="s1")

    events = store.list_events(session_id="s1", limit=40)
    finalizing = next(
        event for event in events if event.metadata.get("event_type") == "tool_loop.finalizing"
    )
    assert result.response == "final answer after round limit"
    assert len(provider.calls) == 3
    assert provider.calls[-1][1] == {}
    assert provider.calls[-1][0][-1]["role"] == "user"
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
    working = WorkingMemoryManager(store)
    provider = FinalizationToolCallingProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
        max_tool_iterations=1,
    )

    with pytest.raises(ToolLoopLimitExceeded, match="finalization returned tool calls"):
        agent.respond("use echo", session_id="s1")

    events = store.list_events(session_id="s1", limit=40)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    finalization_failed = next(
        event
        for event in events
        if event.metadata.get("event_type") == "tool_loop.finalization_failed"
    )
    assert len(provider.calls) == 3
    assert provider.calls[-1][1] == {}
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
    working = WorkingMemoryManager(store)
    provider = MissingIdProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    with pytest.raises(ToolProtocolError, match="missing an id"):
        agent.respond("write", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    assert provider.calls == 1
    assert tool.run_count == 0
    assert not any(event.metadata.get("event_type") == "tool.started" for event in events)
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
    working = WorkingMemoryManager(store)
    provider = EmptyToolCallsProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    with pytest.raises(ToolProtocolError, match="finish_reason=tool_calls"):
        agent.respond("noop", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    llm_started = [
        event for event in events if event.metadata.get("event_type") == "llm.started"
    ]
    assert provider.calls == 1
    assert len(llm_started) == 1
    assert not any(event.metadata.get("event_type") == "tool.started" for event in events)
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
    working = WorkingMemoryManager(store)
    provider = InvalidArgumentsProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    result = agent.respond("lookup", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    assert result.response == "recovered after tool error"
    assert provider.calls == 2
    assert tool.run_count == 0
    assert tool_failed.metadata["tool_name"] == "lookup"
    assert tool_failed.metadata["error_type"] == "ToolExecutionError"
    assert tool_failed.metadata["source"] == "provider"
    assert '"failed":true' in tool_failed.content
    assert '"error_type":"ToolExecutionError"' in tool_failed.content
    assert not any(event.metadata.get("event_type") == "turn.failed" for event in events)


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
    working = WorkingMemoryManager(store)
    provider = UnknownToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=ToolRegistry(),
    )

    result = agent.respond("use missing tool", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    follow_up_messages = provider.calls[1][0]
    assert result.response == "recovered after unknown tool"
    assert len(provider.calls) == 2
    assert tool_failed.metadata["tool_name"] == "missing_provider_tool"
    assert tool_failed.metadata["error_type"] == "ToolExecutionError"
    assert tool_failed.metadata["source"] == "provider"
    assert '"failed":true' in tool_failed.content
    assert '"error":"Unknown tool: missing_provider_tool"' in tool_failed.content
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_call_id"] == "call_missing"
    assert follow_up_messages[-1]["content"] == tool_failed.content
    assert not any(event.metadata.get("event_type") == "turn.failed" for event in events)


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
    working = WorkingMemoryManager(store)
    provider = ExplodingToolProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    result = agent.respond("use exploding tool", session_id="s1")

    events = store.list_events(session_id="s1", limit=30)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    follow_up_messages = provider.calls[1][0]
    assert result.response == "recovered after runtime exception"
    assert len(provider.calls) == 2
    assert tool_failed.metadata["tool_name"] == "explode_provider"
    assert tool_failed.metadata["error_type"] == "RuntimeError"
    assert tool_failed.metadata["source"] == "provider"
    assert '"failed":true' in tool_failed.content
    assert '"error":"provider boom"' in tool_failed.content
    assert '"error_type":"RuntimeError"' in tool_failed.content
    assert follow_up_messages[-1]["role"] == "tool"
    assert follow_up_messages[-1]["tool_call_id"] == "call_explode"
    assert follow_up_messages[-1]["content"] == tool_failed.content
    assert not any(event.metadata.get("event_type") == "turn.failed" for event in events)


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
    working = WorkingMemoryManager(store)
    provider = FlakyProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        max_llm_retries=2,
    )

    result = agent.respond("hello", session_id="s1")

    events = store.list_events(session_id="s1", limit=20)
    completed = next(
        event for event in events if event.metadata.get("event_type") == "llm.completed"
    )
    assert result.response == "recovered"
    assert provider.calls == 2
    assert result.debug["llm_retry_count"] == 1
    assert completed.metadata["retry_count"] == 1


def test_agent_cancellation_writes_failed_event_and_clears_flag(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
    )
    agent.cancel("s1")

    with pytest.raises(AgentCanceledError):
        agent.respond("hello", session_id="s1")

    events = store.list_events(session_id="s1", limit=20)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    assert failed.metadata["status"] == "canceled"
    assert failed.metadata["stage"] == "before_user_event"
    assert not agent.is_canceled("s1")


def test_agent_executes_explicit_tool_calls_with_deterministic_events(tmp_path: Path) -> None:
    class EchoTool:
        name = "echo"
        description = "Echo arguments."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(
                name=self.name,
                content="echoed",
                metadata={"arguments": arguments},
            )

    registry = ToolRegistry()
    registry.register(EchoTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    result = agent.respond(
        "use the tool",
        session_id="s1",
        tool_calls=[ToolCall(name="echo", arguments={"b": 2, "a": 1})],
    )

    events = list(reversed(store.list_events(session_id="s1", limit=20)))
    tool_started = next(
        event for event in events if event.metadata.get("event_type") == "tool.started"
    )
    tool_completed = next(
        event for event in events if event.metadata.get("event_type") == "tool.completed"
    )
    assert tool_started.role == "tool"
    assert tool_completed.role == "tool"
    assert tool_completed.content == (
        '{"content":"echoed","metadata":{"arguments":{"a":1,"b":2}},"name":"echo"}'
    )
    assert tool_completed.metadata["result"]["metadata"]["arguments"] == {"a": 1, "b": 2}
    assert result.debug["tool_call_count"] == 1


def test_tool_cancellation_after_run_records_tool_failed(tmp_path: Path) -> None:
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
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )
    registry.register(CancelAfterRunTool(agent))

    with pytest.raises(AgentCanceledError):
        agent.respond(
            "use the tool",
            session_id="s1",
            tool_calls=[ToolCall(name="cancel_after_run")],
        )

    events = store.list_events(session_id="s1", limit=20)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    turn_failed = next(
        event for event in events if event.metadata.get("event_type") == "turn.failed"
    )
    assert tool_failed.metadata["tool_name"] == "cancel_after_run"
    assert tool_failed.metadata["error_type"] == "AgentCanceledError"
    assert turn_failed.metadata["status"] == "canceled"
    assert turn_failed.metadata["stage"] == "after_tool"


def test_tool_result_serialization_failure_records_tool_failed(tmp_path: Path) -> None:
    class UnserializableTool:
        name = "unserializable"
        description = "Return non-json metadata."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            return ToolResult(name=self.name, content="done", metadata={"bad": object()})

    registry = ToolRegistry()
    registry.register(UnserializableTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    with pytest.raises(ToolExecutionError):
        agent.respond(
            "use the tool",
            session_id="s1",
            tool_calls=[ToolCall(name="unserializable")],
        )

    events = store.list_events(session_id="s1", limit=20)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    turn_failed = next(
        event for event in events if event.metadata.get("event_type") == "turn.failed"
    )
    assert tool_failed.metadata["tool_name"] == "unserializable"
    assert tool_failed.metadata["error_type"] == "TypeError"
    assert turn_failed.metadata["stage"] == "tool"


def test_unknown_tool_records_tool_failed_and_turn_failed_tool_stage(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=ToolRegistry(),
    )

    with pytest.raises(ToolExecutionError):
        agent.respond(
            "use the tool",
            session_id="s1",
            tool_calls=[ToolCall(name="missing_tool")],
        )

    events = store.list_events(session_id="s1", limit=20)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    turn_failed = next(
        event for event in events if event.metadata.get("event_type") == "turn.failed"
    )
    assert tool_failed.metadata["tool_name"] == "missing_tool"
    assert tool_failed.metadata["error_type"] == "ToolExecutionError"
    assert "Unknown tool: missing_tool" in tool_failed.content
    assert turn_failed.metadata["stage"] == "tool"


def test_tool_run_exception_records_tool_failed_and_turn_failed_tool_stage(
    tmp_path: Path,
) -> None:
    class ExplodingTool:
        name = "explode"
        description = "Raise during run."

        def run(self, arguments: dict[str, Any]) -> ToolResult:
            raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register(ExplodingTool())
    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=MockLLMProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        tool_registry=registry,
    )

    with pytest.raises(ToolExecutionError):
        agent.respond(
            "use the tool",
            session_id="s1",
            tool_calls=[ToolCall(name="explode")],
        )

    events = store.list_events(session_id="s1", limit=20)
    tool_failed = next(
        event for event in events if event.metadata.get("event_type") == "tool.failed"
    )
    turn_failed = next(
        event for event in events if event.metadata.get("event_type") == "turn.failed"
    )
    assert tool_failed.metadata["tool_name"] == "explode"
    assert tool_failed.metadata["error_type"] == "RuntimeError"
    assert tool_failed.content == "boom"
    assert turn_failed.metadata["stage"] == "tool"


def test_agent_retry_exhaustion_records_llm_stage_and_retry_count(tmp_path: Path) -> None:
    class AlwaysTimeoutProvider:
        name = "always-timeout"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise httpx.TimeoutException("temporary timeout")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=AlwaysTimeoutProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        max_llm_retries=2,
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_events(session_id="s1", limit=20)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 2


def test_agent_zero_retry_transient_failure_records_llm_stage(tmp_path: Path) -> None:
    class TimeoutProvider:
        name = "timeout"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise httpx.TimeoutException("temporary timeout")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=TimeoutProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
        max_llm_retries=0,
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_events(session_id="s1", limit=20)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 0


def test_agent_non_transient_provider_failure_records_llm_stage(tmp_path: Path) -> None:
    class BadProvider:
        name = "bad"

        def complete(self, messages: list[ChatMessage], **kwargs: Any) -> LLMResponse:
            raise ValueError("bad request")

    store = MemoryStore(tmp_path / "alpha.db")
    store.initialize()
    working = WorkingMemoryManager(store)
    agent = AlphaAgent(
        store=store,
        llm_provider=BadProvider(),
        working_memory=working,
        retriever=MemoryRetriever(store, working),
    )

    with pytest.raises(LLMCallError):
        agent.respond("hello", session_id="s1")

    events = store.list_events(session_id="s1", limit=20)
    failed = next(event for event in events if event.metadata.get("event_type") == "turn.failed")
    assert failed.metadata["stage"] == "llm"
    assert failed.metadata["retry_count"] == 0


def test_cli_basic_commands_with_mock_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "alpha.db"
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(tmp_path / "config.toml"))
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    runner = CliRunner()

    init_result = runner.invoke(app, ["init"])
    ask_result = runner.invoke(app, ["ask", "hello"])

    assert init_result.exit_code == 0
    assert ask_result.exit_code == 0
    assert "Mock response" in ask_result.output
