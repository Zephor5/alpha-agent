from __future__ import annotations

import json
from collections.abc import Sequence

from alpha_agent.llm.base import ChatMessage, LLMResponse, LLMToolChoice, LLMToolDefinitionInput
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import Tool, ToolResult
from alpha_agent.tools.registry import ToolRegistry


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def test_agent_responds_and_persists_conversation_messages(tmp_path) -> None:
    store = _store(tmp_path)
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())

    result = agent.respond("hello", session_id="s1")

    assert result.session_id == "s1"
    assert result.response == "Mock response: I heard you say: hello."
    assert result.debug["note"] == "reactive cognition tick enabled; projections are Phase 02 stubs"
    messages = store.list_conversation_messages("s1")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].raw_content == "hello"
    assert messages[1].raw_content == result.response
    assert [trace.event_type for trace in store.list_runtime_traces("s1")] == [
        "llm.started",
        "llm.completed",
    ]


def test_agent_uses_context_window_foreground_for_llm_input(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _RecordingProvider("context response")
    agent = AlphaAgent(
        store=store,
        llm_provider=provider,
        context_recent_tail_messages=1,
    )
    agent.respond("first", session_id="s1")

    result = agent.respond("current", session_id="s1")

    assert result.response == "context response"
    rendered_contents = "\n".join(str(message.get("content", "")) for message in provider.calls[-1])
    assert "Foreground:" in rendered_contents
    assert "first" in rendered_contents
    assert provider.calls[-1][-1] == {"role": "user", "content": "current"}


def test_agent_executes_provider_tool_calls_and_stores_tool_round(tmp_path) -> None:
    store = _store(tmp_path)
    registry = ToolRegistry()
    registry.register(_EchoTool())
    provider = _ToolCallingProvider()
    agent = AlphaAgent(store=store, llm_provider=provider, tool_registry=registry)

    result = agent.respond("use tool", session_id="s1")

    assert result.response == "final answer"
    assert result.debug["tool_call_count"] == 1
    assert [message.role for message in store.list_conversation_messages("s1")] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
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
        for message in store.list_conversation_messages("s1")
        if message.role == "assistant"
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
    assert "secret response payload" in response_log_json


def test_agent_cancel_before_turn_raises_and_clears_flag(tmp_path) -> None:
    import pytest

    store = _store(tmp_path)
    agent = AlphaAgent(store=store, llm_provider=MockLLMProvider())
    agent.cancel("s1")

    with pytest.raises(Exception, match="canceled"):
        agent.respond("hello", session_id="s1")

    assert agent.is_canceled("s1") is False
    assert store.list_runtime_traces("s1", event_type="turn.failed")


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
                    {
                        "id": "call_1",
                        "name": "echo",
                        "arguments": {"text": "hello"},
                        "raw_arguments": '{"text":"hello"}',
                    }
                ],
            )
        return LLMResponse(content="final answer", model="test", provider=self.name)


class _EchoTool(Tool):
    name = "echo"
    description = "Echo input."

    def run(self, arguments):
        return ToolResult(name=self.name, content=str(arguments["text"]), metadata={})
