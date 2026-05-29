from __future__ import annotations

from collections.abc import Sequence
from typing import cast

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.event_log.memory import InMemoryEventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    Stimulus,
    StimulusKind,
    ThreadId,
)
from alpha_agent.llm.base import (
    AssistantChatMessage,
    ChatMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import Tool, ToolExecutionContext, ToolResult
from alpha_agent.tools.registry import ToolRegistry


def test_reactive_tick_tool_call_path_emits_acted_and_feedback(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    agent = AlphaAgent(store=store, llm_provider=_ToolCallingProvider(), tool_registry=registry)

    result = agent.respond("use tool", session_id="s1")

    assert result.response == "final answer"
    messages = store.list_session_messages("s1")
    assert [message.kind for message in messages] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    assert messages[1].reasoning_content == "Need to call echo."
    assert messages[3].reasoning_content is None
    events = list(SQLiteEventLog(store).iter())
    perceived = [event for event in events if event.kind == CognitiveEventKind.PERCEIVED][0]
    acted = [event for event in events if event.kind == CognitiveEventKind.ACTED][0]
    feedback = [event for event in events if event.kind == CognitiveEventKind.RECEIVED_FEEDBACK][0]
    decided = [event for event in events if event.kind == CognitiveEventKind.DECIDED][0]
    source_recorded = [
        event for event in events if event.kind.value == "turn_sources_recorded"
    ][0]
    assert decided.payload["action"] == "use_tool"
    assert acted.payload["tool_call_count"] == 1
    assert acted.payload["tool_call_ids"] == ["call_1"]
    assert acted.payload["provider_tool_message_ids"] == [messages[1].id, messages[2].id]
    assert acted.payload["provider_tool_trace_ids"] == [messages[2].tool_result_id]
    assert feedback.payload["matched_expected"] is True
    assert perceived.payload["session_id"] == "s1"
    assert perceived.payload["user_message_id"] == messages[0].id
    assert {"kind": "session", "id": "s1"} in perceived.payload["source_refs"]
    assert {"kind": "session_message", "id": messages[0].id} in perceived.payload[
        "source_refs"
    ]
    assert source_recorded.payload["tick_id"] == acted.payload["tick_id"]
    assert source_recorded.payload["session_id"] == "s1"
    assert source_recorded.payload["user_message_id"] == messages[0].id
    assert source_recorded.payload["assistant_message_id"] == messages[3].id
    assert source_recorded.payload["provider_tool_message_ids"] == [
        messages[1].id,
        messages[2].id,
    ]
    assert source_recorded.payload["provider_tool_trace_ids"] == [messages[2].tool_result_id]


def test_default_effector_executes_tool_and_final_llm_round() -> None:
    log = InMemoryEventLog()
    registry = ToolRegistry()
    registry.register(_EchoTool())
    provider = _ToolCallingProvider()
    thread_id = ThreadId.from_session("s1")
    controller = CognitiveController(
        event_log=log,
        projections=default_projection_registry(log),
        llm=provider,
        tools=registry,
    )

    result = controller.reactive_tick(
        stimulus=Stimulus(
            kind=StimulusKind.USER_MESSAGE,
            source=None,
            payload="use tool",
            thread_id=thread_id,
            received_at=Instant("2026-01-01T00:00:00+00:00"),
        ),
        thread_id=thread_id,
    )

    assert result.response_text == "final answer"
    assert result.outcome.tool_results[0].output == "hello"
    assert result.debug["llm_round_count"] == 2
    assert [message["role"] for message in provider.messages[1][-2:]] == ["assistant", "tool"]
    assistant_message = cast(AssistantChatMessage, provider.messages[1][-2])
    assert assistant_message["reasoning_content"] == "Need to call echo."
    acted = [event for event in log.iter(kinds=[CognitiveEventKind.ACTED])][0]
    assert acted.payload["tool_call_count"] == 1
    assert acted.payload["tool_result_count"] == 1


class _ToolCallingProvider:
    name = "tool-provider"

    def __init__(self):
        self.calls = 0
        self.messages: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        self.calls += 1
        self.messages.append(messages)
        if self.calls == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                reasoning_content="Need to call echo.",
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


class _EchoTool(Tool):
    name = "echo"
    description = "Echo input."

    def run(self, arguments, context: ToolExecutionContext):
        del context
        return ToolResult(name=self.name, output=str(arguments["text"]), metadata={})
