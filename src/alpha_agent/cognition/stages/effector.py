"""Effector stage for reactive ticks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Decision,
    EventId,
    NLStatement,
    Reference,
)
from alpha_agent.cognition.render.base import RenderBudget, Renderer, RenderResult
from alpha_agent.cognition.render.text_chat import TextChatRenderer
from alpha_agent.cognition.render.view import CognitionView
from alpha_agent.cognition.stages.types import Emitted, Outcome
from alpha_agent.llm.base import (
    ChatCompletionToolCall,
    ChatMessage,
    LLMProvider,
    LLMResponse,
)
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.runtime.tools import ToolExecutor
from alpha_agent.state.models import RuntimeTrace
from alpha_agent.tools.base import ToolCall, ToolResult
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

CompletionRunner = Callable[[Decision, CognitionView, RenderResult], Outcome]


class Effector:
    """Execute a decision against the outside world through the LLM/tool loop."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry,
        renderer: Renderer | None = None,
        render_budget: RenderBudget | None = None,
        completion_runner: CompletionRunner | None = None,
        max_tool_iterations: int = 1,
    ):
        self.llm_provider = llm_provider
        self.tool_registry = tool_registry
        self.tool_executor = ToolExecutor(tool_registry)
        self.renderer = renderer or TextChatRenderer()
        self.render_budget = render_budget or RenderBudget()
        self.completion_runner = completion_runner or self._complete_once
        self.max_tool_iterations = max(0, max_tool_iterations)

    def execute(
        self,
        decision: Decision,
        view: CognitionView,
        *,
        emitter: EventEmitter,
        tick_id: str,
        causal_parent: EventId,
    ) -> Emitted[Outcome]:
        rendered = self.renderer.render(view, self.render_budget)
        outcome = self.completion_runner(decision, view, rendered)
        event = emitter.emit(
            CognitiveEventKind.ACTED,
            situation=view.window.situation_at,
            inputs=[Reference("decision", str(decision.id))],
            rationale=NLStatement("Executed decision through effector."),
            causal_parents=[causal_parent],
            payload={
                "tick_id": tick_id,
                "outcome_text_len": len(outcome.text or ""),
                "tool_call_count": len(outcome.tool_calls),
                "tool_result_count": len(outcome.tool_results),
            },
        )
        return Emitted(outcome, event)

    def _complete_once(
        self,
        decision: Decision,
        view: CognitionView,
        rendered: RenderResult,
    ) -> Outcome:
        messages = list(rendered.payload)
        tools = self.tool_registry.to_llm_tool_definitions()
        llm_round_count = 1
        response = self.llm_provider.complete(
            messages,
            tools=tools or None,
            tool_choice="auto" if tools else None,
        )
        tool_calls = self.tool_executor.normalize_calls(response.tool_calls)
        tool_results: list[ToolResult] = []
        if tool_calls and self.max_tool_iterations > 0:
            executed = self.tool_executor.execute(
                calls=tool_calls,
                write_trace=_in_memory_trace,
                check_canceled=lambda _stage: None,
                recover_errors=True,
            )
            tool_results = [item.result for item in executed]
            messages.append(_assistant_tool_call_message(response, tool_calls))
            messages.extend(
                _tool_result_message(call, result)
                for call, result in zip(tool_calls, tool_results, strict=True)
            )
            response = self.llm_provider.complete(
                messages,
                tools=tools or None,
                tool_choice="none" if tools else None,
            )
            llm_round_count += 1
        return Outcome(
            text=response.content,
            tool_calls=tool_calls,
            tool_results=tool_results,
            raw_llm_response=response,
            debug={
                "provider": response.provider,
                "final_provider": response.provider,
                "renderer": self.renderer.name,
                "render_used_tokens": rendered.used_tokens,
                "render_dropped_sections": list(rendered.dropped_sections),
                "llm_round_count": llm_round_count,
                "llm_retry_count": 0,
                "tool_iteration_count": 1 if tool_results else 0,
                "tool_call_count": len(tool_results),
                "provider_tool_call_count": len(tool_calls),
                "final_finish_reason": response.finish_reason,
            },
        )


def _assistant_tool_call_message(
    response: LLMResponse,
    tool_calls: Sequence[ToolCall],
) -> ChatMessage:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": [_wire_tool_call(call) for call in tool_calls],
    }
    if response.reasoning_content is not None:
        message["reasoning_content"] = response.reasoning_content
    return cast(ChatMessage, message)


def _tool_result_message(call: ToolCall, result: ToolResult) -> ChatMessage:
    return {
        "role": "tool",
        "tool_call_id": _required_tool_call_id(call),
        "content": deterministic_json(
            {
                "content": result.content,
                "metadata": dict(result.metadata),
                "name": result.name,
            }
        ),
    }


def _wire_tool_call(call: ToolCall) -> ChatCompletionToolCall:
    raw_arguments = call.metadata.get("raw_arguments")
    return {
        "id": _required_tool_call_id(call),
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": raw_arguments
            if isinstance(raw_arguments, str)
            else deterministic_json(call.arguments),
        },
    }


def _required_tool_call_id(call: ToolCall) -> str:
    if not call.id:
        raise ValueError(f"Provider tool call for {call.name} is missing an id")
    return call.id


def _in_memory_trace(event_type: str, content: str, metadata: dict[str, Any]) -> RuntimeTrace:
    return RuntimeTrace(
        id=new_id("trace"),
        session_id="reactive-effector",
        event_type=event_type,
        content=content,
        timestamp=utc_now_iso(),
        metadata=metadata,
    )


def outcome_tool_metadata(
    *,
    tool_calls: Sequence[ToolCall],
    tool_results: Sequence[ToolResult],
    llm_round_count: int,
    llm_retry_count: int,
    tool_iteration_count: int,
    provider: str | None,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "final_provider": provider,
        "final_finish_reason": finish_reason,
        "llm_round_count": llm_round_count,
        "llm_retry_count": llm_retry_count,
        "tool_iteration_count": tool_iteration_count,
        "tool_call_count": len(tool_results),
        "provider_tool_call_count": len(tool_calls),
    }
