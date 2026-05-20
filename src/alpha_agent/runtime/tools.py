"""Explicit synchronous tool execution for agent turns."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_agent.llm.base import LLMToolCall
from alpha_agent.memory.models import Event
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.tools.base import ToolCall, ToolResult
from alpha_agent.tools.registry import ToolRegistry

ToolEventWriter = Callable[[str, str, dict[str, Any]], Event]
CancelCheck = Callable[[str], None]
ToolCallInput = ToolCall | LLMToolCall | Mapping[str, Any]
ToolCallInputs = ToolCallInput | Sequence[ToolCallInput] | None


class ToolExecutionError(RuntimeError):
    """Raised when an explicit tool call cannot complete."""

    def __init__(self, call: ToolCall, message: str):
        super().__init__(message)
        self.call = call


@dataclass(frozen=True)
class ExecutedToolResult:
    """Tool result plus the event that persisted it."""

    call: ToolCall
    result: ToolResult
    event: Event


class ToolExecutor:
    """Small deterministic tool executor backed by the explicit registry."""

    def __init__(self, registry: ToolRegistry | None = None):
        self.registry = registry or ToolRegistry()

    def normalize_calls(
        self,
        raw_calls: ToolCallInputs,
    ) -> list[ToolCall]:
        """Normalize caller/provider tool call shapes into ToolCall objects."""

        if not raw_calls:
            return []
        if isinstance(raw_calls, ToolCall):
            return [raw_calls]
        if isinstance(raw_calls, LLMToolCall):
            return [self._normalize_call(raw_calls)]
        if isinstance(raw_calls, Mapping):
            nested = raw_calls.get("tool_calls")
            if nested is not None:
                return self.normalize_calls(nested)
            return [self._normalize_call(raw_calls)]
        if isinstance(raw_calls, str):
            raise ValueError("tool calls must be mappings, not strings")
        return [self._normalize_call(raw_call) for raw_call in raw_calls]

    def execute(
        self,
        *,
        calls: Sequence[ToolCall],
        write_event: ToolEventWriter,
        check_canceled: CancelCheck,
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        """Execute a finite list of tool calls and persist started/completed/failed events."""

        executed: list[ExecutedToolResult] = []
        for index, call in enumerate(calls):
            check_canceled("before_tool")
            started_event = write_event(
                "tool.started",
                deterministic_json(self._call_payload(call)),
                {
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_index": index,
                    "call": self._call_payload(call),
                },
            )
            try:
                parse_error = call.metadata.get("arguments_parse_error")
                if parse_error:
                    raise ToolExecutionError(
                        call,
                        f"Invalid tool call arguments for {call.name}: {parse_error}",
                    )
                tool = self.registry.get(call.name)
                if tool is None:
                    raise ToolExecutionError(call, f"Unknown tool: {call.name}")
                result = tool.run(dict(call.arguments))
                check_canceled("after_tool")
                result_payload = self._result_payload(result)
                result_content = deterministic_json(result_payload)
                completed_event = write_event(
                    "tool.completed",
                    result_content,
                    {
                        "tool_name": call.name,
                        "tool_call_id": call.id,
                        "tool_index": index,
                        "started_event_id": started_event.id,
                        "result": result_payload,
                    },
                )
            except Exception as exc:
                failed_metadata = {
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_index": index,
                    "started_event_id": started_event.id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if type(exc).__name__ == "AgentCanceledError":
                    write_event("tool.failed", str(exc), failed_metadata)
                    raise
                if not recover_errors:
                    write_event("tool.failed", str(exc), failed_metadata)
                    if isinstance(exc, ToolExecutionError):
                        raise
                    raise ToolExecutionError(call, str(exc)) from exc

                result = self._error_result(call, exc)
                result_payload = self._result_payload(result)
                result_content = deterministic_json(result_payload)
                completed_event = write_event(
                    "tool.failed",
                    result_content,
                    {**failed_metadata, "result": result_payload},
                )
            executed.append(ExecutedToolResult(call=call, result=result, event=completed_event))
        return executed

    def _normalize_call(self, raw_call: ToolCall | LLMToolCall | Mapping[str, Any]) -> ToolCall:
        if isinstance(raw_call, ToolCall):
            return raw_call
        if isinstance(raw_call, LLMToolCall):
            return ToolCall(
                name=raw_call.name,
                arguments=dict(raw_call.arguments),
                id=raw_call.id,
                metadata={
                    **dict(raw_call.metadata),
                    "raw_arguments": raw_call.raw_arguments,
                    "type": raw_call.type,
                },
            )
        arguments = raw_call.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("tool call arguments must be a mapping")
        metadata = raw_call.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        name = raw_call.get("name") or raw_call.get("tool_name")
        if not name:
            raise ValueError("tool call name is required")
        call_id = raw_call.get("id") or raw_call.get("tool_call_id")
        return ToolCall(
            name=str(name),
            arguments=dict(arguments),
            id=str(call_id) if call_id is not None else None,
            metadata=dict(metadata),
        )

    def _call_payload(self, call: ToolCall) -> dict[str, Any]:
        return {
            "arguments": dict(call.arguments),
            "id": call.id,
            "metadata": dict(call.metadata),
            "name": call.name,
        }

    def _result_payload(self, result: ToolResult) -> dict[str, Any]:
        return {
            "content": result.content,
            "metadata": dict(result.metadata),
            "name": result.name,
        }

    def _error_result(self, call: ToolCall, exc: Exception) -> ToolResult:
        message = str(exc)
        return ToolResult(
            name=call.name,
            content=f"Tool execution failed: {message}",
            metadata={
                "failed": True,
                "error": message,
                "error_type": type(exc).__name__,
                "tool_call_id": call.id,
            },
        )
