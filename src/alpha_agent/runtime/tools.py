"""Explicit synchronous tool execution for agent turns."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_agent.llm.base import LLMToolCall
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.state.models import RuntimeTrace
from alpha_agent.tools.base import (
    ToolAvailability,
    ToolCall,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
    tool_availability,
    tool_output_to_model_content,
    tool_spec,
)
from alpha_agent.tools.registry import ToolRegistry

ToolTraceWriter = Callable[[str, str, dict[str, Any]], RuntimeTrace]
CancelCheck = Callable[[str], None]
ToolCallInput = ToolCall | LLMToolCall | Mapping[str, Any]
ToolCallInputs = ToolCallInput | Sequence[ToolCallInput] | None
TRUNCATION_MARKER_TEMPLATE = "[tool output truncated: {omitted_chars} chars omitted]"


class ToolExecutionError(RuntimeError):
    """Raised when an explicit tool call cannot complete."""

    def __init__(self, call: ToolCall, message: str):
        super().__init__(message)
        self.call = call


class ToolUnavailableError(ToolExecutionError):
    """Raised when a known tool is registered but cannot currently run."""

    def __init__(self, call: ToolCall, availability: ToolAvailability):
        reason = availability.reason or "tool is unavailable"
        super().__init__(call, f"Tool unavailable: {call.name}: {reason}")
        self.availability = availability


@dataclass(frozen=True)
class ExecutedToolResult:
    """Tool result plus the trace that persisted its diagnostic record."""

    call: ToolCall
    result: ToolResult
    trace: RuntimeTrace


@dataclass(frozen=True)
class _ToolResultView:
    """Persisted/model-facing view of a tool result."""

    content: str
    result_payload: dict[str, Any]
    trace_metadata: dict[str, Any]


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
        session_id: str = "unknown",
        output_dir: str | Path = Path(".alpha-agent/tool-results"),
        extensions: Mapping[str, Any] | None = None,
        write_trace: ToolTraceWriter,
        check_canceled: CancelCheck,
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        """Execute a finite list of tool calls and persist diagnostic traces."""

        executed: list[ExecutedToolResult] = []
        for index, call in enumerate(calls):
            check_canceled("before_tool")
            parse_error = call.metadata.get("arguments_parse_error")
            tool = self.registry.get(call.name) if not parse_error else None
            spec = tool_spec(tool) if tool is not None else None
            availability = (
                tool_availability(tool)
                if tool is not None and not parse_error
                else None
            )
            trace_payload = self._call_payload(
                call,
                trace_arguments=self._trace_arguments(tool, call),
            )
            started_trace = write_trace(
                "tool.started",
                deterministic_json(trace_payload),
                {
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_index": index,
                    "call": trace_payload,
                    **self._trace_governance_metadata(spec, availability),
                },
            )
            try:
                if parse_error:
                    raise ToolExecutionError(
                        call,
                        f"Invalid tool call arguments for {call.name}: {parse_error}",
                    )
                if tool is None:
                    raise ToolExecutionError(call, f"Unknown tool: {call.name}")
                if availability is not None and not availability.available:
                    raise ToolUnavailableError(call, availability)
                context = ToolExecutionContext(
                    session_id=session_id,
                    tool_call_id=call.id,
                    output_dir=Path(output_dir).expanduser(),
                    check_canceled=check_canceled,
                    extensions=dict(extensions or {}),
                )
                result = tool.run(dict(call.arguments), context)
                if not self._is_canceled_result(result):
                    check_canceled("after_tool")
                result_view = self._result_view(result, spec)
                completed_trace = write_trace(
                    "tool.completed",
                    result_view.content,
                    {
                        "tool_name": call.name,
                        "tool_call_id": call.id,
                        "tool_index": index,
                        "started_trace_id": started_trace.id,
                        "result": result_view.result_payload,
                        **self._trace_governance_metadata(spec, availability),
                        **result_view.trace_metadata,
                    },
                )
            except Exception as exc:
                failed_metadata = {
                    "tool_name": call.name,
                    "tool_call_id": call.id,
                    "tool_index": index,
                    "started_trace_id": started_trace.id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **self._trace_governance_metadata(spec, availability),
                }
                if type(exc).__name__ == "AgentCanceledError":
                    write_trace("tool.failed", str(exc), failed_metadata)
                    raise
                if not recover_errors:
                    write_trace("tool.failed", str(exc), failed_metadata)
                    if isinstance(exc, ToolExecutionError):
                        raise
                    raise ToolExecutionError(call, str(exc)) from exc

                result = self._error_result(call, exc)
                result_view = self._result_view(result, spec)
                completed_trace = write_trace(
                    "tool.failed",
                    result_view.content,
                    {
                        **failed_metadata,
                        "result": result_view.result_payload,
                        **result_view.trace_metadata,
                    },
                )
            executed.append(ExecutedToolResult(call=call, result=result, trace=completed_trace))
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

    def _call_payload(
        self,
        call: ToolCall,
        *,
        trace_arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "arguments": dict(trace_arguments if trace_arguments is not None else call.arguments),
            "id": call.id,
            "metadata": self._trace_metadata(call),
            "name": call.name,
        }

    def _trace_arguments(self, tool: Any, call: ToolCall) -> Mapping[str, Any]:
        trace_arguments = getattr(tool, "trace_arguments", None)
        if callable(trace_arguments):
            return trace_arguments(dict(call.arguments))
        return dict(call.arguments)

    def _trace_metadata(self, call: ToolCall) -> dict[str, Any]:
        metadata = dict(call.metadata)
        metadata.pop("raw_arguments", None)
        return metadata

    def _result_payload(self, result: ToolResult) -> dict[str, Any]:
        return {
            "metadata": dict(result.metadata),
            "name": result.name,
            "output": result.output,
        }

    def _result_view(
        self,
        result: ToolResult,
        spec: ToolSpec | None,
    ) -> _ToolResultView:
        content = tool_output_to_model_content(result.output)
        limit = spec.max_result_size_chars if spec is not None else None
        if limit is None:
            return _ToolResultView(
                content=content,
                result_payload=self._result_payload(result),
                trace_metadata={},
            )

        original_chars = len(content)
        truncated = original_chars > limit
        omitted_chars = max(0, original_chars - limit)
        output_limit: dict[str, Any] = {
            "limit_chars": limit,
            "omitted_chars": omitted_chars,
            "original_chars": original_chars,
            "truncated": truncated,
        }
        if not truncated:
            return _ToolResultView(
                content=content,
                result_payload=self._result_payload(result),
                trace_metadata={"tool_output_limit": output_limit},
            )

        if isinstance(result.output, Mapping):
            bounded_output = _bounded_structured_mapping_output(
                result.output,
                limit=limit,
                original_chars=original_chars,
            )
            bounded_content = tool_output_to_model_content(bounded_output)
            output_limit["bounded_chars"] = len(bounded_content)
            output_limit["limit_enforced"] = len(bounded_content) <= limit
            output_limit["omitted_chars"] = max(0, original_chars - len(bounded_content))
            output_limit["structured"] = True
            if not output_limit["limit_enforced"]:
                output_limit["structured_limit_conflict"] = True
            return _ToolResultView(
                content=bounded_content,
                result_payload={
                    "metadata": dict(result.metadata),
                    "name": result.name,
                    "output": bounded_output,
                },
                trace_metadata={"tool_output_limit": output_limit},
            )

        preview, omitted_chars = _bounded_truncated_content(content, limit)
        output_limit["omitted_chars"] = omitted_chars
        return _ToolResultView(
            content=preview,
            result_payload={
                "metadata": dict(result.metadata),
                "name": result.name,
                "output": preview,
            },
            trace_metadata={"tool_output_limit": output_limit},
        )

    def _trace_governance_metadata(
        self,
        spec: ToolSpec | None,
        availability: ToolAvailability | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if spec is not None:
            payload["tool_spec"] = spec.to_dict()
        if availability is not None:
            payload["tool_availability"] = availability.to_dict()
        return payload

    def _is_canceled_result(self, result: ToolResult) -> bool:
        if result.metadata.get("status") == "canceled":
            return True
        if isinstance(result.output, Mapping):
            return result.output.get("status") == "canceled"
        return False

    def _error_result(
        self,
        call: ToolCall,
        exc: Exception,
    ) -> ToolResult:
        message = str(exc)
        if isinstance(exc, ToolUnavailableError):
            availability_payload = exc.availability.to_dict()
            return ToolResult(
                name=call.name,
                output={
                    "error": {
                        "code": "tool_unavailable",
                        "message": message,
                        "details": {
                            "availability": availability_payload,
                        },
                    }
                },
                metadata={
                    "failed": True,
                    "error": message,
                    "error_type": type(exc).__name__,
                    "tool_call_id": call.id,
                },
            )
        return ToolResult(
            name=call.name,
            output=f"Tool execution failed: {message}",
            metadata={
                "failed": True,
                "error": message,
                "error_type": type(exc).__name__,
                "tool_call_id": call.id,
            },
        )


def _bounded_truncated_content(content: str, limit: int) -> tuple[str, int]:
    """Return bounded truncated content with a marker when the limit can fit it."""

    original_chars = len(content)
    for prefix_chars in range(min(original_chars, limit), -1, -1):
        omitted_chars = original_chars - prefix_chars
        marker = TRUNCATION_MARKER_TEMPLATE.format(omitted_chars=omitted_chars)
        separator = "\n" if prefix_chars else ""
        if prefix_chars + len(separator) + len(marker) <= limit:
            return f"{content[:prefix_chars]}{separator}{marker}", omitted_chars

    return content[:limit], original_chars - limit


def _bounded_structured_mapping_output(
    output: Mapping[str, Any],
    *,
    limit: int,
    original_chars: int,
) -> dict[str, Any]:
    """Return a JSON-safe bounded mapping without cutting serialized JSON."""

    raw_error = output.get("error")
    if isinstance(raw_error, Mapping):
        return _bounded_error_output(
            raw_error,
            limit=limit,
            original_chars=original_chars,
        )

    truncation = _structured_truncation_payload(
        limit=limit,
        original_chars=original_chars,
    )
    return _first_bounded_mapping(
        [
            {
                "output": "[structured tool output omitted: result exceeded size limit]",
                "truncation": truncation,
            },
            {"output": "[structured tool output omitted: result exceeded size limit]"},
        ],
        limit=limit,
    )


def _bounded_error_output(
    error: Mapping[str, Any],
    *,
    limit: int,
    original_chars: int,
) -> dict[str, Any]:
    code = str(error.get("code") or "tool_error")
    message = str(error.get("message") or "")
    bounded_error: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    truncation = _structured_truncation_payload(
        limit=limit,
        original_chars=original_chars,
    )
    details = _bounded_error_details(error.get("details"))
    candidates: list[dict[str, Any]] = []
    if details:
        candidates.extend(
            [
                {
                    "error": {**bounded_error, "details": details},
                    "truncation": truncation,
                },
                {"error": {**bounded_error, "details": details}},
            ]
        )
    candidates.extend(
        [
            {"error": bounded_error, "truncation": truncation},
            {"error": bounded_error},
            {"error": {"code": code}, "truncation": truncation},
            {"error": {"code": code}},
        ]
    )
    return _first_bounded_mapping(candidates, limit=limit)


def _bounded_error_details(raw_details: Any) -> dict[str, Any]:
    if not isinstance(raw_details, Mapping):
        return {}

    details: dict[str, Any] = {}
    for key, value in raw_details.items():
        if key == "availability" and isinstance(value, Mapping):
            details[str(key)] = _bounded_availability(value)
        else:
            details[str(key)] = "[omitted: result exceeded size limit]"
    return details


def _bounded_availability(raw_availability: Mapping[str, Any]) -> dict[str, Any]:
    availability: dict[str, Any] = {}
    if "available" in raw_availability:
        availability["available"] = bool(raw_availability["available"])
    reason = raw_availability.get("reason")
    if reason is not None:
        availability["reason"] = str(reason)
    if raw_availability.get("details"):
        availability["details"] = {"truncated": True}
    return availability


def _structured_truncation_payload(
    *,
    limit: int,
    original_chars: int,
) -> dict[str, int | bool]:
    return {
        "limit_chars": limit,
        "original_chars": original_chars,
        "truncated": True,
    }


def _first_bounded_mapping(
    candidates: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    for candidate in candidates:
        if len(tool_output_to_model_content(candidate)) <= limit:
            return candidate
    return candidates[-1]
