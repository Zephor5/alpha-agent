"""Shared LLM request/response tracing helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Protocol

from alpha_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

DEFAULT_LLM_TRACE_LOG_FILENAME = "llm.jsonl"


class LLMTraceConfig(Protocol):
    """Runtime config fields needed to construct an LLM trace logger."""

    @property
    def llm_debug_logging(self) -> bool: ...

    @property
    def log_dir(self) -> Path: ...


@dataclass(frozen=True, slots=True, init=False)
class LLMTraceLogger:
    """Append raw LLM debug request/response payloads to a JSONL log."""

    trace_log_path: Path | None = None

    def __init__(
        self,
        *,
        trace_log_path: str | Path | None = None,
    ) -> None:
        object.__setattr__(
            self,
            "trace_log_path",
            Path(trace_log_path).expanduser() if trace_log_path is not None else None,
        )

    @classmethod
    def from_config(cls, config: LLMTraceConfig) -> LLMTraceLogger:
        """Build the logger from the runtime-level trace flag and log directory."""

        if not config.llm_debug_logging:
            return cls(trace_log_path=None)
        return cls(trace_log_path=config.log_dir / DEFAULT_LLM_TRACE_LOG_FILENAME)

    @property
    def enabled(self) -> bool:
        return self.trace_log_path is not None

    def append(self, *, event: str, metadata: Mapping[str, Any]) -> None:
        """Append one JSONL debug trace entry when logging is enabled."""

        if not self.enabled or self.trace_log_path is None:
            return
        self.trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": utc_now_iso(),
            "level": "debug",
            "event": event,
            "metadata": json_safe(metadata),
        }
        with self.trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def append_request(
        self,
        *,
        metadata: Mapping[str, Any],
        messages: Sequence[ChatMessage],
        tools: Sequence[LLMToolDefinitionInput] | None,
        tool_choice: LLMToolChoice | None,
        response_format: LLMResponseFormat | None = None,
    ) -> None:
        """Append one raw LLM request trace when logging is enabled."""

        if not self.enabled:
            return
        self.append(
            event="llm.request",
            metadata={
                **dict(metadata),
                "request": llm_request_log(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                ),
            },
        )

    def append_response(
        self,
        *,
        metadata: Mapping[str, Any],
        response: LLMResponse,
    ) -> None:
        """Append one raw LLM response trace when logging is enabled."""

        if not self.enabled:
            return
        self.append(
            event="llm.response",
            metadata={**dict(metadata), "response": llm_response_log(response)},
        )


def traced_llm_complete(
    provider: LLMProvider,
    messages: Sequence[ChatMessage],
    *,
    trace_logger: LLMTraceLogger | None = None,
    trace_metadata: Mapping[str, Any] | None = None,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
    tool_choice: LLMToolChoice | None = None,
    response_format: LLMResponseFormat | None = None,
) -> LLMResponse:
    """Call an LLM provider and write matching raw debug traces when configured."""

    call_messages = list(messages)
    raw_trace_metadata = dict(trace_metadata or {})
    base_metadata = {
        **raw_trace_metadata,
        "llm_call_id": raw_trace_metadata.get("llm_call_id") or new_id("llm"),
        "provider": getattr(provider, "name", ""),
    }
    if trace_logger is not None:
        trace_logger.append_request(
            metadata=base_metadata,
            messages=call_messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    kwargs: dict[str, Any] = {}
    if tools is not None:
        kwargs["tools"] = tools
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    if response_format is not None:
        kwargs["response_format"] = response_format
    response = provider.complete(call_messages, **kwargs)

    if trace_logger is not None:
        trace_logger.append_response(metadata=base_metadata, response=response)
    return response


def llm_request_log(
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[LLMToolDefinitionInput] | None,
    tool_choice: LLMToolChoice | None,
    response_format: LLMResponseFormat | None = None,
) -> dict[str, Any]:
    payload = {
        "messages": json_safe(list(messages)),
        "tools": json_safe(list(tools)) if tools is not None else None,
        "tool_choice": json_safe(tool_choice),
    }
    if response_format is not None:
        payload["response_format"] = json_safe(response_format)
    return payload


def llm_request_summary(
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[LLMToolDefinitionInput] | None,
    tool_choice: LLMToolChoice | None,
    response_format: LLMResponseFormat | None = None,
) -> dict[str, Any]:
    summary = {
        "message_count": len(messages),
        "roles": [str(message.get("role", "")) for message in messages],
        "tool_count": len(tools) if tools is not None else 0,
        "tool_names": [_llm_tool_name(tool) for tool in tools or []],
        "tool_choice": json_safe(tool_choice),
    }
    if response_format is not None:
        summary["response_format"] = json_safe(response_format)
    return summary


def llm_response_log(response: LLMResponse) -> dict[str, Any]:
    response_payload = response.metadata.get("response_payload")
    if isinstance(response_payload, dict):
        return json_safe(response_payload)

    return {
        "content": response.content,
        "finish_reason": response.finish_reason,
        "model": response.model,
        "provider": response.provider,
        "tool_calls": [tool_call.to_dict() for tool_call in response.tool_calls],
    }


def llm_metadata_summary(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return json_safe(
        {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "request_payload",
                "response_payload",
                "raw_tool_calls",
                "normalized_tool_calls",
                "tool_calls",
            }
        }
    )


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [json_safe(item) for item in value]
    return str(value)


def _llm_tool_name(tool: LLMToolDefinitionInput) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") is not None:
            return str(function["name"])
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))
