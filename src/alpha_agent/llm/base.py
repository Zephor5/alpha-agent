"""LLM provider interface."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict


class SystemChatMessage(TypedDict):
    """System message for chat-completions wire payloads."""

    role: Literal["system"]
    content: str


class UserChatMessage(TypedDict):
    """User message for chat-completions wire payloads."""

    role: Literal["user"]
    content: str


class ChatCompletionFunctionCall(TypedDict):
    """OpenAI-compatible function tool call wire payload."""

    name: str
    arguments: str


class ChatCompletionToolCall(TypedDict):
    """OpenAI-compatible assistant tool call wire payload."""

    id: str
    type: Literal["function"]
    function: ChatCompletionFunctionCall


class AssistantChatMessage(TypedDict):
    """Assistant message, including nullable content when tool calls are present."""

    role: Literal["assistant"]
    content: NotRequired[str | None]
    tool_calls: NotRequired[list[ChatCompletionToolCall]]


class ToolChatMessage(TypedDict):
    """Tool result message linked to an assistant tool call."""

    role: Literal["tool"]
    content: str
    tool_call_id: str


ChatMessage = SystemChatMessage | UserChatMessage | AssistantChatMessage | ToolChatMessage


class ChatCompletionAssistantToolMessage(TypedDict):
    """Assistant message with tool calls for follow-up chat-completion requests."""

    role: Literal["assistant"]
    content: str | None
    tool_calls: list[ChatCompletionToolCall]


class ChatCompletionToolResultMessage(TypedDict):
    """Tool result message for follow-up chat-completion requests."""

    role: Literal["tool"]
    content: str
    tool_call_id: str


ChatCompletionToolRoundMessage = (
    ChatCompletionAssistantToolMessage | ChatCompletionToolResultMessage
)


class _LLMResponseMetadata(TypedDict):
    response_id: Any
    finish_reason: Any
    raw_tool_calls: list[Any]
    normalized_tool_calls: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]


class LLMNamedToolChoice(TypedDict):
    """Select a specific function tool by name."""

    type: Literal["function"]
    function: dict[str, str]


LLMToolChoice = Literal["none", "auto", "required"] | LLMNamedToolChoice


@dataclass(frozen=True)
class LLMToolDefinition:
    """Provider-neutral function tool definition."""

    name: str
    description: str
    parameters: dict[str, Any]
    strict: bool | None = None


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral assistant function tool call."""

    id: str | None
    name: str
    arguments: dict[str, Any]
    raw_arguments: str
    type: str = "function"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable mapping for metadata and legacy runtime consumers."""

        return {
            "arguments": dict(self.arguments),
            "id": self.id,
            "metadata": dict(self.metadata),
            "name": self.name,
            "raw_arguments": self.raw_arguments,
            "type": self.type,
        }


LLMToolDefinitionInput = LLMToolDefinition | Mapping[str, Any]


def openai_compatible_tool_payload(tool: LLMToolDefinitionInput) -> dict[str, Any]:
    """Convert a neutral tool definition into OpenAI-compatible wire shape.

    If supplied, ``strict`` is serialized only as provider-neutral wire data. Whether strict
    mode is accepted, requires a beta endpoint, or needs extra configuration is owned by the
    concrete provider setup.
    """

    if isinstance(tool, LLMToolDefinition):
        function: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "parameters": dict(tool.parameters),
        }
        if tool.strict is not None:
            function["strict"] = tool.strict
        return {"type": "function", "function": function}

    if tool.get("type") == "function" and isinstance(tool.get("function"), Mapping):
        return {"type": "function", "function": dict(tool["function"])}

    function = {
        "name": str(tool["name"]),
        "description": str(tool.get("description", "")),
        "parameters": dict(tool.get("parameters", {})),
    }
    if "strict" in tool:
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


def openai_compatible_tool_choice_payload(
    tool_choice: LLMToolChoice,
) -> str | dict[str, Any]:
    """Convert a neutral tool choice into OpenAI-compatible wire shape."""

    if isinstance(tool_choice, str):
        return tool_choice
    return {
        "type": tool_choice["type"],
        "function": dict(tool_choice["function"]),
    }


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM completion response."""

    content: str
    model: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str | None = None


def openai_compatible_response(
    *,
    payload: dict[str, Any],
    fallback_model: str,
    provider: str,
) -> LLMResponse:
    """Normalize a chat-completions response from an OpenAI-compatible provider."""

    choice = payload["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason")
    raw_tool_calls = message.get("tool_calls")
    tool_calls = normalize_openai_compatible_tool_calls(raw_tool_calls)
    normalized_tool_calls = [tool_call.to_dict() for tool_call in tool_calls]
    return LLMResponse(
        content=str(content),
        model=str(payload.get("model", fallback_model)),
        provider=provider,
        metadata={
            "response_id": payload.get("id"),
            "finish_reason": finish_reason,
            "raw_tool_calls": raw_tool_calls if isinstance(raw_tool_calls, list) else [],
            "normalized_tool_calls": normalized_tool_calls,
            "tool_calls": normalized_tool_calls,
        },
        tool_calls=tool_calls,
        finish_reason=str(finish_reason) if finish_reason is not None else None,
    )


def normalize_openai_compatible_tool_calls(raw_tool_calls: Any) -> list[LLMToolCall]:
    """Normalize OpenAI-compatible ``message.tool_calls`` into provider-neutral calls."""

    if not isinstance(raw_tool_calls, list):
        return []

    normalized: list[LLMToolCall] = []
    for raw_tool_call in raw_tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        raw_arguments = function.get("arguments", "")
        if not isinstance(raw_arguments, str):
            raw_arguments = json.dumps(raw_arguments, sort_keys=True)
        arguments, argument_metadata = _parse_tool_call_arguments(raw_arguments)
        call_id = raw_tool_call.get("id")
        call_type = raw_tool_call.get("type", "function")
        metadata = {
            "raw_arguments": raw_arguments,
            "raw_tool_call": dict(raw_tool_call),
            **argument_metadata,
        }
        normalized.append(
            LLMToolCall(
                id=str(call_id) if call_id is not None else None,
                name=name,
                arguments=arguments,
                raw_arguments=raw_arguments,
                type=str(call_type),
                metadata=metadata,
            )
        )
    return normalized


def _parse_tool_call_arguments(raw_arguments: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        return {}, {"arguments_parse_error": str(exc), "raw_arguments": raw_arguments}

    if not isinstance(parsed, dict):
        return (
            {},
            {
                "arguments_parse_error": "tool call arguments JSON must decode to an object",
                "raw_arguments": raw_arguments,
            },
        )
    return dict(parsed), {}


class LLMProvider(Protocol):
    """Synchronous LLM provider interface."""

    name: str

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        """Complete a chat-style prompt."""
