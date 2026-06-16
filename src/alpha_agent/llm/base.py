"""LLM provider interface."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, Protocol, TypedDict, cast


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
    reasoning_content: NotRequired[str]
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
    reasoning_content: NotRequired[str]
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


class LLMResponseFormat(TypedDict):
    """Provider-neutral response format request."""

    type: Literal["text", "json_object"]


JSON_OBJECT_RESPONSE_FORMAT: LLMResponseFormat = {"type": "json_object"}


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


@dataclass(frozen=True)
class LLMUsage:
    """Provider-neutral token usage for a successful LLM completion."""

    total_tokens: int
    cached_tokens: int
    prompt_cache_miss_tokens: int
    reasoning_tokens: int
    completion_tokens: int

    def __post_init__(self) -> None:
        for name in (
            "total_tokens",
            "cached_tokens",
            "prompt_cache_miss_tokens",
            "reasoning_tokens",
            "completion_tokens",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")


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


def openai_compatible_response_format_payload(
    response_format: LLMResponseFormat,
) -> dict[str, str]:
    """Convert a neutral response format into OpenAI-compatible wire shape."""

    return {"type": response_format["type"]}


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM completion response."""

    content: str
    model: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    reasoning_content: str | None = None
    usage: LLMUsage | None = None


def chat_completion_messages_payload(
    messages: Sequence[ChatMessage],
    *,
    include_reasoning_content: bool = False,
) -> list[dict[str, Any]]:
    """Return provider wire messages with only supported chat-completions fields.

    ``reasoning_content`` is a runtime-level assistant field. Providers opt in to
    receiving it explicitly; compatible APIs that do not understand it get a clean
    OpenAI-style payload.
    """

    return [
        _chat_completion_message_payload(
            cast(Mapping[str, Any], message),
            include_reasoning_content=include_reasoning_content,
        )
        for message in messages
    ]


def _chat_completion_message_payload(
    message: Mapping[str, Any],
    *,
    include_reasoning_content: bool,
) -> dict[str, Any]:
    role = message.get("role")
    if role == "assistant":
        payload = _copy_message_fields(message, ("role", "content", "tool_calls"))
        reasoning_content = message.get("reasoning_content")
        if include_reasoning_content and reasoning_content is not None:
            payload["reasoning_content"] = str(reasoning_content)
        return payload
    if role == "tool":
        return _copy_message_fields(message, ("role", "content", "tool_call_id"))
    return _copy_message_fields(message, ("role", "content"))


def _copy_message_fields(
    message: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    return {field: message[field] for field in fields if field in message}


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
        usage=normalize_llm_usage(payload.get("usage")),
    )


def normalize_llm_usage(raw_usage: Any) -> LLMUsage | None:
    """Normalize supported provider usage payloads into ``LLMUsage``.

    Supports OpenAI-compatible chat-completions fields and the equivalent
    Responses-style aliases used by Codex when all required counts are present.
    """

    if not isinstance(raw_usage, Mapping):
        return None

    total_tokens = _int_value(raw_usage.get("total_tokens"))
    completion_tokens = _first_int(raw_usage, ("completion_tokens", "output_tokens"))
    if total_tokens is None or completion_tokens is None:
        return None

    completion_details = _first_mapping(
        raw_usage,
        ("completion_tokens_details", "output_tokens_details"),
    )
    reasoning_tokens = (
        _int_value(completion_details.get("reasoning_tokens"))
        if completion_details is not None
        else None
    )

    prompt_details = _first_mapping(
        raw_usage,
        ("prompt_tokens_details", "input_tokens_details"),
    )
    cached_tokens = (
        _int_value(prompt_details.get("cached_tokens"))
        if prompt_details is not None
        else None
    )
    if cached_tokens is None:
        cached_tokens = _int_value(raw_usage.get("prompt_cache_hit_tokens"))
    if cached_tokens is None:
        cached_tokens = 0

    prompt_cache_miss_tokens = _int_value(raw_usage.get("prompt_cache_miss_tokens"))
    if prompt_cache_miss_tokens is None:
        prompt_tokens = _first_int(raw_usage, ("prompt_tokens", "input_tokens"))
        if prompt_tokens is None:
            return None
        prompt_cache_miss_tokens = prompt_tokens - cached_tokens

    usage = LLMUsage(
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        reasoning_tokens=reasoning_tokens or 0,
        completion_tokens=completion_tokens,
    )
    if any(
        value < 0
        for value in (
            usage.total_tokens,
            usage.cached_tokens,
            usage.prompt_cache_miss_tokens,
            usage.reasoning_tokens,
            usage.completion_tokens,
        )
    ):
        return None
    return usage


def _int_value(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _first_int(payload: Mapping[str, Any], keys: Sequence[str]) -> int | None:
    for key in keys:
        value = _int_value(payload.get(key))
        if value is not None:
            return value
    return None


def _first_mapping(
    payload: Mapping[str, Any],
    keys: Sequence[str],
) -> Mapping[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    return None


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
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        """Complete a chat-style prompt."""
