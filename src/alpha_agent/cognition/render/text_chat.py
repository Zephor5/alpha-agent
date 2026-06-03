"""Chat-completions helper functions for replayable runtime messages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from alpha_agent.llm.base import ChatCompletionToolCall, ChatMessage, LLMToolDefinitionInput
from alpha_agent.runtime.context_budget import estimate_context_budget
from alpha_agent.state.models import SessionMessage

SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"
COUNTERPART_PROFILE_LABEL = "Counterpart profile:"


def wrap_system_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content.strip()}\n{SYSTEM_REMINDER_CLOSE}"


def render_counterpart_profile(content: str) -> str:
    return f"{COUNTERPART_PROFILE_LABEL} {content}"


def estimate_chat_tokens(
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
) -> int:
    estimate = estimate_context_budget(messages, tools=tools, max_context_tokens=0)
    return estimate.message_tokens + estimate.tool_schema_tokens


def source_message_to_chat(message: SessionMessage) -> ChatMessage:
    """Convert a durable source message into a replayable chat message."""

    content = message.model_content if message.model_content is not None else message.raw_content
    if message.kind == "compressed_message":
        raise ValueError("compressed_message must be projected by SessionContextAssembler")
    if message.llm_role == "user":
        return {"role": "user", "content": content}
    if message.llm_role == "assistant":
        assistant_content = (content or None) if message.tool_calls else content
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if message.reasoning_content is not None:
            assistant_message["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                _source_tool_call(tool_call) for tool_call in message.tool_calls
            ]
        return cast(ChatMessage, assistant_message)
    if message.llm_role != "tool":
        raise ValueError(f"session message {message.id!r} is missing llm_role")
    if not message.tool_call_id:
        raise ValueError(f"tool session message {message.id!r} is missing tool_call_id")
    return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}


def _source_tool_call(payload: Mapping[str, Any]) -> ChatCompletionToolCall:
    function = payload.get("function")
    if not isinstance(function, Mapping):
        function = {}
    return {
        "id": str(payload.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": str(function.get("arguments") or "{}"),
        },
    }
