"""Deprecated Phase 00 prompt builder kept until Phase 09 renderer split."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from alpha_agent.llm.base import ChatCompletionToolCall, ChatMessage, LLMToolDefinitionInput
from alpha_agent.state.models import ConversationMessage

SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"


def wrap_system_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content.strip()}\n{SYSTEM_REMINDER_CLOSE}"


class PromptBuilder:
    """Build minimal chat prompts while Effector owns Phase 02 rendering."""

    system_prompt = """Identity: Alpha Agent.

Behavior rules:
- Be concise but useful.
- Use the current conversation transcript as operational context.
- Reactive cognition is active; long-term belief/context projections are Phase 02 stubs.
- Prefer asking clarifying questions only when necessary.
- Keep the runtime understandable and avoid hidden agent behavior."""

    def build(
        self,
        user_message: str,
        conversation_messages: Sequence[ConversationMessage] = (),
    ) -> list[ChatMessage]:
        """Build messages compatible with chat completions APIs."""

        messages: list[ChatMessage] = [{"role": "system", "content": self.system_prompt}]
        messages.extend(
            self.conversation_message_to_chat(message)
            for message in conversation_messages
        )
        messages.append({"role": "user", "content": user_message})
        return messages

    def rough_token_estimate(self, messages: list[ChatMessage]) -> int:
        """Estimate prompt tokens with a simple character-based approximation."""

        return self.estimate_prompt_tokens(messages)

    def estimate_prompt_tokens(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
    ) -> int:
        """Estimate prompt tokens including message content, tool calls, and schemas."""

        character_count = sum(_message_character_count(message) for message in messages)
        if tools is not None:
            character_count += sum(len(_stable_json(_tool_payload(tool))) for tool in tools)
        return character_count // 4

    def conversation_message_to_chat(self, message: ConversationMessage) -> ChatMessage:
        """Convert a durable transcript message into a replayable chat message."""

        content = (
            message.model_content
            if message.model_content is not None
            else message.raw_content
        )
        if message.role == "user":
            return {"role": "user", "content": content}
        if message.role == "assistant":
            if message.tool_calls:
                return {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        self._conversation_tool_call(tool_call)
                        for tool_call in message.tool_calls
                    ],
                }
            return {"role": "assistant", "content": content}
        if not message.tool_call_id:
            raise ValueError(
                f"tool conversation message {message.id!r} is missing tool_call_id"
            )
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id,
            "content": content,
        }

    def _conversation_tool_call(self, payload: dict[str, Any]) -> ChatCompletionToolCall:
        function = payload.get("function")
        if not isinstance(function, dict):
            function = {}
        return {
            "id": str(payload.get("id") or ""),
            "type": "function",
            "function": {
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or "{}"),
            },
        }


def _message_character_count(message: ChatMessage) -> int:
    count = len(str(message.get("role", "")))
    content = message.get("content")
    if isinstance(content, str):
        count += len(content)
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        count += len(_stable_json(tool_calls))
    tool_call_id = message.get("tool_call_id") if isinstance(message, dict) else None
    if isinstance(tool_call_id, str):
        count += len(tool_call_id)
    return count


def _tool_payload(tool: LLMToolDefinitionInput) -> dict[str, Any]:
    if hasattr(tool, "name"):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    return dict(tool)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
