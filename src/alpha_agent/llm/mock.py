"""Deterministic mock LLM provider."""

from __future__ import annotations

from collections.abc import Sequence

from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
)


class MockLLMProvider:
    """Local deterministic provider for tests and development."""

    name = "mock"

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        user_message = _message_content(messages[-1]) if messages else ""
        current_message = self._extract_current_message(user_message)
        content = f"Mock response: I heard you say: {current_message}."
        return LLMResponse(content=content, model="mock", provider=self.name, metadata={})

    def _extract_current_message(self, prompt_content: str) -> str:
        marker = "## Current User Message"
        if marker not in prompt_content:
            return prompt_content.strip()[:200]
        return prompt_content.split(marker, 1)[1].strip().strip('"')[:200]


def _message_content(message: ChatMessage) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""
