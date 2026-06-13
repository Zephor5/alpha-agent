"""OpenAI-compatible HTTP provider."""

from __future__ import annotations

from collections.abc import Sequence

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.llm.chat_completions import complete_chat_completions

OPENAI_COMPATIBLE_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAICompatibleProvider:
    """Provider for /chat/completions compatible APIs."""

    name = "openai-compatible"

    def __init__(self, config: AlphaConfig, timeout: float = 60.0):
        if not config.compatible_base_url:
            raise ValueError("compatible.base_url is required for openai-compatible provider")
        if not config.compatible_api_key:
            raise ValueError("compatible.api_key is required for openai-compatible provider")
        self.base_url = config.compatible_base_url.rstrip("/")
        self.api_key = config.compatible_api_key
        self.model = config.compatible_model or OPENAI_COMPATIBLE_DEFAULT_MODEL
        self.timeout = timeout

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        """Call the configured compatible chat completions API."""

        return complete_chat_completions(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            model=self.model,
            provider=self.name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            timeout=self.timeout,
        )
