"""Xiaomi MiMo chat-completions provider."""

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

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5-pro"


class MiMoProvider:
    """Provider for Xiaomi MiMo's OpenAI-compatible chat completions API."""

    name = "mimo"

    def __init__(self, config: AlphaConfig, timeout: float = 60.0):
        api_key = config.mimo_api_key
        if not api_key:
            raise ValueError("mimo.api_key is required for mimo provider")
        self.base_url = MIMO_BASE_URL
        self.api_key = api_key
        self.model = config.llm_model or MIMO_DEFAULT_MODEL
        self.timeout = timeout

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        """Call MiMo and normalize the chat completion response."""

        return complete_chat_completions(
            base_url=self.base_url,
            headers={"api-key": self.api_key},
            model=self.model,
            provider=self.name,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            timeout=self.timeout,
        )
