"""Xiaomi MiMo chat-completions provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
    LLMUsage,
)
from alpha_agent.llm.chat_completions import complete_chat_completions
from alpha_agent.llm.usage import build_llm_usage, usage_int, usage_mapping

MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_DEFAULT_MODEL = "mimo-v2.5"


class MiMoProvider:
    """Provider for Xiaomi MiMo's OpenAI-compatible chat completions API."""

    name = "mimo"

    def __init__(self, config: AlphaConfig, timeout: float = 60.0):
        api_key = config.mimo_api_key
        if not api_key:
            raise ValueError("mimo.api_key is required for mimo provider")
        self.base_url = MIMO_BASE_URL
        self.api_key = api_key
        self.model = config.mimo_model or MIMO_DEFAULT_MODEL
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
            usage_normalizer=normalize_mimo_usage,
        )


def normalize_mimo_usage(raw_usage: Any) -> LLMUsage | None:
    """Normalize MiMo's chat-completions usage payload."""

    usage = usage_mapping(raw_usage)
    if usage is None:
        return None

    completion_details = usage_mapping(usage.get("completion_tokens_details"))
    prompt_details = usage_mapping(usage.get("prompt_tokens_details"))
    cached_tokens = (
        usage_int(prompt_details.get("cached_tokens"))
        if prompt_details is not None
        else None
    )
    if cached_tokens is None:
        cached_tokens = 0

    prompt_tokens = usage_int(usage.get("prompt_tokens"))
    prompt_cache_miss_tokens = (
        prompt_tokens - cached_tokens if prompt_tokens is not None else None
    )
    reasoning_tokens = (
        usage_int(completion_details.get("reasoning_tokens"))
        if completion_details is not None
        else 0
    )
    if reasoning_tokens is None:
        reasoning_tokens = 0

    return build_llm_usage(
        total_tokens=usage_int(usage.get("total_tokens")),
        cached_tokens=cached_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        reasoning_tokens=reasoning_tokens,
        completion_tokens=usage_int(usage.get("completion_tokens")),
    )
