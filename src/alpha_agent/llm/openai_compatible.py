"""OpenAI-compatible HTTP provider."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import httpx

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
    chat_completion_messages_payload,
    openai_compatible_response,
    openai_compatible_tool_choice_payload,
    openai_compatible_tool_payload,
)

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
        self.model = config.llm_model or OPENAI_COMPATIBLE_DEFAULT_MODEL
        self.timeout = timeout

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        """Call the configured compatible chat completions API."""

        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_completion_messages_payload(messages),
        }
        if tools is not None:
            body["tools"] = [openai_compatible_tool_payload(tool) for tool in tools]
        if tool_choice is not None:
            body["tool_choice"] = openai_compatible_tool_choice_payload(tool_choice)
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        normalized = openai_compatible_response(
            payload=payload,
            fallback_model=self.model,
            provider=self.name,
        )
        return replace(
            normalized,
            metadata={
                **normalized.metadata,
                "request_payload": body,
                "response_payload": payload,
            },
        )
