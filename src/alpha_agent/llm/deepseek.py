"""DeepSeek chat-completions provider."""

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

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


class DeepSeekProvider:
    """Provider for DeepSeek's OpenAI-compatible chat completions API."""

    name = "deepseek"

    def __init__(self, config: AlphaConfig, timeout: float = 60.0):
        api_key = config.deepseek_api_key
        if not api_key:
            raise ValueError("deepseek.api_key is required for deepseek provider")
        self.base_url = DEEPSEEK_BASE_URL
        self.api_key = api_key
        self.model = config.llm_model or DEEPSEEK_DEFAULT_MODEL
        self.reasoning_enabled = config.deepseek_reasoning_enabled
        self.reasoning_effort = config.deepseek_reasoning_effort
        self.timeout = timeout

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        """Call DeepSeek and normalize the chat completion response."""

        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_completion_messages_payload(
                messages,
                include_reasoning_content=True,
            ),
        }
        if tools is not None:
            body["tools"] = [openai_compatible_tool_payload(tool) for tool in tools]
        if tool_choice is not None:
            body["tool_choice"] = openai_compatible_tool_choice_payload(tool_choice)
        body.update(
            deepseek_reasoning_parameters(
                model=self.model,
                enabled=self.reasoning_enabled,
                effort=self.reasoning_effort,
            )
        )
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
            reasoning_content=_deepseek_reasoning_content(payload),
            metadata={
                **normalized.metadata,
                "request_payload": body,
                "response_payload": payload,
            },
        )


def deepseek_reasoning_parameters(
    *,
    model: str | None,
    enabled: bool,
    effort: str | None,
) -> dict[str, Any]:
    """Return DeepSeek V4/R1 thinking parameters for direct HTTP JSON bodies."""

    if not _model_supports_thinking(model):
        return {}

    params: dict[str, Any] = {
        "thinking": {"type": "enabled" if enabled else "disabled"},
    }
    if not enabled:
        return params

    normalized_effort = (effort or "").strip().lower()
    if normalized_effort in {"xhigh", "max"}:
        params["reasoning_effort"] = "max"
    elif normalized_effort in {"low", "medium", "high"}:
        params["reasoning_effort"] = normalized_effort
    return params


def _model_supports_thinking(model: str | None) -> bool:
    """Return whether a DeepSeek model family expects explicit thinking config."""

    value = (model or "").strip().lower()
    if not value:
        return False
    if value.startswith("deepseek-v") and not value.startswith("deepseek-v3"):
        return True
    return value == "deepseek-reasoner"


def _deepseek_reasoning_content(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    reasoning_content = message.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else None
