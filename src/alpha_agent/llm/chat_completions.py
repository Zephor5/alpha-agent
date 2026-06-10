"""Shared OpenAI-compatible chat-completions HTTP client."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import httpx

from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
    chat_completion_messages_payload,
    openai_compatible_response,
    openai_compatible_response_format_payload,
    openai_compatible_tool_choice_payload,
    openai_compatible_tool_payload,
)


def complete_chat_completions(
    *,
    base_url: str,
    headers: Mapping[str, str],
    model: str,
    provider: str,
    messages: Sequence[ChatMessage],
    timeout: float,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
    tool_choice: LLMToolChoice | None = None,
    response_format: LLMResponseFormat | None = None,
    include_reasoning_content: bool = False,
    extra_body: Mapping[str, Any] | None = None,
) -> LLMResponse:
    """Call an OpenAI-compatible chat-completions endpoint and normalize it."""

    body: dict[str, Any] = {
        "model": model,
        "messages": chat_completion_messages_payload(
            messages,
            include_reasoning_content=include_reasoning_content,
        ),
    }
    if tools is not None:
        body["tools"] = [openai_compatible_tool_payload(tool) for tool in tools]
    if tool_choice is not None:
        body["tool_choice"] = openai_compatible_tool_choice_payload(tool_choice)
    if response_format is not None:
        body["response_format"] = openai_compatible_response_format_payload(
            response_format
        )
    if extra_body:
        body.update(dict(extra_body))

    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Content-Type": "application/json",
            **dict(headers),
        },
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    normalized = openai_compatible_response(
        payload=payload,
        fallback_model=model,
        provider=provider,
    )
    return replace(
        normalized,
        metadata={
            **normalized.metadata,
            "request_payload": body,
            "response_payload": payload,
        },
    )
