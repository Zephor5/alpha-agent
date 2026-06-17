"""OpenAI Codex Responses API provider."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
    LLMUsage,
)
from alpha_agent.llm.usage import build_llm_usage, usage_int, usage_mapping

CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
CODEX_DEFAULT_MODEL = "gpt-5.3-codex"


class CodexResponsesProvider:
    """Provider for the Codex Responses API surface used by Codex-style models."""

    name = "codex"

    def __init__(self, config: AlphaConfig, timeout: float = 60.0):
        access_token = resolve_codex_access_token(config)
        if not access_token:
            raise ValueError(
                "Codex OAuth token not found. Set ALPHA_CODEX_ACCESS_TOKEN or run Codex CLI "
                "login so CODEX_HOME/auth.json or ~/.codex/auth.json contains tokens."
            )
        self.base_url = CODEX_BASE_URL
        self.access_token = access_token
        self.model = config.codex_model or CODEX_DEFAULT_MODEL
        self.timeout = timeout

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        """Call the Responses API and normalize the assistant text."""

        del tools, tool_choice, response_format
        body = codex_responses_payload(model=self.model, messages=messages)
        response = httpx.post(
            f"{self.base_url}/responses",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        return LLMResponse(
            content=_extract_response_text(payload),
            model=str(payload.get("model", self.model)),
            provider=self.name,
            metadata={
                "response_id": payload.get("id"),
                "request_payload": body,
                "response_payload": payload,
            },
            usage=normalize_codex_usage(payload.get("usage")),
        )


def resolve_codex_access_token(config: AlphaConfig) -> str | None:
    """Resolve a Codex OAuth bearer token from env/config or Codex CLI auth state."""

    explicit = (config.codex_access_token or "").strip()
    if explicit:
        return explicit
    return _read_codex_cli_access_token()


def _read_codex_cli_access_token() -> str | None:
    codex_home = os.getenv("CODEX_HOME", "").strip() or str(Path.home() / ".codex")
    auth_path = Path(codex_home).expanduser() / "auth.json"
    if not auth_path.is_file():
        return None
    try:
        payload = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if isinstance(tokens, dict):
        token = tokens.get("access_token")
        if isinstance(token, str) and token.strip():
            return token.strip()

    token = payload.get("access_token") if isinstance(payload, dict) else None
    if isinstance(token, str) and token.strip():
        return token.strip()
    return None


def codex_responses_payload(*, model: str, messages: list[ChatMessage]) -> dict[str, Any]:
    """Convert OpenAI-style chat messages to a minimal Responses API payload."""

    instructions = "\n\n".join(
        _message_content(message).strip()
        for message in messages
        if message["role"] == "system" and _message_content(message).strip()
    )
    input_items = [
        _message_to_input_item(message)
        for message in messages
        if message["role"] != "system" and _message_content(message).strip()
    ]
    payload: dict[str, Any] = {
        "model": model,
        "input": input_items,
        "store": False,
    }
    if instructions:
        payload["instructions"] = instructions
    return payload


def normalize_codex_usage(raw_usage: Any) -> LLMUsage | None:
    """Normalize Codex Responses API usage payloads."""

    usage = usage_mapping(raw_usage)
    if usage is None:
        return None

    output_details = usage_mapping(usage.get("output_tokens_details"))
    input_details = usage_mapping(usage.get("input_tokens_details"))
    cached_tokens = (
        usage_int(input_details.get("cached_tokens"))
        if input_details is not None
        else None
    )
    if cached_tokens is None:
        cached_tokens = 0

    input_tokens = usage_int(usage.get("input_tokens"))
    prompt_cache_miss_tokens = (
        input_tokens - cached_tokens if input_tokens is not None else None
    )
    reasoning_tokens = (
        usage_int(output_details.get("reasoning_tokens"))
        if output_details is not None
        else 0
    )
    if reasoning_tokens is None:
        reasoning_tokens = 0

    return build_llm_usage(
        total_tokens=usage_int(usage.get("total_tokens")),
        cached_tokens=cached_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        reasoning_tokens=reasoning_tokens,
        completion_tokens=usage_int(usage.get("output_tokens")),
    )


def _message_to_input_item(message: ChatMessage) -> dict[str, Any]:
    role = message["role"]
    content = _message_content(message)
    if role == "assistant":
        return {
            "role": "assistant",
            "content": [{"type": "output_text", "text": content}],
        }
    if role == "tool":
        return {
            "role": "user",
            "content": [{"type": "input_text", "text": f"Tool result:\n{content}"}],
        }
    return {
        "role": "user",
        "content": [{"type": "input_text", "text": content}],
    }


def _message_content(message: ChatMessage) -> str:
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = payload.get("output")
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content

    return ""
