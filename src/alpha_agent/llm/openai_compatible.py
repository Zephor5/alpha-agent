"""OpenAI-compatible HTTP provider."""

from __future__ import annotations

from typing import Any

import httpx

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import ChatMessage, LLMResponse

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

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        """Call the configured compatible chat completions API."""

        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "messages": messages},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
        return LLMResponse(
            content=str(content),
            model=str(payload.get("model", self.model)),
            provider=self.name,
            metadata={"response_id": payload.get("id")},
        )
