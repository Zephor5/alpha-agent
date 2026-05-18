"""DeepSeek chat-completions provider."""

from __future__ import annotations

from typing import Any

import httpx

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import ChatMessage, LLMResponse

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

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        """Call DeepSeek and normalize the chat completion response."""

        body: dict[str, Any] = {"model": self.model, "messages": messages}
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
        content = payload["choices"][0]["message"]["content"]
        return LLMResponse(
            content=str(content),
            model=str(payload.get("model", self.model)),
            provider=self.name,
            metadata={"response_id": payload.get("id")},
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
