"""LLM provider interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict


class ChatMessage(TypedDict):
    """OpenAI-style chat message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM completion response."""

    content: str
    model: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Synchronous LLM provider interface."""

    name: str

    def complete(self, messages: list[ChatMessage]) -> LLMResponse:
        """Complete a chat-style prompt."""
