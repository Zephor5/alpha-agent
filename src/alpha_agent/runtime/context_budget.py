"""Deterministic context budget estimates for LLM requests."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any

from alpha_agent.config import LLMContextConfig
from alpha_agent.llm.base import ChatMessage, LLMToolDefinitionInput

_ENGLISH_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z0-9_]+)?")


@dataclass(frozen=True)
class ContextBudgetEstimate:
    """Token estimate components for one projected LLM request."""

    message_tokens: int
    tool_schema_tokens: int
    expected_output_reserve_tokens: int
    safety_margin_tokens: int
    used_context_tokens: int
    max_context_tokens: int
    remaining_context_tokens: int


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens as English-like words plus CJK characters."""

    return len(_ENGLISH_WORD_RE.findall(text)) + sum(1 for char in text if _is_cjk(char))


def estimate_context_budget(
    messages: Sequence[ChatMessage | Mapping[str, Any]],
    *,
    tools: Sequence[LLMToolDefinitionInput | Mapping[str, Any]] | None = None,
    context_config: LLMContextConfig | None = None,
    max_context_tokens: int,
) -> ContextBudgetEstimate:
    """Estimate used and remaining context using stable serialized payloads."""

    config = context_config or LLMContextConfig()
    message_tokens = sum(estimate_text_tokens(stable_json(message)) for message in messages)
    tool_schema_tokens = sum(estimate_text_tokens(stable_json(tool)) for tool in tools or [])
    used_context_tokens = (
        message_tokens
        + tool_schema_tokens
        + config.expected_output_reserve_tokens
        + config.safety_margin_tokens
    )
    return ContextBudgetEstimate(
        message_tokens=message_tokens,
        tool_schema_tokens=tool_schema_tokens,
        expected_output_reserve_tokens=config.expected_output_reserve_tokens,
        safety_margin_tokens=config.safety_margin_tokens,
        used_context_tokens=used_context_tokens,
        max_context_tokens=max_context_tokens,
        remaining_context_tokens=max_context_tokens - used_context_tokens,
    )


def stable_json(value: Any) -> str:
    """Serialize a JSON-like payload deterministically for token estimation."""

    return json.dumps(
        _json_payload(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_payload(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_payload(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_payload(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_payload(item) for item in value]
    return str(value)


def _is_cjk(char: str) -> bool:
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
    )
