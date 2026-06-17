"""Provider-neutral helpers for validated LLM usage construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alpha_agent.llm.base import LLMUsage


def usage_int(value: Any) -> int | None:
    """Return an integer usage field, excluding bools."""

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def usage_mapping(value: Any) -> Mapping[str, Any] | None:
    """Return a mapping usage sub-object when present."""

    return value if isinstance(value, Mapping) else None


def build_llm_usage(
    *,
    total_tokens: int | None,
    cached_tokens: int | None,
    prompt_cache_miss_tokens: int | None,
    reasoning_tokens: int | None,
    completion_tokens: int | None,
) -> LLMUsage | None:
    """Construct ``LLMUsage`` when all required normalized counts are valid."""

    if (
        total_tokens is None
        or cached_tokens is None
        or prompt_cache_miss_tokens is None
        or reasoning_tokens is None
        or completion_tokens is None
    ):
        return None

    usage = LLMUsage(
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
        reasoning_tokens=reasoning_tokens,
        completion_tokens=completion_tokens,
    )
    if any(
        value < 0
        for value in (
            usage.total_tokens,
            usage.cached_tokens,
            usage.prompt_cache_miss_tokens,
            usage.reasoning_tokens,
            usage.completion_tokens,
        )
    ):
        return None
    return usage
