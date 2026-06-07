"""Compatibility accessors for file tool configuration."""

from __future__ import annotations

from typing import Any

from alpha_agent.config import FileToolConfig


def max_read_lines(config: FileToolConfig) -> int:
    return _positive_int(config, "max_read_lines", default=200)


def max_search_results(config: FileToolConfig) -> int:
    return _positive_int(config, "max_search_results", default=100)


def max_glob_results(config: FileToolConfig) -> int:
    return _positive_int(config, "max_glob_results", default=500)


def create_parent_dirs_enabled(config: FileToolConfig) -> bool:
    value = getattr(config, "create_parent_dirs_enabled", False)
    return bool(value)


def _positive_int(
    config: FileToolConfig,
    name: str,
    *,
    default: int,
) -> int:
    value: Any = getattr(config, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value
