"""Shared argument and output validation helpers."""

from __future__ import annotations

import importlib
import json
import py_compile
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from alpha_agent.tools.base import JSONValue
from alpha_agent.tools.files.errors import FileToolError


def optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise FileToolError("value must be a string")
    text = value.strip()
    return text or None


def required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FileToolError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise FileToolError(f"{field_name} must not contain NUL characters")
    return value


def optional_bool(value: Any, field_name: str, *, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise FileToolError(f"{field_name} must be a boolean")
    return value


def bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    field_name: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < minimum:
        raise FileToolError(f"{field_name} must be at least {minimum}")
    return min(value, maximum)


def optional_non_negative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < 0:
        raise FileToolError(f"{field_name} must be non-negative")
    return value


def required_sha256(value: Any) -> str:
    text = optional_sha256(value)
    if text is None:
        raise FileToolError("expected_sha256 is required for existing files")
    return text


def optional_sha256(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise FileToolError("expected_sha256 must be a SHA-256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise FileToolError("expected_sha256 must be a SHA-256 hex digest")
    return normalized


def reject_nul_text(value: str, field_name: str) -> None:
    if "\x00" in value:
        raise FileToolError(f"{field_name} must not contain NUL characters")


def truncate_text(
    value: str, *, max_chars: int, marker: str = "\n... truncated ...\n"
) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False
    limit = max(0, max_chars - len(marker))
    return value[:limit] + marker, True


def syntax_validation(path: Path, content: str) -> list[JSONValue]:
    suffix = path.suffix.lower()
    diagnostics: list[JSONValue] = []
    try:
        if suffix == ".py":
            with tempfile.NamedTemporaryFile(
                "w", suffix=".py", encoding="utf-8", delete=True
            ) as handle:
                handle.write(content)
                handle.flush()
                py_compile.compile(handle.name, doraise=True)
        elif suffix == ".json":
            json.loads(content)
        elif suffix == ".toml":
            tomllib.loads(content)
        elif suffix in {".yaml", ".yml"}:
            try:
                yaml = importlib.import_module("yaml")
            except Exception:
                return []
            yaml.safe_load(content)
    except Exception as exc:
        diagnostics.append(
            {
                "path": path.as_posix(),
                "severity": "error",
                "message": str(exc),
                "validator": suffix.lstrip(".") or "text",
            }
        )
    return diagnostics
