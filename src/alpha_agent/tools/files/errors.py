"""Shared file tool errors."""

from __future__ import annotations

from collections.abc import Mapping

from alpha_agent.tools.base import JSONValue, ToolUserError


class FileToolError(ToolUserError):
    """Raised when a file tool request violates local file policy."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "file_tool_error",
        details: Mapping[str, JSONValue] | None = None,
    ):
        super().__init__(message, code=code, details=details)
