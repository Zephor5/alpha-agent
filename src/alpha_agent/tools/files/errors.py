"""Shared file tool errors."""

from __future__ import annotations


class FileToolError(ValueError):
    """Raised when a file tool request violates local file policy."""
