"""Minimal tool interface reserved for future tool execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A requested tool invocation."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    """Result returned by a tool."""

    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """Protocol for tool implementations.

    Tools may also expose provider-neutral JSON schema metadata:
    ``parameters`` as a JSON schema object and ``strict`` as an optional
    structured-output strictness hint. The registry supplies an empty object
    schema for tools that do not define parameters.
    """

    name: str
    description: str

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool with validated arguments."""


class ToolWithParameters(Tool, Protocol):
    """Tool protocol extension for tools with an explicit parameter schema."""

    parameters: Mapping[str, Any]


class ToolWithStrict(Tool, Protocol):
    """Tool protocol extension for tools that opt into strict schema handling."""

    strict: bool | None
