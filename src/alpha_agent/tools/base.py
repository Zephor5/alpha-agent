"""Minimal tool interface reserved for future tool execution."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

type JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]


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
    output: JSONValue
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime context available to one concrete tool invocation."""

    session_id: str
    tool_call_id: str | None
    output_dir: Path
    check_canceled: Callable[[str], None]


def tool_output_kind(output: JSONValue) -> str:
    """Return the persistence kind for a tool output payload."""

    return "text" if isinstance(output, str) else "json"


def tool_output_to_model_content(output: JSONValue) -> str:
    """Serialize a tool output for provider tool-result message content."""

    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class Tool(Protocol):
    """Protocol for tool implementations.

    Tools may also expose provider-neutral JSON schema metadata:
    ``parameters`` as a JSON schema object and ``strict`` as an optional
    structured-output strictness hint. The registry supplies an empty object
    schema for tools that do not define parameters.
    """

    name: str
    description: str

    def run(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Run the tool with validated arguments."""


class ToolWithParameters(Tool, Protocol):
    """Tool protocol extension for tools with an explicit parameter schema."""

    parameters: Mapping[str, Any]


class ToolWithStrict(Tool, Protocol):
    """Tool protocol extension for tools that opt into strict schema handling."""

    strict: bool | None


class ToolWithTraceArguments(Tool, Protocol):
    """Tool protocol extension for custom tool.started argument summaries."""

    def trace_arguments(self, arguments: dict[str, Any]) -> Mapping[str, Any]:
        """Return trace-safe arguments for runtime trace metadata."""
