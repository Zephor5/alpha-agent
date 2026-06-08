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


class ToolUserError(ValueError):
    """User-actionable tool error that can be returned to the model as structured JSON."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "tool_error",
        details: Mapping[str, JSONValue] | None = None,
    ):
        super().__init__(_required_text(message, "message"))
        self.code = _required_text(code, "code")
        self.details = dict(details or {})


@dataclass
class TurnToolState:
    """Mutable tool governance state scoped to one agent turn."""

    repeated_call_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    """Single static contract for one tool implementation."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    strict: bool = True
    toolset: str = "default"
    read_only: bool = False
    concurrency_safe: bool = False
    destructive: bool = False
    requires_user_interaction: bool = False
    max_result_size_chars: int = 100_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required_text(self.name, "name"))
        object.__setattr__(
            self,
            "description",
            _required_text(self.description, "description"),
        )
        object.__setattr__(self, "toolset", _required_text(self.toolset, "toolset"))
        if not isinstance(self.parameters, Mapping):
            raise ValueError("parameters must be a JSON schema mapping")
        object.__setattr__(self, "parameters", dict(self.parameters))
        for field_name in (
            "strict",
            "read_only",
            "concurrency_safe",
            "destructive",
            "requires_user_interaction",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be a boolean")
        if (
            isinstance(self.max_result_size_chars, bool)
            or not isinstance(self.max_result_size_chars, int)
            or self.max_result_size_chars < 1
        ):
            raise ValueError("max_result_size_chars must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly representation."""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": self.strict,
            "toolset": self.toolset,
            "read_only": self.read_only,
            "concurrency_safe": self.concurrency_safe,
            "destructive": self.destructive,
            "requires_user_interaction": self.requires_user_interaction,
            "max_result_size_chars": self.max_result_size_chars,
        }


@dataclass(frozen=True)
class ToolAvailability:
    """Current availability for a registered tool."""

    available: bool = True
    reason: str | None = None
    details: Mapping[str, JSONValue] = field(default_factory=dict)

    @classmethod
    def unavailable(
        cls,
        reason: str,
        *,
        details: Mapping[str, JSONValue] | None = None,
    ) -> ToolAvailability:
        """Build a standard unavailable result."""

        return cls(available=False, reason=reason, details=dict(details or {}))

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a stable JSON-friendly representation."""

        payload: dict[str, JSONValue] = {"available": self.available}
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True)
class ToolExecutionContext:
    """Runtime context available to one concrete tool invocation."""

    session_id: str
    tool_call_id: str | None
    output_dir: Path
    check_canceled: Callable[[str], None]
    extensions: Mapping[str, Any] = field(default_factory=dict)
    turn_state: TurnToolState = field(default_factory=TurnToolState)


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

    Static schema and governance data lives in ``spec``. Dynamic runtime
    availability remains a method because it may depend on configuration,
    credentials, or environment state.
    """

    @property
    def spec(self) -> ToolSpec:
        """Return the static tool contract."""
        ...

    def check_available(self) -> ToolAvailability:
        """Return whether the tool can currently run."""

    def run(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        """Run the tool with validated arguments."""


class ToolWithTraceArguments(Tool, Protocol):
    """Tool protocol extension for custom tool.started argument summaries."""

    def trace_arguments(self, arguments: dict[str, Any]) -> Mapping[str, Any]:
        """Return trace-safe arguments for runtime trace metadata."""


def tool_spec(tool: Tool) -> ToolSpec:
    """Return the validated static tool contract."""

    try:
        spec = tool.spec
    except AttributeError as exc:
        raise TypeError("tool spec must be a ToolSpec") from exc
    if not isinstance(spec, ToolSpec):
        raise TypeError("tool spec must be a ToolSpec")
    return spec


def tool_availability(tool: Tool) -> ToolAvailability:
    """Return current availability for a tool without leaking check failures."""

    try:
        availability = tool.check_available()
        if not isinstance(availability, ToolAvailability):
            raise TypeError("check_available must return ToolAvailability")
        return availability
    except Exception as exc:
        return ToolAvailability.unavailable(
            f"availability check failed: {exc}",
            details={"error_type": type(exc).__name__},
        )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value.strip()
