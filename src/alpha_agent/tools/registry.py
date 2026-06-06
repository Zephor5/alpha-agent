"""Simple tool registry for future expansion."""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.llm.base import LLMToolDefinition
from alpha_agent.tools.base import (
    Tool,
    ToolAvailability,
    ToolSpec,
    tool_availability,
    tool_spec,
)


@dataclass(frozen=True)
class ToolInfo:
    """Runtime introspection record for one registered tool."""

    spec: ToolSpec
    availability: ToolAvailability

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-friendly representation."""

        return {
            "name": self.spec.name,
            "description": self.spec.description,
            "tool_spec": self.spec.to_dict(),
            "availability": self.availability.to_dict(),
        }


@dataclass
class ToolRegistry:
    """In-memory registry for explicit tool lookup."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Register or replace a tool by its explicit name."""

        spec = tool_spec(tool)
        checker = getattr(tool, "check_available", None)
        if not callable(checker):
            raise TypeError("tool check_available must be implemented")
        self._tools[spec.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a registered tool by name."""

        return self._tools.get(name)

    def list(self) -> builtins.list[Tool]:
        """Return registered tools in registration order."""

        return [self._tools[name] for name in self.names()]

    def names(self) -> builtins.list[str]:
        """Return registered tool names in registration order."""

        return list(self._tools)

    def spec_for(self, name: str) -> ToolSpec | None:
        """Return the static spec for a registered tool."""

        tool = self.get(name)
        return tool_spec(tool) if tool is not None else None

    def availability_for(self, name: str) -> ToolAvailability:
        """Return current availability for a registered tool."""

        tool = self.get(name)
        if tool is None:
            return ToolAvailability.unavailable(f"Unknown tool: {name}")
        return tool_availability(tool)

    def describe(self, name: str) -> ToolInfo | None:
        """Return runtime introspection data for a registered tool."""

        tool = self.get(name)
        if tool is None:
            return None
        spec = tool_spec(tool)
        return ToolInfo(
            spec=spec,
            availability=tool_availability(tool),
        )

    def list_info(self, *, include_unavailable: bool = True) -> builtins.list[ToolInfo]:
        """Return runtime introspection records in registration order."""

        infos = [self.describe(name) for name in self.names()]
        return [
            info
            for info in infos
            if info is not None and (include_unavailable or info.availability.available)
        ]

    def list_available(self) -> builtins.list[Tool]:
        """Return currently available tools in registration order."""

        return [tool for tool in self.list() if tool_availability(tool).available]

    def available_names(self) -> builtins.list[str]:
        """Return currently available tool names in registration order."""

        return [tool_spec(tool).name for tool in self.list_available()]

    def to_llm_tool_definitions(
        self,
        *,
        include_unavailable: bool = False,
    ) -> builtins.list[LLMToolDefinition]:
        """Return provider-neutral LLM tool definitions in registration order.

        Runtime governance metadata and availability details are intentionally
        not serialized into LLM tool definitions.
        """

        definitions: list[LLMToolDefinition] = []
        for tool in self._tools.values():
            if not include_unavailable and not tool_availability(tool).available:
                continue
            spec = tool_spec(tool)
            definitions.append(
                LLMToolDefinition(
                    name=spec.name,
                    description=spec.description,
                    parameters=dict(spec.parameters),
                    strict=spec.strict,
                )
            )
        return definitions
