"""Simple tool registry for future expansion."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from alpha_agent.llm.base import LLMToolDefinition
from alpha_agent.tools.base import Tool

EMPTY_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


@dataclass
class ToolRegistry:
    """In-memory registry for explicit tool lookup."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        """Register or replace a tool by its explicit name."""

        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a registered tool by name."""

        return self._tools.get(name)

    def list(self) -> builtins.list[Tool]:
        """Return registered tools in deterministic name order."""

        return [self._tools[name] for name in sorted(self._tools)]

    def names(self) -> builtins.list[str]:
        """Return registered tool names in deterministic order."""

        return sorted(self._tools)

    def to_llm_tool_definitions(self) -> builtins.list[LLMToolDefinition]:
        """Return provider-neutral LLM tool definitions in deterministic order."""

        definitions: list[LLMToolDefinition] = []
        for tool in self.list():
            parameters = self._parameters_for(tool)
            strict = getattr(tool, "strict", None)
            definitions.append(
                LLMToolDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=parameters,
                    strict=strict if strict is None else bool(strict),
                )
            )
        return definitions

    def _parameters_for(self, tool: Tool) -> dict[str, Any]:
        parameters = getattr(tool, "parameters", None)
        if parameters is None:
            return dict(EMPTY_TOOL_PARAMETERS)
        if not isinstance(parameters, Mapping):
            raise TypeError(f"Tool {tool.name} parameters must be a JSON schema mapping")
        return dict(parameters)
