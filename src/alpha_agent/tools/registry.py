"""Simple tool registry for future expansion."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.tools.base import Tool


@dataclass
class ToolRegistry:
    """In-memory registry for explicit tool lookup."""

    _tools: dict[str, Tool] = field(default_factory=dict)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())
