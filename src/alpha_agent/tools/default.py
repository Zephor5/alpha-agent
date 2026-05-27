"""Default runtime tool registry construction."""

from __future__ import annotations

from alpha_agent.config import AlphaConfig
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.tools.web_search import TavilyWebSearchTool


def build_default_tool_registry(config: AlphaConfig) -> ToolRegistry:
    """Build the default tool registry for configured runtime agents."""

    registry = ToolRegistry()
    if config.tavily_api_key:
        registry.register(TavilyWebSearchTool(api_key=config.tavily_api_key))
    return registry
