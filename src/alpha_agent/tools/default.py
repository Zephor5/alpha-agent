"""Default runtime tool registry construction."""

from __future__ import annotations

from alpha_agent.config import AlphaConfig
from alpha_agent.tools.bash import BashTool
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.tools.web_search import TavilyWebSearchTool


def build_default_tool_registry(config: AlphaConfig) -> ToolRegistry:
    """Build the default tool registry for configured runtime agents."""

    registry = ToolRegistry()
    if config.bash_tool.enabled:
        registry.register(BashTool(config=config.bash_tool, secret_values=_config_secrets(config)))
    if config.tavily_api_key:
        registry.register(TavilyWebSearchTool(api_key=config.tavily_api_key))
    return registry


def _config_secrets(config: AlphaConfig) -> tuple[str, ...]:
    values = (
        config.compatible_api_key,
        config.deepseek_api_key,
        config.codex_access_token,
        config.tavily_api_key,
    )
    return tuple(value for value in values if value)
