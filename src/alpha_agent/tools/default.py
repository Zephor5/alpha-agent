"""Default runtime tool registry construction."""

from __future__ import annotations

from alpha_agent.config import AlphaConfig
from alpha_agent.tools.bash import BashTool
from alpha_agent.tools.files import (
    FileGlobTool,
    FilePatchTool,
    FileReadTool,
    FileSearchTool,
    FileWriteTool,
)
from alpha_agent.tools.memory_propose import MemoryProposeTool
from alpha_agent.tools.memory_recall import MemoryRecallTool
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.tools.web_fetch import TavilyWebFetchTool
from alpha_agent.tools.web_search import TavilyWebSearchTool


def build_tool_registry(config: AlphaConfig | None = None) -> ToolRegistry:
    """Build the complete runtime tool registry."""

    registry = ToolRegistry()
    registry.register(MemoryProposeTool())
    registry.register(MemoryRecallTool())
    if config is None:
        return registry
    if config.file_tool.enabled:
        registry.register(FileGlobTool(config=config.file_tool))
        registry.register(FileReadTool(config=config.file_tool))
        registry.register(FileSearchTool(config=config.file_tool))
        if config.file_tool.patch_enabled and config.file_tool.write_roots:
            registry.register(FilePatchTool(config=config.file_tool))
            registry.register(FileWriteTool(config=config.file_tool))
    if config.bash_tool.enabled:
        registry.register(BashTool(config=config.bash_tool, secret_values=_config_secrets(config)))
    if config.tavily_api_key:
        registry.register(TavilyWebSearchTool(api_key=config.tavily_api_key))
        registry.register(TavilyWebFetchTool(api_key=config.tavily_api_key))
    return registry


def _config_secrets(config: AlphaConfig) -> tuple[str, ...]:
    values = (
        config.compatible_api_key,
        config.deepseek_api_key,
        config.codex_access_token,
        config.tavily_api_key,
    )
    return tuple(value for value in values if value)
