from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from alpha_agent.config import AlphaConfig, BashToolConfig
from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME
from alpha_agent.tools.web_search import TavilyWebSearchTool


def _tool_context(tmp_path: Path | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path or Path("."),
        check_canceled=lambda _stage: None,
    )


def test_tavily_web_search_tool_exposes_general_search_schema() -> None:
    tool = TavilyWebSearchTool(api_key="tvly-test")

    assert tool.spec.name == "web_search"
    assert "Tavily" not in tool.spec.description
    assert tool.spec.parameters["required"] == ["query"]
    assert set(tool.spec.parameters["properties"]) >= {
        "query",
        "search_depth",
        "max_results",
        "time_range",
        "start_date",
        "end_date",
        "country",
        "include_domains",
        "exclude_domains",
    }


def test_tavily_web_search_tool_posts_sanitized_request_and_formats_results() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "query": "alpha agent web search",
                "answer": "A short synthesized answer.",
                "results": [
                    {
                        "title": "Result One",
                        "url": "https://example.com/one",
                        "content": "Useful snippet.",
                        "score": 0.82,
                        "raw_content": "ignored raw content",
                    },
                    {
                        "title": "Result Two",
                        "url": "https://example.com/two",
                        "content": "Another snippet.",
                    },
                ],
                "response_time": 1.23,
                "request_id": "req_123",
                "usage": {"credits": 2},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebSearchTool(api_key="tvly-test", client=client)

    result = tool.run(
        {
            "query": "alpha agent web search",
            "search_depth": "advanced",
            "max_results": 6,
            "time_range": "day",
            "start_date": "2026-05-26",
            "end_date": "2026-05-27",
            "country": "united states",
            "include_domains": ["example.com", ""],
            "exclude_domains": ["spam.example"],
            "include_answer": True,
            "unknown": "dropped",
        },
        _tool_context(),
    )

    assert captured == {
        "url": "https://api.tavily.com/search",
        "authorization": "Bearer tvly-test",
        "body": {
            "query": "alpha agent web search",
            "search_depth": "advanced",
            "max_results": 6,
            "time_range": "day",
            "start_date": "2026-05-26",
            "end_date": "2026-05-27",
            "country": "united states",
            "include_domains": ["example.com"],
            "exclude_domains": ["spam.example"],
            "include_answer": True,
            "include_raw_content": False,
        },
    }
    assert result.name == "web_search"
    assert result.output == {
        "answer": "A short synthesized answer.",
        "query": "alpha agent web search",
        "request_id": "req_123",
        "response_time": 1.23,
        "results": [
            {
                "content": "Useful snippet.",
                "score": 0.82,
                "title": "Result One",
                "url": "https://example.com/one",
            },
            {
                "content": "Another snippet.",
                "score": None,
                "title": "Result Two",
                "url": "https://example.com/two",
            },
        ],
    }
    assert result.metadata == {
        "provider": "tavily",
        "request_id": "req_123",
        "result_count": 2,
        "usage": {"credits": 2},
    }


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "query is required"),
        ({"query": "x", "search_depth": "deep"}, "search_depth must be one of"),
        ({"query": "x", "max_results": 0}, "max_results must be between 1 and 20"),
        ({"query": "x", "time_range": "hour"}, "time_range must be one of"),
        ({"query": "x", "start_date": "05/26/2026"}, "start_date must use YYYY-MM-DD"),
    ],
)
def test_tavily_web_search_tool_validates_arguments(
    arguments: dict[str, object],
    match: str,
) -> None:
    tool = TavilyWebSearchTool(api_key="tvly-test")

    with pytest.raises(ValueError, match=match):
        tool.run(arguments, _tool_context())


def test_tavily_web_search_tool_requires_api_key() -> None:
    tool = TavilyWebSearchTool(api_key="")

    with pytest.raises(ValueError, match="tavily.api_key"):
        tool.run({"query": "alpha"}, _tool_context())


def test_tool_registry_includes_memory_propose_and_configured_tools(
    tmp_path: Path,
) -> None:
    empty_config = AlphaConfig(
        db_path=tmp_path / "empty.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
    )
    configured = AlphaConfig(
        db_path=tmp_path / "configured.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        tavily_api_key="tvly-test",
    )
    bash_configured = AlphaConfig(
        db_path=tmp_path / "bash.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        bash_tool=BashToolConfig(
            enabled=True,
            default_workdir=tmp_path,
            allowed_workdirs=(tmp_path,),
        ),
    )
    both_configured = AlphaConfig(
        db_path=tmp_path / "both.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        bash_tool=BashToolConfig(
            enabled=True,
            default_workdir=tmp_path,
            allowed_workdirs=(tmp_path,),
        ),
        tavily_api_key="tvly-test",
    )

    empty_registry = build_tool_registry(empty_config)

    assert empty_registry.names() == [MEMORY_PROPOSE_TOOL_NAME, MEMORY_RECALL_TOOL_NAME]
    assert [tool.name for tool in empty_registry.to_llm_tool_definitions()] == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
    ]
    assert build_tool_registry(configured).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        "web_search",
    ]
    assert build_tool_registry(bash_configured).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        "bash",
    ]
    assert build_tool_registry(both_configured).names() == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        "bash",
        "web_search",
    ]
    assert [
        tool.name for tool in build_tool_registry(both_configured).to_llm_tool_definitions()
    ] == [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        "bash",
        "web_search",
    ]
