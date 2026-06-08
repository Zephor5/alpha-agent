from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from alpha_agent.config import AlphaConfig, BashToolConfig
from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.files import (
    FILE_GLOB_TOOL_NAME,
    FILE_READ_TOOL_NAME,
    FILE_SEARCH_TOOL_NAME,
)
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from alpha_agent.tools.memory_recall import MEMORY_RECALL_TOOL_NAME
from alpha_agent.tools.web_fetch import TavilyWebFetchTool


def _tool_context(tmp_path: Path | None = None) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path or Path("."),
        check_canceled=lambda _stage: None,
    )


def _safe_url_checker(_url: str) -> None:
    return None


def test_web_fetch_tool_exposes_general_fetch_schema() -> None:
    tool = TavilyWebFetchTool(api_key="tvly-test", url_safety_checker=_safe_url_checker)

    assert tool.spec.name == "web_fetch"
    assert tool.spec.toolset == "web"
    assert tool.spec.read_only is True
    assert tool.spec.concurrency_safe is True
    assert tool.spec.max_result_size_chars == 100_000
    assert tool.spec.parameters["required"] == ["url"]
    assert tool.spec.parameters["additionalProperties"] is False
    assert set(tool.spec.parameters["properties"]) == {
        "url",
        "extract_depth",
        "format",
        "timeout_seconds",
    }
    assert tool.spec.parameters["properties"]["extract_depth"]["enum"] == ["advanced", "basic"]
    assert tool.spec.parameters["properties"]["format"]["enum"] == ["markdown", "text"]
    assert tool.spec.parameters["properties"]["timeout_seconds"]["minimum"] == 1
    assert tool.spec.parameters["properties"]["timeout_seconds"]["maximum"] == 120
    serialized_spec = json.dumps(tool.spec.to_dict())
    assert "Tavily" not in serialized_spec
    assert "tavily" not in serialized_spec.lower()


def test_web_fetch_tool_posts_sanitized_request_and_formats_success_result() -> None:
    captured: dict[str, object] = {}
    checked_urls: list[str] = []

    def url_checker(url: str) -> None:
        checked_urls.append(url)

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "content": "Fallback content.",
                        "raw_content": "# Example Article\n\nExtracted content.",
                    }
                ],
                "failed_results": [],
                "response_time": 0.42,
                "request_id": "req_456",
                "usage": {"credits": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(api_key="tvly-test", client=client, url_safety_checker=url_checker)

    result = tool.run(
        {
            "url": "https://example.com/article?utm_source=test",
            "extract_depth": "advanced",
            "format": "text",
            "timeout_seconds": 12,
            "unknown": "dropped",
        },
        _tool_context(),
    )

    assert checked_urls == ["https://example.com/article?utm_source=test"]
    assert captured == {
        "url": "https://api.tavily.com/extract",
        "authorization": "Bearer tvly-test",
        "body": {
            "urls": ["https://example.com/article?utm_source=test"],
            "extract_depth": "advanced",
            "format": "text",
            "timeout": 12,
        },
        "timeout": {"connect": 12.0, "read": 12.0, "write": 12.0, "pool": 12.0},
    }
    assert result.name == "web_fetch"
    assert result.output == {
        "url": "https://example.com/article?utm_source=test",
        "final_url": "https://example.com/article",
        "title": "Example Article",
        "content": "# Example Article\n\nExtracted content.",
        "content_format": "text",
        "status": "success",
        "error": None,
    }
    assert "provider" not in result.output
    assert result.metadata == {
        "provider": "tavily",
        "request_id": "req_456",
        "result_count": 1,
        "failed_count": 0,
        "usage": {"credits": 1},
        "timeout_seconds": {"requested": 12, "effective": 12},
    }


def test_web_fetch_tool_clamps_provider_timeout_and_records_metadata() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        captured["timeout"] = request.extensions.get("timeout")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/slow",
                        "raw_content": "Slow content.",
                    }
                ],
                "failed_results": [],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(
        api_key="tvly-test",
        client=client,
        url_safety_checker=_safe_url_checker,
    )

    result = tool.run(
        {"url": "https://example.com/slow", "timeout_seconds": 120},
        _tool_context(),
    )

    assert captured == {
        "body": {
            "urls": ["https://example.com/slow"],
            "extract_depth": "basic",
            "format": "markdown",
            "timeout": 60,
        },
        "timeout": {"connect": 120.0, "read": 120.0, "write": 120.0, "pool": 120.0},
    }
    assert result.metadata["timeout_seconds"] == {"requested": 120, "effective": 60}


def test_web_fetch_tool_falls_back_to_content_when_raw_content_is_missing() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/plain",
                        "title": "Plain Article",
                        "content": "Plain extracted text.",
                    }
                ],
                "failed_results": [],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(
        api_key="tvly-test",
        client=client,
        url_safety_checker=_safe_url_checker,
    )

    result = tool.run({"url": "https://example.com/plain"}, _tool_context())

    output = result.output
    assert isinstance(output, dict)
    assert output["content"] == "Plain extracted text."
    assert output["content_format"] == "markdown"
    assert output["status"] == "success"


def test_web_fetch_tool_formats_failed_result() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [],
                "failed_results": [
                    {
                        "url": "https://example.com/missing",
                        "error": "Could not extract content.",
                    }
                ],
                "request_id": "req_failed",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(
        api_key="tvly-test",
        client=client,
        url_safety_checker=_safe_url_checker,
    )

    result = tool.run({"url": "https://example.com/missing"}, _tool_context())

    assert result.output == {
        "url": "https://example.com/missing",
        "final_url": "https://example.com/missing",
        "title": "",
        "content": "",
        "content_format": "markdown",
        "status": "failed",
        "error": "Could not extract content.",
    }
    assert "provider" not in result.output
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["result_count"] == 0
    assert result.metadata["failed_count"] == 1


def test_web_fetch_tool_returns_neutral_failure_result_for_provider_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "upstream detail"}, request=request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(
        api_key="tvly-test",
        client=client,
        url_safety_checker=_safe_url_checker,
    )

    result = tool.run({"url": "https://example.com/error"}, _tool_context())

    assert result.output == {
        "url": "https://example.com/error",
        "final_url": "https://example.com/error",
        "title": "",
        "content": "",
        "content_format": "markdown",
        "status": "failed",
        "error": "Web fetch failed with HTTP status 500.",
    }
    serialized_output = json.dumps(result.output)
    assert "Tavily" not in serialized_output
    assert "api.tavily.com" not in serialized_output
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["error_type"] == "HTTPStatusError"
    assert result.metadata["http_status"] == 500


def test_web_fetch_tool_returns_neutral_failure_result_for_invalid_provider_json() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(
        api_key="tvly-test",
        client=client,
        url_safety_checker=_safe_url_checker,
    )

    result = tool.run({"url": "https://example.com/bad-json"}, _tool_context())

    assert result.output == {
        "url": "https://example.com/bad-json",
        "final_url": "https://example.com/bad-json",
        "title": "",
        "content": "",
        "content_format": "markdown",
        "status": "failed",
        "error": "Web fetch returned an invalid response.",
    }
    serialized_output = json.dumps(result.output)
    assert "Tavily" not in serialized_output
    assert "api.tavily.com" not in serialized_output
    assert result.metadata["provider"] == "tavily"
    assert result.metadata["error_type"] == "JSONDecodeError"


def test_web_fetch_tool_rejects_unsafe_url_before_http_request() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={})

    def url_checker(_url: str) -> None:
        raise ValueError("URL is not allowed")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    tool = TavilyWebFetchTool(api_key="tvly-test", client=client, url_safety_checker=url_checker)

    with pytest.raises(ValueError, match="not allowed"):
        tool.run({"url": "http://127.0.0.1/"}, _tool_context())

    assert requests == []


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({}, "url is required"),
        ({"url": "https://example.com", "extract_depth": "deep"}, "extract_depth must be one of"),
        ({"url": "https://example.com", "format": "html"}, "format must be one of"),
        ({"url": "https://example.com", "timeout_seconds": 0}, "timeout_seconds must be between"),
        ({"url": "https://example.com", "timeout_seconds": 121}, "timeout_seconds must be between"),
    ],
)
def test_web_fetch_tool_validates_arguments(arguments: dict[str, object], match: str) -> None:
    tool = TavilyWebFetchTool(api_key="tvly-test", url_safety_checker=_safe_url_checker)

    with pytest.raises(ValueError, match=match):
        tool.run(arguments, _tool_context())


def test_web_fetch_tool_requires_api_key() -> None:
    tool = TavilyWebFetchTool(api_key="", url_safety_checker=_safe_url_checker)

    with pytest.raises(ValueError, match="web access API key"):
        tool.run({"url": "https://example.com"}, _tool_context())


def test_tool_registry_includes_web_fetch_when_web_key_is_configured(tmp_path: Path) -> None:
    configured = AlphaConfig(
        db_path=tmp_path / "configured.db",
        log_dir=tmp_path / "logs",
        gateway_status_path=tmp_path / "gateway.json",
        tavily_api_key="tvly-test",
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
    default_names = [
        MEMORY_PROPOSE_TOOL_NAME,
        MEMORY_RECALL_TOOL_NAME,
        FILE_GLOB_TOOL_NAME,
        FILE_READ_TOOL_NAME,
        FILE_SEARCH_TOOL_NAME,
    ]

    assert build_tool_registry(configured).names() == [*default_names, "web_search", "web_fetch"]
    assert build_tool_registry(both_configured).names() == [
        *default_names,
        "bash",
        "web_search",
        "web_fetch",
    ]
    assert [
        tool.name for tool in build_tool_registry(both_configured).to_llm_tool_definitions()
    ] == [*default_names, "bash", "web_search", "web_fetch"]
