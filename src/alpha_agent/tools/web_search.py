"""General web search tool backed by Tavily Search."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from alpha_agent.tools.base import ToolResult

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
SEARCH_DEPTH_VALUES = {"advanced", "basic", "fast", "ultra-fast"}
TIME_RANGE_VALUES = {"day", "week", "month", "year", "d", "w", "m", "y"}
TOPIC_VALUES = {"general", "news", "finance"}
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class TavilyWebSearchTool:
    """Provider-specific implementation for the generic web_search tool."""

    name = "web_search"
    description = (
        "Search the public web for current or factual information. Use this when the answer "
        "depends on recent events, external sources, precise citations, or facts not already "
        "available in the conversation."
    )
    strict = True
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The natural-language search query.",
            },
            "search_depth": {
                "type": "string",
                "enum": sorted(SEARCH_DEPTH_VALUES),
                "description": "Latency versus relevance tradeoff. Use basic by default.",
            },
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of search results to return.",
            },
            "topic": {
                "type": "string",
                "enum": sorted(TOPIC_VALUES),
                "description": "Optional broad result category.",
            },
            "time_range": {
                "type": "string",
                "enum": sorted(TIME_RANGE_VALUES),
                "description": "Optional recency filter such as day, week, month, or year.",
            },
            "start_date": {
                "type": "string",
                "description": "Optional inclusive lower date bound in YYYY-MM-DD format.",
            },
            "end_date": {
                "type": "string",
                "description": "Optional inclusive upper date bound in YYYY-MM-DD format.",
            },
            "country": {
                "type": "string",
                "description": "Optional country name to boost local results.",
            },
            "include_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to include.",
            },
            "exclude_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional domains to exclude.",
            },
            "include_answer": {
                "type": "boolean",
                "description": "Whether to include a generated short answer from search results.",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        api_key: str | None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.api_key = (api_key or "").strip()
        self.client = client
        self.timeout = timeout

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        """Run a web search and return compact JSON suitable for LLM follow-up."""

        if not self.api_key:
            raise ValueError("tavily.api_key is required to use web_search")
        payload = self._request_payload(arguments)
        response = self._post(payload)
        normalized = self._response_payload(response)
        return ToolResult(
            name=self.name,
            content=json.dumps(normalized, sort_keys=True, separators=(",", ":")),
            metadata={
                "provider": "tavily",
                "request_id": response.get("request_id"),
                "result_count": len(normalized["results"]),
                "usage": response.get("usage"),
            },
        )

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.client is not None:
            response = self.client.post(
                TAVILY_SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        else:
            response = httpx.post(
                TAVILY_SEARCH_URL,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Tavily search response must be a JSON object")
        return data

    def _request_payload(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        query = str(arguments.get("query") or "").strip()
        if not query:
            raise ValueError("query is required")

        payload: dict[str, Any] = {
            "query": query,
            "search_depth": self._enum_value(arguments, "search_depth", SEARCH_DEPTH_VALUES)
            or "basic",
            "max_results": self._max_results(arguments),
            "include_raw_content": False,
        }

        topic = self._enum_value(arguments, "topic", TOPIC_VALUES)
        if topic:
            payload["topic"] = topic
        time_range = self._enum_value(arguments, "time_range", TIME_RANGE_VALUES)
        if time_range:
            payload["time_range"] = time_range
        for key in ("start_date", "end_date"):
            value = self._date_value(arguments, key)
            if value:
                payload[key] = value
        country = str(arguments.get("country") or "").strip().lower()
        if country:
            payload["country"] = country
        for key in ("include_domains", "exclude_domains"):
            values = self._string_list(arguments.get(key))
            if values:
                payload[key] = values
        if "include_answer" in arguments:
            payload["include_answer"] = bool(arguments["include_answer"])
        return payload

    def _response_payload(self, response: Mapping[str, Any]) -> dict[str, Any]:
        raw_results = response.get("results")
        raw_results = raw_results if isinstance(raw_results, list) else []
        results: list[dict[str, Any]] = []
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping):
                continue
            results.append(
                {
                    "title": str(raw_result.get("title") or ""),
                    "url": str(raw_result.get("url") or ""),
                    "content": str(raw_result.get("content") or ""),
                    "score": raw_result.get("score"),
                }
            )
        return {
            "answer": response.get("answer"),
            "query": response.get("query"),
            "request_id": response.get("request_id"),
            "response_time": response.get("response_time"),
            "results": results,
        }

    def _enum_value(
        self,
        arguments: Mapping[str, Any],
        key: str,
        allowed: set[str],
    ) -> str | None:
        value = arguments.get(key)
        if value is None or value == "":
            return None
        normalized = str(value).strip().lower()
        if normalized not in allowed:
            options = ", ".join(sorted(allowed))
            raise ValueError(f"{key} must be one of: {options}")
        return normalized

    def _max_results(self, arguments: Mapping[str, Any]) -> int:
        value = arguments.get("max_results", 6)
        try:
            max_results = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_results must be between 1 and 20") from exc
        if max_results < 1 or max_results > 20:
            raise ValueError("max_results must be between 1 and 20")
        return max_results

    def _date_value(self, arguments: Mapping[str, Any], key: str) -> str | None:
        value = str(arguments.get(key) or "").strip()
        if not value:
            return None
        if not DATE_PATTERN.match(value):
            raise ValueError(f"{key} must use YYYY-MM-DD format")
        return value

    def _string_list(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            items: Sequence[Any] = [value]
        elif isinstance(value, Sequence):
            items = value
        else:
            raise ValueError("domain filters must be strings or arrays of strings")
        return [str(item).strip() for item in items if str(item).strip()]
