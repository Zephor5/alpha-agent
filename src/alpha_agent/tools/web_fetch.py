"""General web page fetch tool backed by an extraction provider."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from alpha_agent.tools.base import (
    ToolAvailability,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)
from alpha_agent.tools.url_safety import validate_public_http_url

TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"
WEB_FETCH_TOOL_NAME = "web_fetch"
EXTRACT_DEPTH_VALUES = {"advanced", "basic"}
FORMAT_VALUES = {"markdown", "text"}
MAX_PROVIDER_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class _FetchRequest:
    url: str
    content_format: str
    local_timeout: float
    requested_timeout_seconds: int | None
    effective_timeout_seconds: int | None
    payload: dict[str, Any]


class _ProviderFetchError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        http_status: int | None = None,
    ):
        super().__init__(message)
        self.public_message = message
        self.error_type = error_type
        self.http_status = http_status


class TavilyWebFetchTool:
    """Provider-specific implementation for the generic web_fetch tool."""

    spec = ToolSpec(
        name=WEB_FETCH_TOOL_NAME,
        description=(
            "Fetch and extract readable content from a specific public web page URL. "
            "Use this when the URL is already known and the page content is needed."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "A complete http or https URL to fetch.",
                },
                "extract_depth": {
                    "type": "string",
                    "enum": sorted(EXTRACT_DEPTH_VALUES),
                    "description": "Extraction thoroughness. Use basic by default.",
                },
                "format": {
                    "type": "string",
                    "enum": sorted(FORMAT_VALUES),
                    "description": "Content format to return.",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 120,
                    "description": (
                        "Requested extraction timeout in seconds. The backend may cap "
                        "values above its supported range."
                    ),
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
        toolset="web",
        read_only=True,
        concurrency_safe=True,
        max_result_size_chars=100_000,
    )

    def __init__(
        self,
        api_key: str | None,
        *,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        url_safety_checker: Callable[[str], None] = validate_public_http_url,
    ):
        self.api_key = (api_key or "").strip()
        self.client = client
        self.timeout = timeout
        self.url_safety_checker = url_safety_checker

    def check_available(self) -> ToolAvailability:
        """Return whether generic web fetch can currently run."""

        if not self.api_key:
            return ToolAvailability.unavailable("web access API key is required")
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Fetch one public URL and return normalized extracted content."""

        del context
        if not self.api_key:
            raise ValueError("web access API key is required to use web_fetch")

        request = self._request(arguments)
        self.url_safety_checker(request.url)
        try:
            response = self._post(
                payload=request.payload,
                timeout=request.local_timeout,
            )
        except _ProviderFetchError as exc:
            return ToolResult(
                name=self.spec.name,
                output=self._failed_output(
                    requested_url=request.url,
                    content_format=request.content_format,
                    error=exc.public_message,
                ),
                metadata=self._metadata(
                    request=request,
                    response=None,
                    result_count=0,
                    failed_count=1,
                    error=exc,
                ),
            )
        normalized = self._response_payload(
            response,
            requested_url=request.url,
            content_format=request.content_format,
        )
        return ToolResult(
            name=self.spec.name,
            output=normalized,
            metadata=self._metadata(
                request=request,
                response=response,
                result_count=self._mapping_count(response.get("results")),
                failed_count=self._mapping_count(response.get("failed_results")),
            ),
        )

    def _request(self, arguments: Mapping[str, Any]) -> _FetchRequest:
        url = str(arguments.get("url") or "").strip()
        if not url:
            raise ValueError("url is required")

        extract_depth = (
            self._enum_value(arguments, "extract_depth", EXTRACT_DEPTH_VALUES) or "basic"
        )
        content_format = self._enum_value(arguments, "format", FORMAT_VALUES) or "markdown"
        timeout_seconds = self._timeout_seconds(arguments)
        effective_timeout_seconds = (
            min(timeout_seconds, MAX_PROVIDER_TIMEOUT_SECONDS)
            if timeout_seconds is not None
            else None
        )
        payload: dict[str, Any] = {
            "urls": [url],
            "extract_depth": extract_depth,
            "format": content_format,
        }
        if effective_timeout_seconds is not None:
            payload["timeout"] = effective_timeout_seconds
        return _FetchRequest(
            url=url,
            content_format=content_format,
            local_timeout=float(timeout_seconds if timeout_seconds is not None else self.timeout),
            requested_timeout_seconds=timeout_seconds,
            effective_timeout_seconds=effective_timeout_seconds,
            payload=payload,
        )

    def _post(self, *, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self.client is not None:
                response = self.client.post(
                    TAVILY_EXTRACT_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            else:
                response = httpx.post(
                    TAVILY_EXTRACT_URL,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            raise _ProviderFetchError(
                f"Web fetch failed with HTTP status {status}.",
                error_type=type(exc).__name__,
                http_status=status,
            ) from exc
        except httpx.TimeoutException as exc:
            raise _ProviderFetchError(
                "Web fetch timed out.",
                error_type=type(exc).__name__,
            ) from exc
        except httpx.RequestError as exc:
            raise _ProviderFetchError(
                "Web fetch request failed before a response was received.",
                error_type=type(exc).__name__,
            ) from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise _ProviderFetchError(
                "Web fetch returned an invalid response.",
                error_type=type(exc).__name__,
            ) from exc
        if not isinstance(data, dict):
            raise _ProviderFetchError(
                "Web fetch returned an invalid response.",
                error_type="InvalidResponse",
            )
        return data

    def _response_payload(
        self,
        response: Mapping[str, Any],
        *,
        requested_url: str,
        content_format: str,
    ) -> dict[str, Any]:
        success = self._first_mapping(response.get("results"))
        if success is not None:
            return {
                "url": requested_url,
                "final_url": self._text(
                    success.get("final_url") or success.get("url") or requested_url
                ),
                "title": self._text(success.get("title")),
                "content": self._content(success),
                "content_format": content_format,
                "status": "success",
                "error": None,
            }

        failure = self._first_mapping(response.get("failed_results"))
        return self._failed_output(
            requested_url=requested_url,
            content_format=content_format,
            error=self._failure_error(failure),
            final_url=self._text((failure or {}).get("url") or requested_url),
        )

    def _failed_output(
        self,
        *,
        requested_url: str,
        content_format: str,
        error: str,
        final_url: str | None = None,
    ) -> dict[str, Any]:
        return {
            "url": requested_url,
            "final_url": final_url or requested_url,
            "title": "",
            "content": "",
            "content_format": content_format,
            "status": "failed",
            "error": error,
        }

    def _metadata(
        self,
        *,
        request: _FetchRequest,
        response: Mapping[str, Any] | None,
        result_count: int,
        failed_count: int,
        error: _ProviderFetchError | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "provider": "tavily",
            "request_id": response.get("request_id") if response is not None else None,
            "result_count": result_count,
            "failed_count": failed_count,
            "usage": response.get("usage") if response is not None else None,
            "timeout_seconds": {
                "requested": request.requested_timeout_seconds,
                "effective": request.effective_timeout_seconds,
            },
        }
        if error is not None:
            metadata["error_type"] = error.error_type
            if error.http_status is not None:
                metadata["http_status"] = error.http_status
        return metadata

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

    def _timeout_seconds(self, arguments: Mapping[str, Any]) -> int | None:
        value = arguments.get("timeout_seconds")
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("timeout_seconds must be between 1 and 120")
        if isinstance(value, int):
            timeout_seconds = value
        elif isinstance(value, str):
            try:
                timeout_seconds = int(value.strip())
            except ValueError as exc:
                raise ValueError("timeout_seconds must be between 1 and 120") from exc
            if str(timeout_seconds) != value.strip():
                raise ValueError("timeout_seconds must be between 1 and 120")
        else:
            raise ValueError("timeout_seconds must be between 1 and 120")
        if timeout_seconds < 1 or timeout_seconds > 120:
            raise ValueError("timeout_seconds must be between 1 and 120")
        return timeout_seconds

    def _first_mapping(self, value: Any) -> Mapping[str, Any] | None:
        if not isinstance(value, list):
            return None
        for item in value:
            if isinstance(item, Mapping):
                return item
        return None

    def _mapping_count(self, value: Any) -> int:
        if not isinstance(value, list):
            return 0
        return sum(1 for item in value if isinstance(item, Mapping))

    def _content(self, result: Mapping[str, Any]) -> str:
        raw_content = self._text(result.get("raw_content"))
        if raw_content:
            return raw_content
        return self._text(result.get("content"))

    def _failure_error(self, failure: Mapping[str, Any] | None) -> str:
        if failure is None:
            return "No extracted content returned"
        for key in ("error", "message", "reason"):
            value = self._text(failure.get(key))
            if value:
                return self._neutral_error(value, fallback="Extraction failed")
        return "Extraction failed"

    def _neutral_error(self, value: str, *, fallback: str) -> str:
        lowered = value.lower()
        if "tavily" in lowered or "api.tavily.com" in lowered:
            return fallback
        return value

    def _text(self, value: Any) -> str:
        return "" if value is None else str(value)
