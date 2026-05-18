from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.codex import CodexResponsesProvider, resolve_codex_access_token
from alpha_agent.llm.deepseek import DEEPSEEK_BASE_URL, DeepSeekProvider


def _response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        status_code,
        json=payload,
        request=httpx.Request("POST", "https://example.test"),
    )


def _config(**overrides: Any) -> AlphaConfig:
    values: dict[str, Any] = {
        "db_path": Path("alpha.db"),
        "log_dir": Path("logs"),
        "gateway_status_path": Path("gateway-status.json"),
    }
    values.update(overrides)
    return AlphaConfig(**values)


def test_deepseek_provider_uses_deepseek_defaults_and_api_key() -> None:
    config = _config(deepseek_api_key="deepseek-key")

    provider = DeepSeekProvider(config)

    assert provider.base_url == DEEPSEEK_BASE_URL
    assert provider.model == "deepseek-chat"
    assert provider.api_key == "deepseek-key"


def test_deepseek_v4_request_includes_thinking_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-1",
                "model": "deepseek-v4-pro",
                "choices": [{"message": {"content": "pong"}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        deepseek_api_key="deepseek-key",
        llm_model="deepseek-v4-pro",
        deepseek_reasoning_effort="high",
    )

    response = DeepSeekProvider(config).complete([{"role": "user", "content": "ping"}])

    assert response.content == "pong"
    assert response.provider == "deepseek"
    assert captured["url"] == "https://api.deepseek.com/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer deepseek-key"
    assert captured["json"]["thinking"] == {"type": "enabled"}
    assert captured["json"]["reasoning_effort"] == "high"


def test_deepseek_chat_omits_thinking_for_v3(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-1",
                "model": "deepseek-chat",
                "choices": [{"message": {"content": "pong"}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(deepseek_api_key="deepseek-key", llm_model="deepseek-chat")

    DeepSeekProvider(config).complete([{"role": "user", "content": "ping"}])

    assert "thinking" not in captured["json"]
    assert "reasoning_effort" not in captured["json"]


def test_codex_provider_uses_explicit_oauth_access_token() -> None:
    config = _config(codex_access_token="codex-token")

    provider = CodexResponsesProvider(config)

    assert provider.access_token == "codex-token"
    assert provider.base_url == "https://chatgpt.com/backend-api/codex"


def test_codex_provider_reads_codex_cli_oauth_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"tokens": {"access_token": "cli-access-token", "refresh_token": "refresh"}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    config = _config()

    assert resolve_codex_access_token(config) == "cli-access-token"


def test_codex_provider_uses_responses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _response(200, {"id": "resp-1", "output_text": "codex pong"})

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(codex_access_token="codex-token", llm_model="gpt-5.3-codex")

    response = CodexResponsesProvider(config).complete(
        [
            {"role": "system", "content": "You are Alpha."},
            {"role": "user", "content": "ping"},
        ]
    )

    assert response.content == "codex pong"
    assert response.provider == "codex"
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["Authorization"] == "Bearer codex-token"
    assert captured["json"]["model"] == "gpt-5.3-codex"
    assert captured["json"]["instructions"] == "You are Alpha."
    assert captured["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "ping"}]}
    ]
    assert captured["json"]["store"] is False


def test_codex_response_parser_reads_output_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return _response(
            200,
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "from output"}],
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(codex_access_token="codex-token")

    response = CodexResponsesProvider(config).complete([{"role": "user", "content": "ping"}])

    assert response.content == "from output"
