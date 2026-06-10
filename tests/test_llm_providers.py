from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import httpx
import pytest

from alpha_agent.config import AlphaConfig
from alpha_agent.llm.base import AssistantChatMessage, ChatMessage, LLMToolDefinition
from alpha_agent.llm.codex import (
    CodexResponsesProvider,
    codex_responses_payload,
    resolve_codex_access_token,
)
from alpha_agent.llm.deepseek import DEEPSEEK_BASE_URL, DeepSeekProvider
from alpha_agent.llm.mimo import MIMO_BASE_URL, MIMO_DEFAULT_MODEL, MiMoProvider
from alpha_agent.llm.openai_compatible import OpenAICompatibleProvider


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


def test_mimo_provider_uses_mimo_defaults_and_api_key() -> None:
    config = _config(mimo_api_key="mimo-key")

    provider = MiMoProvider(config)

    assert provider.base_url == MIMO_BASE_URL
    assert provider.model == MIMO_DEFAULT_MODEL
    assert provider.api_key == "mimo-key"


def test_mimo_provider_requires_api_key() -> None:
    config = _config()

    with pytest.raises(ValueError, match="mimo.api_key is required"):
        MiMoProvider(config)


def test_mimo_provider_uses_api_key_header_and_normalizes_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    raw_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_context",
                "arguments": '{"query":"alpha"}',
            },
        }
    ]

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["url"] = args[0]
        captured["headers"] = kwargs["headers"]
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-mimo",
                "model": "mimo-v2.5-pro",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"content": None, "tool_calls": raw_tool_calls},
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(mimo_api_key="mimo-key")

    response = MiMoProvider(config).complete(
        [{"role": "user", "content": "ping"}],
        tools=[
            LLMToolDefinition(
                name="lookup_context",
                description="Look up relevant context.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_context"}},
        response_format={"type": "json_object"},
    )

    assert response.provider == "mimo"
    assert response.model == "mimo-v2.5-pro"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].arguments == {"query": "alpha"}
    assert response.metadata["request_payload"] == captured["json"]
    assert response.metadata["response_payload"]["id"] == "chatcmpl-mimo"
    assert captured["url"] == "https://api.xiaomimimo.com/v1/chat/completions"
    assert captured["headers"]["api-key"] == "mimo-key"
    assert "Authorization" not in captured["headers"]
    assert captured["headers"]["Content-Type"] == "application/json"
    assert captured["json"]["model"] == "mimo-v2.5-pro"
    assert captured["json"]["messages"] == [{"role": "user", "content": "ping"}]
    assert captured["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_context",
                "description": "Look up relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    assert captured["json"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup_context"},
    }
    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_deepseek_provider_replays_assistant_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    messages: list[ChatMessage] = [
        cast(ChatMessage, {"role": "user", "content": "ping", "reasoning_content": "drop"}),
        {
            "role": "assistant",
            "content": "pong",
            "reasoning_content": "prior reasoning",
        },
    ]

    DeepSeekProvider(config).complete(messages)

    assert captured["json"]["messages"] == [
        {"role": "user", "content": "ping"},
        {
            "role": "assistant",
            "content": "pong",
            "reasoning_content": "prior reasoning",
        },
    ]


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


def test_deepseek_provider_sends_tools_and_tool_choice_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    DeepSeekProvider(config).complete(
        [{"role": "user", "content": "ping"}],
        tools=[
            LLMToolDefinition(
                name="lookup_context",
                description="Look up relevant context.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        tool_choice={"type": "function", "function": {"name": "lookup_context"}},
    )

    assert captured["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_context",
                "description": "Look up relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]
    assert captured["json"]["tool_choice"] == {
        "type": "function",
        "function": {"name": "lookup_context"},
    }


def test_deepseek_provider_sends_response_format_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-1",
                "model": "deepseek-chat",
                "choices": [{"message": {"content": '{"ok":true}'}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(deepseek_api_key="deepseek-key", llm_model="deepseek-chat")

    DeepSeekProvider(config).complete(
        [{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert captured["json"]["response_format"] == {"type": "json_object"}


def test_deepseek_provider_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_context",
                "arguments": '{"query":"alpha","limit":2}',
            },
        }
    ]

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return _response(
            200,
            {
                "id": "chatcmpl-1",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "reasoning_content": "I should look this up.",
                            "tool_calls": raw_tool_calls,
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(deepseek_api_key="deepseek-key", llm_model="deepseek-chat")

    response = DeepSeekProvider(config).complete([{"role": "user", "content": "ping"}])

    assert response.content == ""
    assert response.reasoning_content == "I should look this up."
    assert response.finish_reason == "tool_calls"
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.id == "call_1"
    assert tool_call.name == "lookup_context"
    assert tool_call.arguments == {"query": "alpha", "limit": 2}
    assert tool_call.raw_arguments == '{"query":"alpha","limit":2}'
    assert response.metadata["response_id"] == "chatcmpl-1"
    assert response.metadata["finish_reason"] == "tool_calls"
    assert response.metadata["raw_tool_calls"] == raw_tool_calls
    assert response.metadata["tool_calls"][0]["arguments"] == {"query": "alpha", "limit": 2}
    assert response.metadata["normalized_tool_calls"][0]["raw_arguments"] == (
        '{"query":"alpha","limit":2}'
    )


def test_deepseek_provider_preserves_invalid_tool_call_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return _response(
            200,
            {
                "id": "chatcmpl-1",
                "model": "deepseek-chat",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup_context",
                                        "arguments": '{"query":',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(deepseek_api_key="deepseek-key", llm_model="deepseek-chat")

    response = DeepSeekProvider(config).complete([{"role": "user", "content": "ping"}])

    tool_call = response.tool_calls[0]
    assert tool_call.arguments == {}
    assert tool_call.raw_arguments == '{"query":'
    assert "arguments_parse_error" in tool_call.metadata
    assert tool_call.metadata["raw_arguments"] == '{"query":'
    assert response.metadata["tool_calls"][0]["metadata"]["raw_arguments"] == '{"query":'
    assert "arguments_parse_error" in response.metadata["tool_calls"][0]["metadata"]


def test_deepseek_response_without_reasoning_content_leaves_field_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
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

    response = DeepSeekProvider(config).complete([{"role": "user", "content": "ping"}])

    assert response.content == "pong"
    assert response.reasoning_content is None


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
    assert "tools" not in captured["json"]
    assert "tool_choice" not in captured["json"]


def test_openai_compatible_provider_parses_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup_context",
                "arguments": '{"query":"alpha"}',
            },
        }
    ]

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        return _response(
            200,
            {
                "id": "chatcmpl-compat",
                "model": "gpt-compatible",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {"content": None, "tool_calls": raw_tool_calls},
                    }
                ],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        compatible_base_url="https://compatible.example",
        compatible_api_key="compatible-key",
        llm_model="gpt-compatible",
    )

    response = OpenAICompatibleProvider(config).complete(
        [{"role": "user", "content": "ping"}]
    )

    assert response.content == ""
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "lookup_context"
    assert response.tool_calls[0].arguments == {"query": "alpha"}
    assert response.metadata["response_id"] == "chatcmpl-compat"
    assert response.metadata["finish_reason"] == "tool_calls"
    assert response.metadata["raw_tool_calls"] == raw_tool_calls
    assert response.metadata["normalized_tool_calls"][0]["raw_arguments"] == '{"query":"alpha"}'
    assert response.metadata["tool_calls"][0]["arguments"] == {"query": "alpha"}
    assert response.metadata["request_payload"]["messages"] == [
        {"role": "user", "content": "ping"}
    ]
    assert response.metadata["response_payload"]["id"] == "chatcmpl-compat"


def test_openai_compatible_provider_preserves_tool_messages_in_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-compat",
                "model": "gpt-compatible",
                "choices": [{"message": {"content": "done"}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        compatible_base_url="https://compatible.example",
        compatible_api_key="compatible-key",
        llm_model="gpt-compatible",
    )
    messages: list[ChatMessage] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup_context", "arguments": '{"query":"alpha"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":"ok"}'},
    ]

    OpenAICompatibleProvider(config).complete(messages)

    assert captured["json"]["messages"] == messages


def test_openai_compatible_provider_strips_internal_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-compat",
                "model": "gpt-compatible",
                "choices": [{"message": {"content": "done"}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        compatible_base_url="https://compatible.example",
        compatible_api_key="compatible-key",
        llm_model="gpt-compatible",
    )
    messages: list[ChatMessage] = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "internal reasoning",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_context",
                        "arguments": '{"query":"alpha"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":"ok"}'},
    ]

    OpenAICompatibleProvider(config).complete(messages)

    assert captured["json"]["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_context",
                        "arguments": '{"query":"alpha"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"result":"ok"}'},
    ]
    assistant_message = cast(AssistantChatMessage, messages[0])
    assert assistant_message["reasoning_content"] == "internal reasoning"


def test_openai_compatible_provider_sends_tools_with_none_tool_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-compat",
                "model": "gpt-compatible",
                "choices": [{"message": {"content": "done"}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        compatible_base_url="https://compatible.example",
        compatible_api_key="compatible-key",
        llm_model="gpt-compatible",
    )

    OpenAICompatibleProvider(config).complete(
        [{"role": "user", "content": "finalize"}],
        tools=[
            LLMToolDefinition(
                name="lookup_context",
                description="Look up relevant context.",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ],
        tool_choice="none",
    )

    assert captured["json"]["tool_choice"] == "none"
    assert captured["json"]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup_context",
                "description": "Look up relevant context.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


def test_openai_compatible_provider_sends_response_format_wire_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*args: Any, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return _response(
            200,
            {
                "id": "chatcmpl-compat",
                "model": "gpt-compatible",
                "choices": [{"message": {"content": '{"ok":true}'}}],
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    config = _config(
        compatible_base_url="https://compatible.example",
        compatible_api_key="compatible-key",
        llm_model="gpt-compatible",
    )

    OpenAICompatibleProvider(config).complete(
        [{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
    )

    assert captured["json"]["response_format"] == {"type": "json_object"}


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
        ],
        response_format={"type": "json_object"},
    )

    assert response.content == "codex pong"
    assert response.provider == "codex"
    assert response.metadata["request_payload"] == captured["json"]
    assert response.metadata["response_payload"] == {"id": "resp-1", "output_text": "codex pong"}
    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["Authorization"] == "Bearer codex-token"
    assert captured["json"]["model"] == "gpt-5.3-codex"
    assert captured["json"]["instructions"] == "You are Alpha."
    assert captured["json"]["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "ping"}]}
    ]
    assert captured["json"]["store"] is False
    assert "response_format" not in captured["json"]


def test_codex_responses_payload_strips_internal_assistant_reasoning_content() -> None:
    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are Alpha."},
        {
            "role": "assistant",
            "content": "Visible prior answer.",
            "reasoning_content": "private reasoning must not leave runtime",
        },
        {"role": "user", "content": "Continue."},
    ]

    payload = codex_responses_payload(model="gpt-5.3-codex", messages=messages)

    assert payload == {
        "model": "gpt-5.3-codex",
        "input": [
            {
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Visible prior answer."}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Continue."}],
            },
        ],
        "store": False,
        "instructions": "You are Alpha.",
    }


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
