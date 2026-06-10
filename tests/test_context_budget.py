from __future__ import annotations

from alpha_agent.config import LLMContextConfig, load_config
from alpha_agent.runtime.context_budget import (
    estimate_context_budget,
    estimate_text_tokens,
)


def test_token_estimator_uses_english_words_plus_cjk_characters() -> None:
    assert estimate_text_tokens("Hello, alpha agent. 你好世界") == 3 + 4


def test_context_budget_includes_messages_tools_output_reserve_and_safety_margin() -> None:
    config = LLMContextConfig(
        expected_output_reserve_tokens=7,
        safety_margin_tokens=3,
    )
    messages = [{"role": "user", "content": "Hello 世界"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Find records",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]

    estimate = estimate_context_budget(
        messages,
        tools=tools,
        context_config=config,
        max_context_tokens=100,
    )

    assert estimate.message_tokens == estimate_text_tokens(
        '{"content":"Hello 世界","role":"user"}'
    )
    assert estimate.tool_schema_tokens == estimate_text_tokens(
        '{"function":{"description":"Find records","name":"lookup",'
        '"parameters":{"properties":{"query":{"type":"string"}},'
        '"required":["query"],"type":"object"}},"type":"function"}'
    )
    assert estimate.used_context_tokens == (
        estimate.message_tokens
        + estimate.tool_schema_tokens
        + config.expected_output_reserve_tokens
        + config.safety_margin_tokens
    )
    assert estimate.remaining_context_tokens == 100 - estimate.used_context_tokens


def test_provider_max_context_config_is_loaded_and_used(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "mimo"

[llm.providers.openai-compatible]
max_context_tokens = 258400

[llm.providers.deepseek]
max_context_tokens = 1000000

[llm.providers.mimo]
max_context_tokens = 1000000
""",
        encoding="utf-8",
    )
    config = load_config(env_file=None, config_file=config_path)

    estimate = estimate_context_budget(
        [{"role": "user", "content": "hello"}],
        context_config=config.llm_context,
        max_context_tokens=config.max_context_tokens_for_provider(config.llm_provider),
    )

    assert config.max_context_tokens_for_provider("openai") == 258400
    assert config.max_context_tokens_for_provider("compatible") == 258400
    assert config.max_context_tokens_for_provider("mimo") == 1000000
    assert estimate.max_context_tokens == 1000000
    assert estimate.remaining_context_tokens == 1000000 - estimate.used_context_tokens


def test_mimo_default_context_window_is_one_million_tokens(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(env_file=None, config_file=config_path)

    assert config.max_context_tokens_for_provider("mimo") == 1000000
