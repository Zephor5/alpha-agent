from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.config import load_config, read_config_value, write_default_config


def test_load_config_reads_toml_file(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[runtime]
db_path = "~/custom-alpha/alpha.db"
log_dir = "~/custom-alpha/logs"
gateway_status_path = "~/custom-alpha/status.json"
daemon_socket_path = "~/custom-alpha/daemon.sock"
daemon_status_path = "~/custom-alpha/daemon-status.json"

[llm]
provider = "deepseek"
model = "deepseek-v4-pro"
debug_logging = true

[compatible]
base_url = "https://compatible.example/v1"
api_key = "compatible-key"

[context]
recent_tail_messages = 6

[deepseek]
api_key = "deepseek-key"
reasoning_enabled = false
reasoning_effort = "high"
""",
        encoding="utf-8",
    )

    config = load_config(env_file=None, config_file=config_path)

    assert config.db_path == Path("~/custom-alpha/alpha.db").expanduser()
    assert config.log_dir == Path("~/custom-alpha/logs").expanduser()
    assert config.gateway_status_path == Path("~/custom-alpha/status.json").expanduser()
    assert config.daemon_socket_path == Path("~/custom-alpha/daemon.sock").expanduser()
    assert config.daemon_status_path == Path("~/custom-alpha/daemon-status.json").expanduser()
    assert config.llm_provider == "deepseek"
    assert config.context_recent_tail_messages == 6
    assert config.deepseek_api_key == "deepseek-key"
    assert config.llm_model == "deepseek-v4-pro"
    assert config.compatible_base_url == "https://compatible.example/v1"
    assert config.compatible_api_key == "compatible-key"
    assert config.deepseek_reasoning_enabled is False
    assert config.deepseek_reasoning_effort == "high"
    assert config.llm_debug_logging is True


def test_environment_overrides_config_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "deepseek"

[deepseek]
api_key = "from-file"

[compatible]
base_url = "from-file"
api_key = "compatible-file-key"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    monkeypatch.setenv("ALPHA_LLM_DEBUG_LOGGING", "true")
    monkeypatch.setenv("ALPHA_DEEPSEEK_API_KEY", "from-env")
    monkeypatch.setenv("ALPHA_COMPATIBLE_BASE_URL", "from-env")
    monkeypatch.setenv("ALPHA_COMPATIBLE_API_KEY", "compatible-env-key")

    config = load_config(env_file=None, config_file=config_path)

    assert config.llm_provider == "mock"
    assert config.llm_debug_logging is True
    assert config.deepseek_api_key == "from-env"
    assert config.compatible_base_url == "from-env"
    assert config.compatible_api_key == "compatible-env-key"


def test_write_default_config_is_non_destructive(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"

    created = write_default_config(config_path)
    second = write_default_config(config_path)

    assert created is True
    assert second is False
    assert "[llm]" in config_path.read_text(encoding="utf-8")


def test_config_cli_init_and_show(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    init_result = runner.invoke(app, ["config", "init"])
    show_result = runner.invoke(app, ["config", "show"])

    assert init_result.exit_code == 0
    assert show_result.exit_code == 0
    assert config_path.exists()
    assert str(config_path) in show_result.output
    assert "llm_provider" in show_result.output
    assert "context_recent_tail_messages" in show_result.output
    assert "compatible_base_url" not in show_result.output


def test_config_show_includes_base_url_only_for_compatible_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    runner.invoke(app, ["config", "set", "llm.provider", "openai-compatible"])
    show_result = runner.invoke(app, ["config", "show"])

    assert show_result.exit_code == 0
    assert "compatible_base_url" in show_result.output


def test_config_cli_set_and_get(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    set_provider = runner.invoke(app, ["config", "set", "llm.provider", "codex"])
    set_debug = runner.invoke(app, ["config", "set", "llm.debug_logging", "true"])
    set_context = runner.invoke(app, ["config", "set", "context.recent_tail_messages", "6"])
    get_provider = runner.invoke(app, ["config", "get", "llm.provider"])

    assert set_provider.exit_code == 0
    assert set_debug.exit_code == 0
    assert set_context.exit_code == 0
    assert get_provider.exit_code == 0
    assert "codex" in get_provider.output
    config = load_config(env_file=None, config_file=config_path)
    assert config.llm_provider == "codex"
    assert config.llm_debug_logging is True
    assert config.context_recent_tail_messages == 6


def test_config_set_rejects_unknown_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.unknown", "value"])

    assert result.exit_code != 0
    assert "Unsupported config key" in result.output


@pytest.mark.parametrize(
    "key",
    [
        "llm.base_url",
        "llm.api_key",
        "deepseek.base_url",
        "codex.base_url",
        "deepseek.model",
        "codex.model",
    ],
)
def test_provider_specific_transport_and_model_keys_are_not_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", key, "some-value"])

    assert result.exit_code != 0
    assert "Unsupported config key" in result.output


def test_config_set_rejects_invalid_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.provider", "deepseek2"])

    assert result.exit_code != 0
    assert "Invalid value for llm.provider" in result.output


def test_config_set_rejects_invalid_reasoning_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "deepseek.reasoning_effort", "extreme"])

    assert result.exit_code != 0
    assert "Invalid value for deepseek.reasoning_effort" in result.output


def test_config_set_rejects_non_positive_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "context.recent_tail_messages", "0"])

    assert result.exit_code != 0
    assert "must be greater than 0" in result.output


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        (
            "[context]\nrecent_tail_messages = 0\n",
            "context.recent_tail_messages must be greater than 0",
        ),
        ('[context]\nrecent_tail_messages = "abc"\n', "Expected integer value"),
    ],
)
def test_load_config_rejects_invalid_toml_values(
    tmp_path: Path,
    toml: str,
    match: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(toml, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_config(env_file=None, config_file=config_path)


@pytest.mark.parametrize(
    ("env_name", "env_value", "match"),
    [
        (
            "ALPHA_CONTEXT_RECENT_TAIL_MESSAGES",
            "0",
            "context.recent_tail_messages must be greater than 0",
        ),
    ],
)
def test_load_config_rejects_invalid_env_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    env_name: str,
    env_value: str,
    match: str,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv(env_name, env_value)

    with pytest.raises(ValueError, match=match):
        load_config(env_file=None, config_file=config_path)


def test_read_config_value_masks_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["config", "set", "deepseek.api_key", "secret-value"],
        env={"ALPHA_CONFIG_PATH": str(config_path)},
    )
    assert result.exit_code == 0

    value = read_config_value("deepseek.api_key", config_path=config_path, reveal_secret=False)

    assert value == "***"
