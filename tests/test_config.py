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

[llm.context]
tool_truncate_threshold_ratio = 0.55
handover_compress_threshold_ratio = 0.85
minimum_remaining_tokens = 12000
tool_string_truncate_chars = 240
expected_output_reserve_tokens = 2048
safety_margin_tokens = 512

[llm.providers.openai-compatible]
max_context_tokens = 200000

[llm.providers.deepseek]
max_context_tokens = 900000

[tools.bash]
enabled = true
default_workdir = "."
allowed_workdirs = [".", "~/custom-alpha"]
default_timeout_seconds = 30
max_timeout_seconds = 90
max_output_chars = 12000
env_passthrough = ["ALPHA_VISIBLE_ENV"]

[deepseek]
api_key = "deepseek-key"
reasoning_enabled = false
reasoning_effort = "high"

[tavily]
api_key = "tvly-file-key"
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
    assert config.llm_context.tool_truncate_threshold_ratio == 0.55
    assert config.llm_context.handover_compress_threshold_ratio == 0.85
    assert config.llm_context.minimum_remaining_tokens == 12000
    assert config.llm_context.tool_string_truncate_chars == 240
    assert config.llm_context.expected_output_reserve_tokens == 2048
    assert config.llm_context.safety_margin_tokens == 512
    assert config.max_context_tokens_for_provider("openai-compatible") == 200000
    assert config.max_context_tokens_for_provider("openai") == 200000
    assert config.max_context_tokens_for_provider("compatible") == 200000
    assert config.max_context_tokens_for_provider("deepseek") == 900000
    assert config.bash_tool.enabled is True
    assert config.bash_tool.default_workdir == Path(".").resolve()
    assert config.bash_tool.allowed_workdirs == (
        Path(".").resolve(),
        Path("~/custom-alpha").expanduser().resolve(),
    )
    assert config.bash_tool.default_timeout_seconds == 30
    assert config.bash_tool.max_timeout_seconds == 90
    assert config.bash_tool.max_output_chars == 12000
    assert config.bash_tool.env_passthrough == ("ALPHA_VISIBLE_ENV",)
    assert config.deepseek_api_key == "deepseek-key"
    assert config.llm_model == "deepseek-v4-pro"
    assert config.compatible_base_url == "https://compatible.example/v1"
    assert config.compatible_api_key == "compatible-key"
    assert config.deepseek_reasoning_enabled is False
    assert config.deepseek_reasoning_effort == "high"
    assert config.tavily_api_key == "tvly-file-key"
    assert config.llm_debug_logging is True
    assert config.cognition_drive_enabled is False
    assert config.cognition_drive_interval_seconds == 300
    assert config.cognition_drive_goal_cooldown_seconds == 3600
    assert config.cognition_drive_active_goal_limit == 8


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

[tavily]
api_key = "tvly-file-key"

[tools.bash]
enabled = false
default_timeout_seconds = 10
max_timeout_seconds = 20
max_output_chars = 100
allowed_workdirs = ["."]
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    monkeypatch.setenv("ALPHA_LLM_DEBUG_LOGGING", "true")
    monkeypatch.setenv("ALPHA_DEEPSEEK_API_KEY", "from-env")
    monkeypatch.setenv("ALPHA_COMPATIBLE_BASE_URL", "from-env")
    monkeypatch.setenv("ALPHA_COMPATIBLE_API_KEY", "compatible-env-key")
    monkeypatch.setenv("ALPHA_TAVILY_API_KEY", "tvly-env-key")
    monkeypatch.setenv("ALPHA_BASH_TOOL_ENABLED", "true")
    monkeypatch.setenv("ALPHA_BASH_TOOL_DEFAULT_WORKDIR", "~")
    monkeypatch.setenv("ALPHA_BASH_TOOL_ALLOWED_WORKDIRS", ".,~")
    monkeypatch.setenv("ALPHA_BASH_TOOL_DEFAULT_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("ALPHA_BASH_TOOL_MAX_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("ALPHA_BASH_TOOL_MAX_OUTPUT_CHARS", "4096")
    monkeypatch.setenv("ALPHA_BASH_TOOL_ENV_PASSTHROUGH", "ALPHA_VISIBLE_ENV,CI")

    config = load_config(env_file=None, config_file=config_path)

    assert config.llm_provider == "mock"
    assert config.llm_debug_logging is True
    assert config.deepseek_api_key == "from-env"
    assert config.compatible_base_url == "from-env"
    assert config.compatible_api_key == "compatible-env-key"
    assert config.tavily_api_key == "tvly-env-key"
    assert config.bash_tool.enabled is True
    assert config.bash_tool.default_workdir == Path("~").expanduser().resolve()
    assert config.bash_tool.allowed_workdirs == (
        Path(".").resolve(),
        Path("~").expanduser().resolve(),
    )
    assert config.bash_tool.default_timeout_seconds == 45
    assert config.bash_tool.max_timeout_seconds == 120
    assert config.bash_tool.max_output_chars == 4096
    assert config.bash_tool.env_passthrough == ("ALPHA_VISIBLE_ENV", "CI")


def test_load_config_accepts_generic_tavily_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-generic-key")

    config = load_config(env_file=None, config_file=config_path)

    assert config.tavily_api_key == "tvly-generic-key"


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
    assert "llm_context_handover_compress_threshold_ratio" in show_result.output
    assert "llm_provider_max_context_tokens" in show_result.output
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
    set_context = runner.invoke(
        app,
        ["config", "set", "llm.context.expected_output_reserve_tokens", "2048"],
    )
    set_tavily = runner.invoke(app, ["config", "set", "tavily.api_key", "tvly-test"])
    set_bash_enabled = runner.invoke(app, ["config", "set", "tools.bash.enabled", "true"])
    set_bash_workdirs = runner.invoke(
        app,
        ["config", "set", "tools.bash.allowed_workdirs", ".,~/alpha-work"],
    )
    set_provider_limit = runner.invoke(
        app,
        ["config", "set", "llm.providers.deepseek.max_context_tokens", "900000"],
    )
    get_provider = runner.invoke(app, ["config", "get", "llm.provider"])
    get_bash_enabled = runner.invoke(app, ["config", "get", "tools.bash.enabled"])

    assert set_provider.exit_code == 0
    assert set_debug.exit_code == 0
    assert set_context.exit_code == 0
    assert set_tavily.exit_code == 0
    assert set_bash_enabled.exit_code == 0
    assert set_bash_workdirs.exit_code == 0
    assert set_provider_limit.exit_code == 0
    assert get_provider.exit_code == 0
    assert get_bash_enabled.exit_code == 0
    assert "codex" in get_provider.output
    assert "true" in get_bash_enabled.output
    config = load_config(env_file=None, config_file=config_path)
    assert config.llm_provider == "codex"
    assert config.llm_debug_logging is True
    assert config.llm_context.expected_output_reserve_tokens == 2048
    assert config.max_context_tokens_for_provider("deepseek") == 900000
    assert config.tavily_api_key == "tvly-test"
    assert config.bash_tool.enabled is True
    assert config.bash_tool.allowed_workdirs == (
        Path(".").resolve(),
        Path("~/alpha-work").expanduser().resolve(),
    )


def test_config_set_preserves_cognition_consolidation_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "mock"

[cognition.consolidation]
enabled = true
context_foreground_max = 7
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.provider", "codex"])

    assert result.exit_code == 0
    saved = config_path.read_text(encoding="utf-8")
    assert "[cognition.consolidation]" in saved
    assert "context_foreground_max = 7" in saved


def test_config_set_preserves_cognition_drive_section(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "mock"

[cognition.consolidation]
enabled = true

[cognition.drive]
enabled = false
goal_cooldown_seconds = 120
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "cognition.drive.enabled", "true"])

    assert result.exit_code == 0
    saved = config_path.read_text(encoding="utf-8")
    assert "[cognition.consolidation]" in saved
    assert "[cognition.drive]" in saved
    assert "enabled = true" in saved
    assert "goal_cooldown_seconds = 120" in saved


def test_drive_config_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("ALPHA_COGNITION_DRIVE_ENABLED", "true")
    monkeypatch.setenv("ALPHA_COGNITION_DRIVE_INTERVAL_SECONDS", "60")
    monkeypatch.setenv("ALPHA_COGNITION_DRIVE_GOAL_COOLDOWN_SECONDS", "90")
    monkeypatch.setenv("ALPHA_COGNITION_DRIVE_ACTIVE_GOAL_LIMIT", "3")

    config = load_config(env_file=None, config_file=config_path)

    assert config.cognition_drive_enabled is True
    assert config.cognition_drive_interval_seconds == 60
    assert config.cognition_drive_goal_cooldown_seconds == 90
    assert config.cognition_drive_active_goal_limit == 3


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

    result = runner.invoke(
        app,
        ["config", "set", "llm.context.minimum_remaining_tokens", "0"],
    )

    assert result.exit_code != 0
    assert "must be greater than 0" in result.output


def test_config_set_rejects_invalid_bash_timeout_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "tools.bash.max_timeout_seconds", "10"])

    assert result.exit_code != 0
    assert "tools.bash.default_timeout_seconds must be at most" in result.output


def test_config_set_rejects_bash_default_workdir_outside_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["config", "set", "tools.bash.default_workdir", "~/outside-alpha-work"],
    )

    assert result.exit_code != 0
    assert "tools.bash.default_workdir must be within" in result.output


@pytest.mark.parametrize(
    ("toml", "match"),
    [
        (
            "[llm.context]\nminimum_remaining_tokens = 0\n",
            "llm.context.minimum_remaining_tokens must be greater than 0",
        ),
        ('[llm.context]\nminimum_remaining_tokens = "abc"\n', "Expected integer value"),
        (
            "[llm.context]\nhandover_compress_threshold_ratio = 2.0\n",
            "llm.context.handover_compress_threshold_ratio must be greater than 0 and at most 1",
        ),
        (
            "[tools.bash]\ndefault_timeout_seconds = 20\nmax_timeout_seconds = 10\n",
            "tools.bash.default_timeout_seconds must be at most tools.bash.max_timeout_seconds",
        ),
        (
            "[tools.bash]\nmax_output_chars = 0\n",
            "tools.bash.max_output_chars must be greater than 0",
        ),
        (
            '[tools.bash]\ndefault_workdir = "~/outside-alpha-work"\nallowed_workdirs = ["."]\n',
            "tools.bash.default_workdir must be within tools.bash.allowed_workdirs",
        ),
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
            "ALPHA_LLM_CONTEXT_MINIMUM_REMAINING_TOKENS",
            "0",
            "llm.context.minimum_remaining_tokens must be greater than 0",
        ),
        (
            "ALPHA_BASH_TOOL_DEFAULT_TIMEOUT_SECONDS",
            "0",
            "tools.bash.default_timeout_seconds must be greater than 0",
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


def test_read_config_value_masks_tavily_secret(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    write_default_config(config_path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["config", "set", "tavily.api_key", "secret-value"],
        env={"ALPHA_CONFIG_PATH": str(config_path)},
    )
    assert result.exit_code == 0

    value = read_config_value("tavily.api_key", config_path=config_path, reveal_secret=False)

    assert value == "***"
