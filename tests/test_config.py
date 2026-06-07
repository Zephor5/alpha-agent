from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.config import FileToolConfig, load_config, read_config_value, write_default_config
from alpha_agent.tools.files.config import max_glob_results, max_search_results


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

[tools.files]
enabled = true
allowed_roots = [".", "~/custom-alpha/files"]
patch_enabled = true
write_roots = ["~/custom-alpha/write"]
max_read_chars = 111
max_file_bytes = 222
max_search_results = 3
max_glob_results = 4
max_read_lines = 77
create_parent_dirs_enabled = true
max_output_chars = 555

[cognition.background]
enabled = false
startup_delay_seconds = 2
interval_seconds = 33
tick_timeout_seconds = 9

[cognition.background.intake]
batch_size = 5
min_sources = 2

[cognition.background.extraction]
batch_size = 6
min_sources = 3

[cognition.background.consolidation]
batch_size = 7
min_drafts = 4

[cognition.background.conflict]
batch_size = 8
min_conflicts = 5

[cognition.background.summary]
batch_size = 9
initial_min_beliefs = 10
changed_source_min = 11
invalidated_source_min = 12

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
    assert config.file_tool.enabled is True
    assert config.file_tool.allowed_roots == (
        Path(".").resolve(),
        Path("~/custom-alpha/files").expanduser().resolve(),
    )
    assert config.file_tool.patch_enabled is True
    assert config.file_tool.write_roots == (Path("~/custom-alpha/write").expanduser().resolve(),)
    assert config.file_tool.max_read_chars == 111
    assert config.file_tool.max_file_bytes == 222
    assert config.file_tool.max_search_results == 3
    assert config.file_tool.max_glob_results == 4
    assert config.file_tool.max_read_lines == 77
    assert config.file_tool.create_parent_dirs_enabled is True
    assert config.file_tool.max_output_chars == 555
    assert config.deepseek_api_key == "deepseek-key"
    assert config.llm_model == "deepseek-v4-pro"
    assert config.compatible_base_url == "https://compatible.example/v1"
    assert config.compatible_api_key == "compatible-key"
    assert config.deepseek_reasoning_enabled is False
    assert config.deepseek_reasoning_effort == "high"
    assert config.tavily_api_key == "tvly-file-key"
    assert config.llm_debug_logging is True
    assert config.cognition_background.enabled is False
    assert config.cognition_background.startup_delay_seconds == 2
    assert config.cognition_background.interval_seconds == 33
    assert config.cognition_background.tick_timeout_seconds == 9
    assert config.cognition_background.intake.batch_size == 5
    assert config.cognition_background.intake.min_sources == 2
    assert config.cognition_background.extraction.batch_size == 6
    assert config.cognition_background.extraction.min_sources == 3
    assert config.cognition_background.consolidation.batch_size == 7
    assert config.cognition_background.consolidation.min_drafts == 4
    assert config.cognition_background.conflict.batch_size == 8
    assert config.cognition_background.conflict.min_conflicts == 5
    assert config.cognition_background.summary.batch_size == 9
    assert config.cognition_background.summary.initial_min_beliefs == 10
    assert config.cognition_background.summary.changed_source_min == 11
    assert config.cognition_background.summary.invalidated_source_min == 12
    assert config.cognition_drive_enabled is False
    assert config.cognition_drive_interval_seconds == 300
    assert config.cognition_drive_goal_cooldown_seconds == 3600
    assert config.cognition_drive_active_goal_limit == 8


def test_background_tick_timeout_default_is_sixty_seconds(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(env_file=None, config_file=config_path)

    assert config.cognition_background.tick_timeout_seconds == 60


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
    monkeypatch.setenv("ALPHA_FILE_TOOL_ENABLED", "true")
    monkeypatch.setenv("ALPHA_FILE_TOOL_ALLOWED_ROOTS", ".,~/alpha-files")
    monkeypatch.setenv("ALPHA_FILE_TOOL_PATCH_ENABLED", "true")
    monkeypatch.setenv("ALPHA_FILE_TOOL_WRITE_ROOTS", "~/alpha-write")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_READ_CHARS", "123")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_FILE_BYTES", "456")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_SEARCH_RESULTS", "7")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_GLOB_RESULTS", "8")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_READ_LINES", "9")
    monkeypatch.setenv("ALPHA_FILE_TOOL_CREATE_PARENT_DIRS_ENABLED", "true")
    monkeypatch.setenv("ALPHA_FILE_TOOL_MAX_OUTPUT_CHARS", "900")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_ENABLED", "false")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_STARTUP_DELAY_SECONDS", "1")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_INTERVAL_SECONDS", "2")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_TICK_TIMEOUT_SECONDS", "3")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_INTAKE_BATCH_SIZE", "4")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_INTAKE_MIN_SOURCES", "5")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_EXTRACTION_BATCH_SIZE", "6")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_EXTRACTION_MIN_SOURCES", "7")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_CONSOLIDATION_BATCH_SIZE", "8")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_CONSOLIDATION_MIN_DRAFTS", "9")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_CONFLICT_BATCH_SIZE", "10")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_CONFLICT_MIN_CONFLICTS", "11")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_SUMMARY_BATCH_SIZE", "12")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_SUMMARY_INITIAL_MIN_BELIEFS", "13")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_SUMMARY_CHANGED_SOURCE_MIN", "14")
    monkeypatch.setenv("ALPHA_COGNITION_BACKGROUND_SUMMARY_INVALIDATED_SOURCE_MIN", "15")

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
    assert config.file_tool.enabled is True
    assert config.file_tool.allowed_roots == (
        Path(".").resolve(),
        Path("~/alpha-files").expanduser().resolve(),
    )
    assert config.file_tool.patch_enabled is True
    assert config.file_tool.write_roots == (Path("~/alpha-write").expanduser().resolve(),)
    assert config.file_tool.max_read_chars == 123
    assert config.file_tool.max_file_bytes == 456
    assert config.file_tool.max_search_results == 7
    assert config.file_tool.max_glob_results == 8
    assert config.file_tool.max_read_lines == 9
    assert config.file_tool.create_parent_dirs_enabled is True
    assert config.file_tool.max_output_chars == 900
    assert config.cognition_background.enabled is False
    assert config.cognition_background.startup_delay_seconds == 1
    assert config.cognition_background.interval_seconds == 2
    assert config.cognition_background.tick_timeout_seconds == 3
    assert config.cognition_background.intake.batch_size == 4
    assert config.cognition_background.intake.min_sources == 5
    assert config.cognition_background.extraction.batch_size == 6
    assert config.cognition_background.extraction.min_sources == 7
    assert config.cognition_background.consolidation.batch_size == 8
    assert config.cognition_background.consolidation.min_drafts == 9
    assert config.cognition_background.conflict.batch_size == 10
    assert config.cognition_background.conflict.min_conflicts == 11
    assert config.cognition_background.summary.batch_size == 12
    assert config.cognition_background.summary.initial_min_beliefs == 13
    assert config.cognition_background.summary.changed_source_min == 14
    assert config.cognition_background.summary.invalidated_source_min == 15


def test_load_config_accepts_generic_tavily_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-generic-key")

    config = load_config(env_file=None, config_file=config_path)

    assert config.tavily_api_key == "tvly-generic-key"


def test_load_config_ignores_legacy_codex_api_key_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[codex]
access_token = "from-file"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CODEX_API_KEY", "legacy-env-key")

    config = load_config(env_file=None, config_file=config_path)

    assert config.codex_access_token == "from-file"


def test_load_config_reads_codex_access_token_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[codex]
access_token = "from-file"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CODEX_ACCESS_TOKEN", "from-env")

    config = load_config(env_file=None, config_file=config_path)

    assert config.codex_access_token == "from-env"


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
    set_files_enabled = runner.invoke(app, ["config", "set", "tools.files.enabled", "true"])
    set_files_roots = runner.invoke(
        app,
        ["config", "set", "tools.files.allowed_roots", ".,~/alpha-files"],
    )
    set_files_write_roots = runner.invoke(
        app,
        ["config", "set", "tools.files.write_roots", "~/alpha-write"],
    )
    set_files_patch_enabled = runner.invoke(
        app,
        ["config", "set", "tools.files.patch_enabled", "true"],
    )
    set_files_search_limit = runner.invoke(
        app,
        ["config", "set", "tools.files.max_search_results", "17"],
    )
    set_files_glob_limit = runner.invoke(
        app,
        ["config", "set", "tools.files.max_glob_results", "18"],
    )
    set_files_read_lines = runner.invoke(
        app,
        ["config", "set", "tools.files.max_read_lines", "19"],
    )
    set_files_create_parent_dirs = runner.invoke(
        app,
        ["config", "set", "tools.files.create_parent_dirs_enabled", "true"],
    )
    set_provider_limit = runner.invoke(
        app,
        ["config", "set", "llm.providers.deepseek.max_context_tokens", "900000"],
    )
    get_provider = runner.invoke(app, ["config", "get", "llm.provider"])
    get_bash_enabled = runner.invoke(app, ["config", "get", "tools.bash.enabled"])
    get_files_enabled = runner.invoke(app, ["config", "get", "tools.files.enabled"])
    get_files_patch_enabled = runner.invoke(
        app,
        ["config", "get", "tools.files.patch_enabled"],
    )

    assert set_provider.exit_code == 0
    assert set_debug.exit_code == 0
    assert set_context.exit_code == 0
    assert set_tavily.exit_code == 0
    assert set_bash_enabled.exit_code == 0
    assert set_bash_workdirs.exit_code == 0
    assert set_files_enabled.exit_code == 0
    assert set_files_roots.exit_code == 0
    assert set_files_patch_enabled.exit_code == 0
    assert set_files_write_roots.exit_code == 0
    assert set_files_search_limit.exit_code == 0
    assert set_files_glob_limit.exit_code == 0
    assert set_files_read_lines.exit_code == 0
    assert set_files_create_parent_dirs.exit_code == 0
    assert set_provider_limit.exit_code == 0
    assert get_provider.exit_code == 0
    assert get_bash_enabled.exit_code == 0
    assert get_files_enabled.exit_code == 0
    assert get_files_patch_enabled.exit_code == 0
    assert "codex" in get_provider.output
    assert "true" in get_bash_enabled.output
    assert "true" in get_files_enabled.output
    assert "true" in get_files_patch_enabled.output
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
    assert config.file_tool.enabled is True
    assert config.file_tool.allowed_roots == (
        Path(".").resolve(),
        Path("~/alpha-files").expanduser().resolve(),
    )
    assert config.file_tool.patch_enabled is True
    assert config.file_tool.write_roots == (Path("~/alpha-write").expanduser().resolve(),)
    assert config.file_tool.max_search_results == 17
    assert config.file_tool.max_glob_results == 18
    assert config.file_tool.max_read_lines == 19
    assert config.file_tool.create_parent_dirs_enabled is True


def test_file_patch_config_defaults_to_disabled(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(env_file=None, config_file=config_path)

    assert config.file_tool.patch_enabled is False
    assert config.file_tool.write_roots == ()


def test_file_tool_limit_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("", encoding="utf-8")

    config = load_config(env_file=None, config_file=config_path)

    assert config.file_tool.max_search_results == 100
    assert config.file_tool.max_glob_results == 500
    assert config.file_tool.max_read_lines == 200
    assert config.file_tool.create_parent_dirs_enabled is False


def test_legacy_file_tool_limit_toml_keys_do_not_override_new_limits(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[tools.files]
max_search_matches = 3
max_list_entries = 4
""",
        encoding="utf-8",
    )

    config = load_config(env_file=None, config_file=config_path)

    assert config.file_tool.max_search_results == 100
    assert config.file_tool.max_glob_results == 500


def test_file_tool_limit_helpers_ignore_legacy_limit_attrs() -> None:
    legacy_only = cast(
        FileToolConfig,
        SimpleNamespace(max_search_matches=3, max_list_entries=4),
    )

    assert max_search_results(legacy_only) == 100
    assert max_glob_results(legacy_only) == 500


def test_file_patch_config_allows_empty_write_roots_when_patch_enabled(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[tools.files]
enabled = false
patch_enabled = true
write_roots = []
""",
        encoding="utf-8",
    )

    config = load_config(env_file=None, config_file=config_path)

    assert config.file_tool.enabled is False
    assert config.file_tool.patch_enabled is True
    assert config.file_tool.write_roots == ()


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
interval_seconds = 7
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.provider", "codex"])

    assert result.exit_code == 0
    saved = config_path.read_text(encoding="utf-8")
    assert "[cognition.consolidation]" in saved
    assert "interval_seconds = 7" in saved


def test_config_set_preserves_cognition_background_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[llm]
provider = "mock"

[cognition.background]
enabled = false
interval_seconds = 7

[cognition.background.extraction]
batch_size = 3
min_sources = 2

[cognition.consolidation]
enabled = true
interval_seconds = 11
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", "llm.provider", "codex"])

    assert result.exit_code == 0
    saved = config_path.read_text(encoding="utf-8")
    assert "[cognition.background]" in saved
    assert "[cognition.background.extraction]" in saved
    assert "[cognition.consolidation]" in saved
    assert "interval_seconds = 7" in saved
    assert "batch_size = 3" in saved
    assert "min_sources = 2" in saved
    assert "interval_seconds = 11" in saved


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


@pytest.mark.parametrize(
    "key",
    [
        "tools.files.max_search_matches",
        "tools.files.max_list_entries",
    ],
)
def test_legacy_file_tool_limit_keys_are_not_configurable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
) -> None:
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    runner = CliRunner()

    result = runner.invoke(app, ["config", "set", key, "1"])

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
            "[tools.files]\nmax_read_lines = 0\n",
            "tools.files.max_read_lines must be greater than 0",
        ),
        (
            "[tools.files]\nmax_search_results = 0\n",
            "tools.files.max_search_results must be greater than 0",
        ),
        (
            "[tools.files]\nmax_glob_results = 0\n",
            "tools.files.max_glob_results must be greater than 0",
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
        (
            "ALPHA_FILE_TOOL_MAX_READ_LINES",
            "0",
            "tools.files.max_read_lines must be greater than 0",
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
