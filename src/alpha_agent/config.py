"""Configuration loading for Alpha Agent."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

DEFAULT_CONFIG_TOML = """# Alpha Agent local configuration.
# Environment variables still override these values for one-off runs or deploys.

[runtime]
db_path = "~/.alpha-agent/alpha.db"
log_dir = "~/.alpha-agent/logs"
gateway_status_path = "~/.alpha-agent/gateway-status.json"
daemon_socket_path = "~/.alpha-agent/daemon.sock"
daemon_status_path = "~/.alpha-agent/daemon-status.json"

[llm]
provider = "mock"
model = ""
debug_logging = false

[compatible]
base_url = "https://api.openai.com/v1"
api_key = ""

[llm.context]
# Active runtime context maintenance settings.
# Tool payload truncation runs before LLM handover compression.
tool_truncate_threshold_ratio = 0.60
handover_compress_threshold_ratio = 0.90
minimum_remaining_tokens = 10000
tool_string_truncate_chars = 300
expected_output_reserve_tokens = 4096
safety_margin_tokens = 1024

[llm.providers.openai-compatible]
max_context_tokens = 258400

[llm.providers.deepseek]
max_context_tokens = 1000000

[tools.bash]
enabled = false
default_workdir = "."
allowed_workdirs = ["."]
default_timeout_seconds = 120
max_timeout_seconds = 600
max_output_chars = 30000
env_passthrough = []

[tools.files]
enabled = true
allowed_roots = ["."]
patch_enabled = false
write_roots = []
max_read_chars = 20000
max_file_bytes = 1000000
max_search_results = 100
max_glob_results = 500
max_read_lines = 200
create_parent_dirs_enabled = false
max_output_chars = 30000

[cognition.consolidation]
enabled = true
interval_seconds = 300

[cognition.background]
enabled = true
startup_delay_seconds = 5
interval_seconds = 300
tick_timeout_seconds = 60

[cognition.background.intake]
batch_size = 64
min_sources = 1

[cognition.background.extraction]
batch_size = 12
min_sources = 1

[cognition.background.consolidation]
batch_size = 12
min_drafts = 1

[cognition.background.conflict]
batch_size = 4
min_conflicts = 1

[cognition.background.summary]
batch_size = 4
initial_min_beliefs = 12
changed_source_min = 6
invalidated_source_min = 1

[cognition.drive]
enabled = false
interval_seconds = 300
goal_cooldown_seconds = 3600
active_goal_limit = 8

[deepseek]
api_key = ""
reasoning_enabled = true
reasoning_effort = ""

[codex]
access_token = ""

[tavily]
api_key = ""
"""

CONFIG_KEY_TYPES: dict[str, type] = {
    "runtime.db_path": str,
    "runtime.log_dir": str,
    "runtime.gateway_status_path": str,
    "runtime.daemon_socket_path": str,
    "runtime.daemon_status_path": str,
    "llm.provider": str,
    "llm.model": str,
    "llm.debug_logging": bool,
    "llm.context.tool_truncate_threshold_ratio": float,
    "llm.context.handover_compress_threshold_ratio": float,
    "llm.context.minimum_remaining_tokens": int,
    "llm.context.tool_string_truncate_chars": int,
    "llm.context.expected_output_reserve_tokens": int,
    "llm.context.safety_margin_tokens": int,
    "llm.providers.openai-compatible.max_context_tokens": int,
    "llm.providers.deepseek.max_context_tokens": int,
    "compatible.base_url": str,
    "compatible.api_key": str,
    "tools.bash.enabled": bool,
    "tools.bash.default_workdir": str,
    "tools.bash.allowed_workdirs": list,
    "tools.bash.default_timeout_seconds": int,
    "tools.bash.max_timeout_seconds": int,
    "tools.bash.max_output_chars": int,
    "tools.bash.env_passthrough": list,
    "tools.files.enabled": bool,
    "tools.files.allowed_roots": list,
    "tools.files.patch_enabled": bool,
    "tools.files.write_roots": list,
    "tools.files.max_read_chars": int,
    "tools.files.max_file_bytes": int,
    "tools.files.max_search_results": int,
    "tools.files.max_glob_results": int,
    "tools.files.max_read_lines": int,
    "tools.files.create_parent_dirs_enabled": bool,
    "tools.files.max_output_chars": int,
    "cognition.background.enabled": bool,
    "cognition.background.startup_delay_seconds": int,
    "cognition.background.interval_seconds": int,
    "cognition.background.tick_timeout_seconds": int,
    "cognition.background.intake.batch_size": int,
    "cognition.background.intake.min_sources": int,
    "cognition.background.extraction.batch_size": int,
    "cognition.background.extraction.min_sources": int,
    "cognition.background.consolidation.batch_size": int,
    "cognition.background.consolidation.min_drafts": int,
    "cognition.background.conflict.batch_size": int,
    "cognition.background.conflict.min_conflicts": int,
    "cognition.background.summary.batch_size": int,
    "cognition.background.summary.initial_min_beliefs": int,
    "cognition.background.summary.changed_source_min": int,
    "cognition.background.summary.invalidated_source_min": int,
    "cognition.drive.enabled": bool,
    "cognition.drive.interval_seconds": int,
    "cognition.drive.goal_cooldown_seconds": int,
    "cognition.drive.active_goal_limit": int,
    "deepseek.api_key": str,
    "deepseek.reasoning_enabled": bool,
    "deepseek.reasoning_effort": str,
    "codex.access_token": str,
    "tavily.api_key": str,
}

CONFIG_KEY_ALLOWED_VALUES: dict[str, set[str]] = {
    "llm.provider": {
        "mock",
        "openai-compatible",
        "openai",
        "compatible",
        "deepseek",
        "codex",
        "openai-codex",
        "openai_codex",
    },
    "deepseek.reasoning_effort": {"", "low", "medium", "high", "max", "xhigh"},
}

POSITIVE_INT_CONFIG_KEYS = {
    "cognition.background.interval_seconds",
    "cognition.background.tick_timeout_seconds",
    "cognition.background.intake.batch_size",
    "cognition.background.intake.min_sources",
    "cognition.background.extraction.batch_size",
    "cognition.background.extraction.min_sources",
    "cognition.background.consolidation.batch_size",
    "cognition.background.consolidation.min_drafts",
    "cognition.background.conflict.batch_size",
    "cognition.background.conflict.min_conflicts",
    "cognition.background.summary.batch_size",
    "cognition.background.summary.initial_min_beliefs",
    "cognition.background.summary.changed_source_min",
    "cognition.background.summary.invalidated_source_min",
    "cognition.drive.active_goal_limit",
    "cognition.drive.goal_cooldown_seconds",
    "cognition.drive.interval_seconds",
    "llm.context.minimum_remaining_tokens",
    "llm.context.tool_string_truncate_chars",
    "llm.context.expected_output_reserve_tokens",
    "llm.context.safety_margin_tokens",
    "llm.providers.openai-compatible.max_context_tokens",
    "llm.providers.deepseek.max_context_tokens",
    "tools.bash.default_timeout_seconds",
    "tools.bash.max_timeout_seconds",
    "tools.bash.max_output_chars",
    "tools.files.max_read_chars",
    "tools.files.max_file_bytes",
    "tools.files.max_search_results",
    "tools.files.max_glob_results",
    "tools.files.max_read_lines",
    "tools.files.max_output_chars",
}

NON_NEGATIVE_INT_CONFIG_KEYS = {
    "cognition.background.startup_delay_seconds",
}

RATIO_CONFIG_KEYS = {
    "llm.context.tool_truncate_threshold_ratio",
    "llm.context.handover_compress_threshold_ratio",
}

SECRET_CONFIG_KEYS = {
    "compatible.api_key",
    "deepseek.api_key",
    "codex.access_token",
    "tavily.api_key",
}


DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS = {
    "mock": 258400,
    "openai-compatible": 258400,
    "deepseek": 1000000,
    "codex": 258400,
}


@dataclass(frozen=True)
class LLMContextConfig:
    """Context budgeting thresholds and reserves for LLM calls."""

    tool_truncate_threshold_ratio: float = 0.60
    handover_compress_threshold_ratio: float = 0.90
    minimum_remaining_tokens: int = 10000
    tool_string_truncate_chars: int = 300
    expected_output_reserve_tokens: int = 4096
    safety_margin_tokens: int = 1024


@dataclass(frozen=True)
class BashToolConfig:
    """Configuration for the opt-in local bash tool."""

    enabled: bool = False
    default_workdir: Path = Path(".")
    allowed_workdirs: tuple[Path, ...] = (Path("."),)
    default_timeout_seconds: int = 120
    max_timeout_seconds: int = 600
    max_output_chars: int = 30000
    env_passthrough: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileToolConfig:
    """Configuration for local file tools."""

    enabled: bool = True
    allowed_roots: tuple[Path, ...] = (Path("."),)
    patch_enabled: bool = False
    write_roots: tuple[Path, ...] = ()
    max_read_chars: int = 20000
    max_file_bytes: int = 1000000
    max_search_results: int = 100
    max_glob_results: int = 500
    max_read_lines: int = 200
    create_parent_dirs_enabled: bool = False
    max_output_chars: int = 30000


@dataclass(frozen=True)
class BackgroundIntakeConfig:
    """Source intake gating and batch size for daemon background cognition."""

    batch_size: int = 64
    min_sources: int = 1


@dataclass(frozen=True)
class BackgroundExtractionConfig:
    """Memory extraction gating and batch size for daemon background cognition."""

    batch_size: int = 12
    min_sources: int = 1


@dataclass(frozen=True)
class BackgroundConsolidationConfig:
    """Memory consolidation gating and batch size for daemon background cognition."""

    batch_size: int = 12
    min_drafts: int = 1


@dataclass(frozen=True)
class BackgroundConflictConfig:
    """Conflict review gating and batch size for daemon background cognition."""

    batch_size: int = 4
    min_conflicts: int = 1


@dataclass(frozen=True)
class BackgroundSummaryConfig:
    """Summary gate placeholders for later background summary phases."""

    batch_size: int = 4
    initial_min_beliefs: int = 12
    changed_source_min: int = 6
    invalidated_source_min: int = 1


@dataclass(frozen=True)
class CognitionBackgroundConfig:
    """Daemon-owned automatic background cognition settings."""

    enabled: bool = True
    startup_delay_seconds: int = 5
    interval_seconds: int = 300
    tick_timeout_seconds: int = 60
    intake: BackgroundIntakeConfig = field(default_factory=BackgroundIntakeConfig)
    extraction: BackgroundExtractionConfig = field(default_factory=BackgroundExtractionConfig)
    consolidation: BackgroundConsolidationConfig = field(
        default_factory=BackgroundConsolidationConfig
    )
    conflict: BackgroundConflictConfig = field(default_factory=BackgroundConflictConfig)
    summary: BackgroundSummaryConfig = field(default_factory=BackgroundSummaryConfig)


@dataclass(frozen=True)
class AlphaConfig:
    """Runtime settings loaded from environment variables and defaults."""

    db_path: Path
    log_dir: Path
    gateway_status_path: Path
    daemon_socket_path: Path = Path("~/.alpha-agent/daemon.sock").expanduser()
    daemon_status_path: Path = Path("~/.alpha-agent/daemon-status.json").expanduser()
    llm_provider: str = "mock"
    llm_model: str = ""
    llm_debug_logging: bool = False
    llm_context: LLMContextConfig = field(default_factory=LLMContextConfig)
    bash_tool: BashToolConfig = field(default_factory=BashToolConfig)
    file_tool: FileToolConfig = field(default_factory=FileToolConfig)
    llm_provider_max_context_tokens: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS)
    )
    compatible_base_url: str | None = None
    compatible_api_key: str | None = None
    cognition_consolidation_enabled: bool = True
    cognition_consolidation_interval_seconds: int = 300
    cognition_background: CognitionBackgroundConfig = field(
        default_factory=CognitionBackgroundConfig
    )
    cognition_drive_enabled: bool = False
    cognition_drive_interval_seconds: int = 300
    cognition_drive_goal_cooldown_seconds: int = 3600
    cognition_drive_active_goal_limit: int = 8
    deepseek_api_key: str | None = None
    deepseek_reasoning_enabled: bool = True
    deepseek_reasoning_effort: str | None = None
    codex_access_token: str | None = None
    tavily_api_key: str | None = None

    def max_context_tokens_for_provider(self, provider_name: str | None = None) -> int:
        """Return the configured max context tokens for a normalized provider name."""

        provider = _normalize_provider_context_key(provider_name or self.llm_provider)
        return self.llm_provider_max_context_tokens.get(
            provider,
            DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS["openai-compatible"],
        )


def default_config_path() -> Path:
    """Return the default user config path."""

    return Path(os.getenv("ALPHA_CONFIG_PATH", "~/.alpha-agent/config.toml")).expanduser()


def write_default_config(path: str | Path | None = None, *, overwrite: bool = False) -> bool:
    """Write a default TOML config file."""

    config_path = Path(path).expanduser() if path is not None else default_config_path()
    if config_path.exists() and not overwrite:
        return False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
    return True


def set_config_value(
    dotted_key: str,
    raw_value: str,
    *,
    config_path: str | Path | None = None,
) -> Any:
    """Set one supported config value in the TOML config file and return parsed value."""

    key = _normalize_config_key(dotted_key)
    parsed_value = _parse_config_value(raw_value, CONFIG_KEY_TYPES[key])
    parsed_value = _validate_config_value(key, parsed_value)
    path = Path(config_path).expanduser() if config_path is not None else default_config_path()
    write_default_config(path)
    config_data = _load_toml_config(path)
    _set_nested_value(config_data, key, parsed_value)
    _validate_config_data(config_data)
    _write_toml_config(path, config_data)
    return parsed_value


def read_config_value(
    dotted_key: str,
    *,
    config_path: str | Path | None = None,
    reveal_secret: bool = False,
) -> str:
    """Read one supported config value from the TOML config file."""

    key = _normalize_config_key(dotted_key)
    path = Path(config_path).expanduser() if config_path is not None else default_config_path()
    write_default_config(path)
    config_data = _load_toml_config(path)
    value = _nested_value(config_data, key, "")
    if key in SECRET_CONFIG_KEYS and not reveal_secret:
        return "***" if value else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def load_config(
    env_file: str | Path | None = ".env",
    config_file: str | Path | None = None,
) -> AlphaConfig:
    """Load configuration from TOML, optional .env, and environment variables."""

    if env_file:
        load_dotenv(env_file, override=False)

    config_data = _load_toml_config(config_file)
    llm = _section(config_data, "llm")
    llm_context = llm.get("context")
    llm_context = llm_context if isinstance(llm_context, dict) else {}
    llm_providers = llm.get("providers")
    llm_providers = llm_providers if isinstance(llm_providers, dict) else {}
    tools = _section(config_data, "tools")
    bash_tool = tools.get("bash")
    bash_tool = bash_tool if isinstance(bash_tool, dict) else {}
    file_tool = tools.get("files")
    file_tool = file_tool if isinstance(file_tool, dict) else {}
    cognition = _section(config_data, "cognition")
    consolidation = cognition.get("consolidation")
    consolidation = consolidation if isinstance(consolidation, dict) else {}
    background = cognition.get("background")
    background = background if isinstance(background, dict) else {}
    drive = cognition.get("drive")
    drive = drive if isinstance(drive, dict) else {}
    deepseek = _section(config_data, "deepseek")
    tavily = _section(config_data, "tavily")

    config = AlphaConfig(
        db_path=Path(
            _env_or_config(
                "ALPHA_DB_PATH",
                config_data,
                "runtime",
                "db_path",
                "~/.alpha-agent/alpha.db",
            )
            or "~/.alpha-agent/alpha.db"
        ).expanduser(),
        log_dir=Path(
            _env_or_config(
                "ALPHA_LOG_DIR",
                config_data,
                "runtime",
                "log_dir",
                "~/.alpha-agent/logs",
            )
            or "~/.alpha-agent/logs"
        ).expanduser(),
        gateway_status_path=Path(
            _env_or_config(
                "ALPHA_GATEWAY_STATUS_PATH",
                config_data,
                "runtime",
                "gateway_status_path",
                "~/.alpha-agent/gateway-status.json",
            )
            or "~/.alpha-agent/gateway-status.json"
        ).expanduser(),
        daemon_socket_path=Path(
            _env_or_config(
                "ALPHA_DAEMON_SOCKET_PATH",
                config_data,
                "runtime",
                "daemon_socket_path",
                "~/.alpha-agent/daemon.sock",
            )
            or "~/.alpha-agent/daemon.sock"
        ).expanduser(),
        daemon_status_path=Path(
            _env_or_config(
                "ALPHA_DAEMON_STATUS_PATH",
                config_data,
                "runtime",
                "daemon_status_path",
                "~/.alpha-agent/daemon-status.json",
            )
            or "~/.alpha-agent/daemon-status.json"
        ).expanduser(),
        llm_provider=(
            _env_or_config("ALPHA_LLM_PROVIDER", config_data, "llm", "provider", "mock")
            or "mock"
        ).strip().lower(),
        llm_model=_env_or_config("ALPHA_LLM_MODEL", config_data, "llm", "model", "") or "",
        llm_debug_logging=_bool_env(
            "ALPHA_LLM_DEBUG_LOGGING",
            _bool_value(llm.get("debug_logging"), False),
        ),
        llm_context=LLMContextConfig(
            tool_truncate_threshold_ratio=_float_env(
                "ALPHA_LLM_CONTEXT_TOOL_TRUNCATE_THRESHOLD_RATIO",
                _float_value(llm_context.get("tool_truncate_threshold_ratio"), 0.60),
            ),
            handover_compress_threshold_ratio=_float_env(
                "ALPHA_LLM_CONTEXT_HANDOVER_COMPRESS_THRESHOLD_RATIO",
                _float_value(llm_context.get("handover_compress_threshold_ratio"), 0.90),
            ),
            minimum_remaining_tokens=_int_env(
                "ALPHA_LLM_CONTEXT_MINIMUM_REMAINING_TOKENS",
                _int_value(llm_context.get("minimum_remaining_tokens"), 10000),
            ),
            tool_string_truncate_chars=_int_env(
                "ALPHA_LLM_CONTEXT_TOOL_STRING_TRUNCATE_CHARS",
                _int_value(llm_context.get("tool_string_truncate_chars"), 300),
            ),
            expected_output_reserve_tokens=_int_env(
                "ALPHA_LLM_CONTEXT_EXPECTED_OUTPUT_RESERVE_TOKENS",
                _int_value(llm_context.get("expected_output_reserve_tokens"), 4096),
            ),
            safety_margin_tokens=_int_env(
                "ALPHA_LLM_CONTEXT_SAFETY_MARGIN_TOKENS",
                _int_value(llm_context.get("safety_margin_tokens"), 1024),
            ),
        ),
        bash_tool=_bash_tool_config(bash_tool),
        file_tool=_file_tool_config(file_tool),
        llm_provider_max_context_tokens=_provider_max_context_tokens(llm_providers),
        compatible_base_url=_env_or_config(
            "ALPHA_COMPATIBLE_BASE_URL",
            config_data,
            "compatible",
            "base_url",
        ),
        compatible_api_key=_env_or_config(
            "ALPHA_COMPATIBLE_API_KEY",
            config_data,
            "compatible",
            "api_key",
        ),
        cognition_consolidation_enabled=_bool_env(
            "ALPHA_COGNITION_CONSOLIDATION_ENABLED",
            _bool_value(consolidation.get("enabled"), True),
        ),
        cognition_consolidation_interval_seconds=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_INTERVAL_SECONDS",
            _int_value(consolidation.get("interval_seconds"), 300),
        ),
        cognition_background=_background_config(background),
        cognition_drive_enabled=_bool_env(
            "ALPHA_COGNITION_DRIVE_ENABLED",
            _bool_value(drive.get("enabled"), False),
        ),
        cognition_drive_interval_seconds=_int_env(
            "ALPHA_COGNITION_DRIVE_INTERVAL_SECONDS",
            _int_value(drive.get("interval_seconds"), 300),
        ),
        cognition_drive_goal_cooldown_seconds=_int_env(
            "ALPHA_COGNITION_DRIVE_GOAL_COOLDOWN_SECONDS",
            _int_value(drive.get("goal_cooldown_seconds"), 3600),
        ),
        cognition_drive_active_goal_limit=_int_env(
            "ALPHA_COGNITION_DRIVE_ACTIVE_GOAL_LIMIT",
            _int_value(drive.get("active_goal_limit"), 8),
        ),
        deepseek_api_key=_env_or_config(
            "ALPHA_DEEPSEEK_API_KEY",
            config_data,
            "deepseek",
            "api_key",
        ),
        deepseek_reasoning_enabled=_bool_env(
            "ALPHA_DEEPSEEK_REASONING_ENABLED",
            _bool_value(deepseek.get("reasoning_enabled"), True),
        ),
        deepseek_reasoning_effort=_env_or_config(
            "ALPHA_DEEPSEEK_REASONING_EFFORT",
            config_data,
            "deepseek",
            "reasoning_effort",
        ),
        codex_access_token=(
            os.getenv("ALPHA_CODEX_ACCESS_TOKEN")
            or _string_setting(config_data, "codex", "access_token")
        ),
        tavily_api_key=(
            os.getenv("ALPHA_TAVILY_API_KEY")
            or os.getenv("TAVILY_API_KEY")
            or str(tavily.get("api_key") or "")
        ),
    )
    return _validate_loaded_config(config)


def _normalize_config_key(dotted_key: str) -> str:
    key = dotted_key.strip().lower()
    if key not in CONFIG_KEY_TYPES:
        raise ValueError(f"Unsupported config key: {dotted_key}")
    return key


def _parse_config_value(raw_value: str, expected_type: type) -> Any:
    if expected_type is bool:
        value = raw_value.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected boolean value, got: {raw_value}")
    if expected_type is int:
        try:
            return int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Expected integer value, got: {raw_value}") from exc
    if expected_type is float:
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Expected float value, got: {raw_value}") from exc
    if expected_type is list:
        return _list_value(raw_value, ())
    return raw_value


def _validate_config_value(key: str, value: Any) -> Any:
    if key in CONFIG_KEY_ALLOWED_VALUES:
        normalized = str(value).strip().lower()
        allowed = CONFIG_KEY_ALLOWED_VALUES[key]
        if normalized not in allowed:
            options = ", ".join(sorted(option or "<empty>" for option in allowed))
            raise ValueError(f"Invalid value for {key}: {value}. Allowed values: {options}")
        return normalized
    if key in POSITIVE_INT_CONFIG_KEYS and isinstance(value, int) and value <= 0:
        raise ValueError(f"{key} must be greater than 0")
    if key in NON_NEGATIVE_INT_CONFIG_KEYS and isinstance(value, int) and value < 0:
        raise ValueError(f"{key} must be greater than or equal to 0")
    if key in {
        "tools.bash.allowed_workdirs",
        "tools.bash.env_passthrough",
        "tools.files.allowed_roots",
        "tools.files.write_roots",
    }:
        return _validate_string_list_config_value(key, value)
    if key in {"tools.bash.default_workdir"}:
        text = str(value)
        if "\x00" in text:
            raise ValueError(f"{key} must not contain NUL characters")
    if key in RATIO_CONFIG_KEYS:
        numeric = float(value)
        if numeric <= 0 or numeric > 1:
            raise ValueError(f"{key} must be greater than 0 and at most 1")
    return value


def _validate_loaded_config(config: AlphaConfig) -> AlphaConfig:
    values = {
        "llm.provider": config.llm_provider,
        "llm.context.tool_truncate_threshold_ratio": (
            config.llm_context.tool_truncate_threshold_ratio
        ),
        "llm.context.handover_compress_threshold_ratio": (
            config.llm_context.handover_compress_threshold_ratio
        ),
        "llm.context.minimum_remaining_tokens": config.llm_context.minimum_remaining_tokens,
        "llm.context.tool_string_truncate_chars": config.llm_context.tool_string_truncate_chars,
        "llm.context.expected_output_reserve_tokens": (
            config.llm_context.expected_output_reserve_tokens
        ),
        "llm.context.safety_margin_tokens": config.llm_context.safety_margin_tokens,
        "deepseek.reasoning_effort": config.deepseek_reasoning_effort or "",
        "tools.bash.enabled": config.bash_tool.enabled,
        "tools.bash.default_timeout_seconds": config.bash_tool.default_timeout_seconds,
        "tools.bash.max_timeout_seconds": config.bash_tool.max_timeout_seconds,
        "tools.bash.max_output_chars": config.bash_tool.max_output_chars,
        "tools.files.enabled": config.file_tool.enabled,
        "tools.files.patch_enabled": config.file_tool.patch_enabled,
        "tools.files.max_read_chars": config.file_tool.max_read_chars,
        "tools.files.max_file_bytes": config.file_tool.max_file_bytes,
        "tools.files.max_search_results": config.file_tool.max_search_results,
        "tools.files.max_glob_results": config.file_tool.max_glob_results,
        "tools.files.max_read_lines": config.file_tool.max_read_lines,
        "tools.files.create_parent_dirs_enabled": (
            config.file_tool.create_parent_dirs_enabled
        ),
        "tools.files.max_output_chars": config.file_tool.max_output_chars,
        "cognition.background.enabled": config.cognition_background.enabled,
        "cognition.background.startup_delay_seconds": (
            config.cognition_background.startup_delay_seconds
        ),
        "cognition.background.interval_seconds": (
            config.cognition_background.interval_seconds
        ),
        "cognition.background.tick_timeout_seconds": (
            config.cognition_background.tick_timeout_seconds
        ),
        "cognition.background.intake.batch_size": (
            config.cognition_background.intake.batch_size
        ),
        "cognition.background.intake.min_sources": (
            config.cognition_background.intake.min_sources
        ),
        "cognition.background.extraction.batch_size": (
            config.cognition_background.extraction.batch_size
        ),
        "cognition.background.extraction.min_sources": (
            config.cognition_background.extraction.min_sources
        ),
        "cognition.background.consolidation.batch_size": (
            config.cognition_background.consolidation.batch_size
        ),
        "cognition.background.consolidation.min_drafts": (
            config.cognition_background.consolidation.min_drafts
        ),
        "cognition.background.conflict.batch_size": (
            config.cognition_background.conflict.batch_size
        ),
        "cognition.background.conflict.min_conflicts": (
            config.cognition_background.conflict.min_conflicts
        ),
        "cognition.background.summary.batch_size": (
            config.cognition_background.summary.batch_size
        ),
        "cognition.background.summary.initial_min_beliefs": (
            config.cognition_background.summary.initial_min_beliefs
        ),
        "cognition.background.summary.changed_source_min": (
            config.cognition_background.summary.changed_source_min
        ),
        "cognition.background.summary.invalidated_source_min": (
            config.cognition_background.summary.invalidated_source_min
        ),
        "cognition.drive.enabled": config.cognition_drive_enabled,
        "cognition.drive.interval_seconds": config.cognition_drive_interval_seconds,
        "cognition.drive.goal_cooldown_seconds": config.cognition_drive_goal_cooldown_seconds,
        "cognition.drive.active_goal_limit": config.cognition_drive_active_goal_limit,
    }
    for key, value in values.items():
        _validate_config_value(key, value)
    for provider, max_context_tokens in config.llm_provider_max_context_tokens.items():
        if max_context_tokens <= 0:
            raise ValueError(f"llm.providers.{provider}.max_context_tokens must be greater than 0")
    if config.bash_tool.default_timeout_seconds > config.bash_tool.max_timeout_seconds:
        raise ValueError(
            "tools.bash.default_timeout_seconds must be at most "
            "tools.bash.max_timeout_seconds"
        )
    if not config.bash_tool.allowed_workdirs:
        raise ValueError("tools.bash.allowed_workdirs must not be empty")
    if not _path_is_inside_allowed(
        config.bash_tool.default_workdir,
        config.bash_tool.allowed_workdirs,
    ):
        raise ValueError("tools.bash.default_workdir must be within tools.bash.allowed_workdirs")
    if not config.file_tool.allowed_roots:
        raise ValueError("tools.files.allowed_roots must not be empty")
    positive_values = (
        (
            "cognition.consolidation.interval_seconds",
            config.cognition_consolidation_interval_seconds,
        ),
        (
            "cognition.background.interval_seconds",
            config.cognition_background.interval_seconds,
        ),
        (
            "cognition.background.tick_timeout_seconds",
            config.cognition_background.tick_timeout_seconds,
        ),
        (
            "cognition.background.intake.batch_size",
            config.cognition_background.intake.batch_size,
        ),
        (
            "cognition.background.intake.min_sources",
            config.cognition_background.intake.min_sources,
        ),
        (
            "cognition.background.extraction.batch_size",
            config.cognition_background.extraction.batch_size,
        ),
        (
            "cognition.background.extraction.min_sources",
            config.cognition_background.extraction.min_sources,
        ),
        (
            "cognition.background.consolidation.batch_size",
            config.cognition_background.consolidation.batch_size,
        ),
        (
            "cognition.background.consolidation.min_drafts",
            config.cognition_background.consolidation.min_drafts,
        ),
        (
            "cognition.background.conflict.batch_size",
            config.cognition_background.conflict.batch_size,
        ),
        (
            "cognition.background.conflict.min_conflicts",
            config.cognition_background.conflict.min_conflicts,
        ),
        (
            "cognition.background.summary.batch_size",
            config.cognition_background.summary.batch_size,
        ),
        (
            "cognition.background.summary.initial_min_beliefs",
            config.cognition_background.summary.initial_min_beliefs,
        ),
        (
            "cognition.background.summary.changed_source_min",
            config.cognition_background.summary.changed_source_min,
        ),
        (
            "cognition.background.summary.invalidated_source_min",
            config.cognition_background.summary.invalidated_source_min,
        ),
        (
            "cognition.drive.interval_seconds",
            config.cognition_drive_interval_seconds,
        ),
        (
            "cognition.drive.goal_cooldown_seconds",
            config.cognition_drive_goal_cooldown_seconds,
        ),
        ("cognition.drive.active_goal_limit", config.cognition_drive_active_goal_limit),
        ("tools.bash.default_timeout_seconds", config.bash_tool.default_timeout_seconds),
        ("tools.bash.max_timeout_seconds", config.bash_tool.max_timeout_seconds),
        ("tools.bash.max_output_chars", config.bash_tool.max_output_chars),
        ("tools.files.max_read_chars", config.file_tool.max_read_chars),
        ("tools.files.max_file_bytes", config.file_tool.max_file_bytes),
        ("tools.files.max_search_results", config.file_tool.max_search_results),
        ("tools.files.max_glob_results", config.file_tool.max_glob_results),
        ("tools.files.max_read_lines", config.file_tool.max_read_lines),
        ("tools.files.max_output_chars", config.file_tool.max_output_chars),
    )
    for key, value in positive_values:
        if value <= 0:
            raise ValueError(f"{key} must be greater than 0")
    if config.cognition_background.startup_delay_seconds < 0:
        raise ValueError(
            "cognition.background.startup_delay_seconds must be greater than or equal to 0"
        )
    return config


def _validate_config_data(config_data: dict[str, Any]) -> None:
    tools = _section(config_data, "tools")
    bash = tools.get("bash")
    bash = bash if isinstance(bash, dict) else {}
    if bash:
        default_timeout = _int_value(bash.get("default_timeout_seconds"), 120)
        max_timeout = _int_value(bash.get("max_timeout_seconds"), 600)
        if default_timeout <= 0:
            raise ValueError("tools.bash.default_timeout_seconds must be greater than 0")
        if max_timeout <= 0:
            raise ValueError("tools.bash.max_timeout_seconds must be greater than 0")
        if default_timeout > max_timeout:
            raise ValueError(
                "tools.bash.default_timeout_seconds must be at most "
                "tools.bash.max_timeout_seconds"
            )
        max_output_chars = _int_value(bash.get("max_output_chars"), 30000)
        if max_output_chars <= 0:
            raise ValueError("tools.bash.max_output_chars must be greater than 0")
        allowed_workdirs = _list_value(bash.get("allowed_workdirs"), (".",))
        if not allowed_workdirs:
            raise ValueError("tools.bash.allowed_workdirs must not be empty")
        default_workdir = _path_setting(
            _string_from_mapping(bash, "default_workdir", "."),
            "tools.bash.default_workdir",
        )
        allowed_paths = _path_tuple(allowed_workdirs, "tools.bash.allowed_workdirs")
        if not _path_is_inside_allowed(default_workdir, allowed_paths):
            raise ValueError(
                "tools.bash.default_workdir must be within tools.bash.allowed_workdirs"
            )
    files = tools.get("files")
    files = files if isinstance(files, dict) else {}
    if files:
        allowed_roots = _list_value(files.get("allowed_roots"), (".",))
        if not allowed_roots:
            raise ValueError("tools.files.allowed_roots must not be empty")
        _path_tuple(allowed_roots, "tools.files.allowed_roots")
        write_roots = _list_value(files.get("write_roots"), ())
        _path_tuple(write_roots, "tools.files.write_roots")
        for key, default in (
            ("max_read_chars", 20000),
            ("max_file_bytes", 1000000),
            ("max_search_results", 100),
            ("max_glob_results", 500),
            ("max_read_lines", 200),
            ("max_output_chars", 30000),
        ):
            if _int_value(files.get(key), default) <= 0:
                raise ValueError(f"tools.files.{key} must be greater than 0")


def _write_toml_config(path: Path, config_data: dict[str, Any]) -> None:
    sections = (
        "runtime",
        "llm",
        "llm.context",
        "llm.providers.openai-compatible",
        "llm.providers.deepseek",
        "compatible",
        "tools.bash",
        "tools.files",
        "cognition.consolidation",
        "cognition.background",
        "cognition.background.intake",
        "cognition.background.extraction",
        "cognition.background.consolidation",
        "cognition.background.conflict",
        "cognition.background.summary",
        "cognition.drive",
        "deepseek",
        "codex",
        "tavily",
    )
    lines = [
        "# Alpha Agent local configuration.",
        "# Environment variables still override these values for one-off runs or deploys.",
        "",
    ]
    for section_name in sections:
        section = _nested_section(config_data, section_name)
        if not section:
            continue
        lines.append(f"[{section_name}]")
        for field_name, value in section.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{field_name} = {_toml_literal(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Expected integer value for {name}, got: {raw}") from exc


def _int_value(value: Any, default: int) -> int:
    if value is None:
        return default
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"Expected integer value, got: {value}") from exc
    raise ValueError(f"Expected integer value, got: {value}")


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Expected float value for {name}, got: {raw}") from exc


def _float_value(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Expected float value, got: {value}") from exc
    raise ValueError(f"Expected float value, got: {value}")


def _background_config(section: dict[str, Any]) -> CognitionBackgroundConfig:
    intake = _mapping_section(section, "intake")
    extraction = _mapping_section(section, "extraction")
    consolidation = _mapping_section(section, "consolidation")
    conflict = _mapping_section(section, "conflict")
    summary = _mapping_section(section, "summary")
    return CognitionBackgroundConfig(
        enabled=_bool_env(
            "ALPHA_COGNITION_BACKGROUND_ENABLED",
            _bool_value(section.get("enabled"), True),
        ),
        startup_delay_seconds=_int_env(
            "ALPHA_COGNITION_BACKGROUND_STARTUP_DELAY_SECONDS",
            _int_value(section.get("startup_delay_seconds"), 5),
        ),
        interval_seconds=_int_env(
            "ALPHA_COGNITION_BACKGROUND_INTERVAL_SECONDS",
            _int_value(section.get("interval_seconds"), 300),
        ),
        tick_timeout_seconds=_int_env(
            "ALPHA_COGNITION_BACKGROUND_TICK_TIMEOUT_SECONDS",
            _int_value(section.get("tick_timeout_seconds"), 60),
        ),
        intake=BackgroundIntakeConfig(
            batch_size=_int_env(
                "ALPHA_COGNITION_BACKGROUND_INTAKE_BATCH_SIZE",
                _int_value(intake.get("batch_size"), 64),
            ),
            min_sources=_int_env(
                "ALPHA_COGNITION_BACKGROUND_INTAKE_MIN_SOURCES",
                _int_value(intake.get("min_sources"), 1),
            ),
        ),
        extraction=BackgroundExtractionConfig(
            batch_size=_int_env(
                "ALPHA_COGNITION_BACKGROUND_EXTRACTION_BATCH_SIZE",
                _int_value(extraction.get("batch_size"), 12),
            ),
            min_sources=_int_env(
                "ALPHA_COGNITION_BACKGROUND_EXTRACTION_MIN_SOURCES",
                _int_value(extraction.get("min_sources"), 1),
            ),
        ),
        consolidation=BackgroundConsolidationConfig(
            batch_size=_int_env(
                "ALPHA_COGNITION_BACKGROUND_CONSOLIDATION_BATCH_SIZE",
                _int_value(consolidation.get("batch_size"), 12),
            ),
            min_drafts=_int_env(
                "ALPHA_COGNITION_BACKGROUND_CONSOLIDATION_MIN_DRAFTS",
                _int_value(consolidation.get("min_drafts"), 1),
            ),
        ),
        conflict=BackgroundConflictConfig(
            batch_size=_int_env(
                "ALPHA_COGNITION_BACKGROUND_CONFLICT_BATCH_SIZE",
                _int_value(conflict.get("batch_size"), 4),
            ),
            min_conflicts=_int_env(
                "ALPHA_COGNITION_BACKGROUND_CONFLICT_MIN_CONFLICTS",
                _int_value(conflict.get("min_conflicts"), 1),
            ),
        ),
        summary=BackgroundSummaryConfig(
            batch_size=_int_env(
                "ALPHA_COGNITION_BACKGROUND_SUMMARY_BATCH_SIZE",
                _int_value(summary.get("batch_size"), 4),
            ),
            initial_min_beliefs=_int_env(
                "ALPHA_COGNITION_BACKGROUND_SUMMARY_INITIAL_MIN_BELIEFS",
                _int_value(summary.get("initial_min_beliefs"), 12),
            ),
            changed_source_min=_int_env(
                "ALPHA_COGNITION_BACKGROUND_SUMMARY_CHANGED_SOURCE_MIN",
                _int_value(summary.get("changed_source_min"), 6),
            ),
            invalidated_source_min=_int_env(
                "ALPHA_COGNITION_BACKGROUND_SUMMARY_INVALIDATED_SOURCE_MIN",
                _int_value(summary.get("invalidated_source_min"), 1),
            ),
        ),
    )


def _mapping_section(mapping: dict[str, Any], name: str) -> dict[str, Any]:
    value = mapping.get(name)
    return value if isinstance(value, dict) else {}


def _bash_tool_config(section: dict[str, Any]) -> BashToolConfig:
    default_workdir = _path_setting(
        os.getenv("ALPHA_BASH_TOOL_DEFAULT_WORKDIR")
        or _string_from_mapping(section, "default_workdir", "."),
        "tools.bash.default_workdir",
    )
    allowed_workdirs = _path_tuple(
        _list_env(
            "ALPHA_BASH_TOOL_ALLOWED_WORKDIRS",
            _list_value(section.get("allowed_workdirs"), (".",)),
        ),
        "tools.bash.allowed_workdirs",
    )
    return BashToolConfig(
        enabled=_bool_env("ALPHA_BASH_TOOL_ENABLED", _bool_value(section.get("enabled"), False)),
        default_workdir=default_workdir,
        allowed_workdirs=allowed_workdirs,
        default_timeout_seconds=_int_env(
            "ALPHA_BASH_TOOL_DEFAULT_TIMEOUT_SECONDS",
            _int_value(section.get("default_timeout_seconds"), 120),
        ),
        max_timeout_seconds=_int_env(
            "ALPHA_BASH_TOOL_MAX_TIMEOUT_SECONDS",
            _int_value(section.get("max_timeout_seconds"), 600),
        ),
        max_output_chars=_int_env(
            "ALPHA_BASH_TOOL_MAX_OUTPUT_CHARS",
            _int_value(section.get("max_output_chars"), 30000),
        ),
        env_passthrough=tuple(
            _list_env(
                "ALPHA_BASH_TOOL_ENV_PASSTHROUGH",
                _list_value(section.get("env_passthrough"), ()),
            )
        ),
    )


def _file_tool_config(section: dict[str, Any]) -> FileToolConfig:
    allowed_roots = _path_tuple(
        _list_env(
            "ALPHA_FILE_TOOL_ALLOWED_ROOTS",
            _list_value(section.get("allowed_roots"), (".",)),
        ),
        "tools.files.allowed_roots",
    )
    write_roots = _path_tuple(
        _list_env(
            "ALPHA_FILE_TOOL_WRITE_ROOTS",
            _list_value(section.get("write_roots"), ()),
        ),
        "tools.files.write_roots",
    )
    return FileToolConfig(
        enabled=_bool_env("ALPHA_FILE_TOOL_ENABLED", _bool_value(section.get("enabled"), True)),
        allowed_roots=allowed_roots,
        patch_enabled=_bool_env(
            "ALPHA_FILE_TOOL_PATCH_ENABLED",
            _bool_value(section.get("patch_enabled"), False),
        ),
        write_roots=write_roots,
        max_read_chars=_int_env(
            "ALPHA_FILE_TOOL_MAX_READ_CHARS",
            _int_value(section.get("max_read_chars"), 20000),
        ),
        max_file_bytes=_int_env(
            "ALPHA_FILE_TOOL_MAX_FILE_BYTES",
            _int_value(section.get("max_file_bytes"), 1000000),
        ),
        max_search_results=_int_env(
            "ALPHA_FILE_TOOL_MAX_SEARCH_RESULTS",
            _int_value(section.get("max_search_results"), 100),
        ),
        max_glob_results=_int_env(
            "ALPHA_FILE_TOOL_MAX_GLOB_RESULTS",
            _int_value(section.get("max_glob_results"), 500),
        ),
        max_read_lines=_int_env(
            "ALPHA_FILE_TOOL_MAX_READ_LINES",
            _int_value(section.get("max_read_lines"), 200),
        ),
        create_parent_dirs_enabled=_bool_env(
            "ALPHA_FILE_TOOL_CREATE_PARENT_DIRS_ENABLED",
            _bool_value(section.get("create_parent_dirs_enabled"), False),
        ),
        max_output_chars=_int_env(
            "ALPHA_FILE_TOOL_MAX_OUTPUT_CHARS",
            _int_value(section.get("max_output_chars"), 30000),
        ),
    )


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _bool_value(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip():
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return _list_value(raw, ())


def _list_value(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, list | tuple):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return tuple(result)
    raise ValueError(f"Expected string list value, got: {value}")


def _validate_string_list_config_value(key: str, value: Any) -> list[str]:
    items = list(_list_value(value, ()))
    for item in items:
        if "\x00" in item:
            raise ValueError(f"{key} must not contain NUL characters")
    return items


def _path_is_inside_allowed(path: Path, allowed_workdirs: tuple[Path, ...]) -> bool:
    resolved = path.expanduser().resolve()
    allowed = tuple(root.expanduser().resolve() for root in allowed_workdirs)
    return any(resolved == root or resolved.is_relative_to(root) for root in allowed)


def _string_from_mapping(mapping: dict[str, Any], key: str, default: str) -> str:
    value = mapping.get(key)
    if value is None:
        return default
    text = str(value)
    if "\x00" in text:
        raise ValueError(f"tools.bash.{key} must not contain NUL characters")
    return text


def _path_setting(value: str, label: str) -> Path:
    if "\x00" in value:
        raise ValueError(f"{label} must not contain NUL characters")
    return Path(value).expanduser().resolve()


def _path_tuple(values: tuple[str, ...], label: str) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        if "\x00" in value:
            raise ValueError(f"{label} must not contain NUL characters")
        path = Path(value).expanduser().resolve()
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    return tuple(resolved)


def _load_toml_config(config_file: str | Path | None) -> dict[str, Any]:
    path = Path(config_file).expanduser() if config_file is not None else default_config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _provider_max_context_tokens(providers: dict[str, Any]) -> dict[str, int]:
    values = dict(DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS)
    for provider_name, provider_config in providers.items():
        if not isinstance(provider_config, dict):
            continue
        normalized = _normalize_provider_context_key(str(provider_name))
        values[normalized] = _int_value(
            provider_config.get("max_context_tokens"),
            values.get(normalized, DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS["openai-compatible"]),
        )
    return values


def _normalize_provider_context_key(provider_name: str) -> str:
    normalized = provider_name.strip().lower()
    if normalized in {"openai", "compatible"}:
        return "openai-compatible"
    if normalized in {"openai-codex", "openai_codex"}:
        return "codex"
    return normalized


def _section(config_data: dict[str, Any], name: str) -> dict[str, Any]:
    value = config_data.get(name)
    return value if isinstance(value, dict) else {}


def _nested_section(config_data: dict[str, Any], dotted_name: str) -> dict[str, Any]:
    current: Any = config_data
    for part in dotted_name.split("."):
        if not isinstance(current, dict):
            return {}
        current = current.get(part)
    return current if isinstance(current, dict) else {}


def _nested_value(config_data: dict[str, Any], dotted_name: str, default: Any) -> Any:
    parts = dotted_name.split(".")
    current: Any = config_data
    for part in parts[:-1]:
        if not isinstance(current, dict):
            return default
        current = current.get(part)
    if not isinstance(current, dict):
        return default
    return current.get(parts[-1], default)


def _set_nested_value(config_data: dict[str, Any], dotted_name: str, value: Any) -> None:
    parts = dotted_name.split(".")
    current: Any = config_data
    for part in parts[:-1]:
        existing = current.setdefault(part, {})
        if not isinstance(existing, dict):
            raise ValueError(f"Config section {part!r} is not editable")
        current = existing
    current[parts[-1]] = value


def _string_setting(
    config_data: dict[str, Any],
    section: str,
    key: str,
    default: str | None = None,
) -> str | None:
    value = _section(config_data, section).get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return default


def _env_or_config(
    env_name: str,
    config_data: dict[str, Any],
    section: str,
    key: str,
    default: str | None = None,
) -> str | None:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value != "":
        return env_value
    return _string_setting(config_data, section, key, default)
