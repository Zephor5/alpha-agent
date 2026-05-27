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

[cognition.consolidation]
enabled = true
interval_seconds = 300
judgment_repeat_window = 20
judgment_repeat_threshold = 3
procedure_success_threshold = 3
context_foreground_max = 8
context_absorb_batch = 4
context_summary_chars = 480
counterpart_digest_min_beliefs = 5
counterpart_digest_min_new_beliefs = 3

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
    "cognition.drive.enabled": bool,
    "cognition.drive.interval_seconds": int,
    "cognition.drive.goal_cooldown_seconds": int,
    "cognition.drive.active_goal_limit": int,
    "deepseek.api_key": str,
    "deepseek.reasoning_enabled": bool,
    "deepseek.reasoning_effort": str,
    "codex.access_token": str,
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
    "cognition.drive.active_goal_limit",
    "cognition.drive.goal_cooldown_seconds",
    "cognition.drive.interval_seconds",
    "llm.context.minimum_remaining_tokens",
    "llm.context.tool_string_truncate_chars",
    "llm.context.expected_output_reserve_tokens",
    "llm.context.safety_margin_tokens",
    "llm.providers.openai-compatible.max_context_tokens",
    "llm.providers.deepseek.max_context_tokens",
}

RATIO_CONFIG_KEYS = {
    "llm.context.tool_truncate_threshold_ratio",
    "llm.context.handover_compress_threshold_ratio",
}

SECRET_CONFIG_KEYS = {
    "compatible.api_key",
    "deepseek.api_key",
    "codex.access_token",
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
    llm_provider_max_context_tokens: dict[str, int] = field(
        default_factory=lambda: dict(DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS)
    )
    compatible_base_url: str | None = None
    compatible_api_key: str | None = None
    cognition_consolidation_enabled: bool = True
    cognition_consolidation_interval_seconds: int = 300
    cognition_consolidation_judgment_repeat_window: int = 20
    cognition_consolidation_judgment_repeat_threshold: int = 3
    cognition_consolidation_procedure_success_threshold: int = 3
    cognition_consolidation_context_foreground_max: int = 8
    cognition_consolidation_context_absorb_batch: int = 4
    cognition_consolidation_context_summary_chars: int = 480
    cognition_consolidation_counterpart_digest_min_beliefs: int = 5
    cognition_consolidation_counterpart_digest_min_new_beliefs: int = 3
    cognition_drive_enabled: bool = False
    cognition_drive_interval_seconds: int = 300
    cognition_drive_goal_cooldown_seconds: int = 3600
    cognition_drive_active_goal_limit: int = 8
    deepseek_api_key: str | None = None
    deepseek_reasoning_enabled: bool = True
    deepseek_reasoning_effort: str | None = None
    codex_access_token: str | None = None

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
    cognition = _section(config_data, "cognition")
    consolidation = cognition.get("consolidation")
    consolidation = consolidation if isinstance(consolidation, dict) else {}
    drive = cognition.get("drive")
    drive = drive if isinstance(drive, dict) else {}
    deepseek = _section(config_data, "deepseek")

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
        cognition_consolidation_judgment_repeat_window=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_JUDGMENT_REPEAT_WINDOW",
            _int_value(consolidation.get("judgment_repeat_window"), 20),
        ),
        cognition_consolidation_judgment_repeat_threshold=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_JUDGMENT_REPEAT_THRESHOLD",
            _int_value(consolidation.get("judgment_repeat_threshold"), 3),
        ),
        cognition_consolidation_procedure_success_threshold=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_PROCEDURE_SUCCESS_THRESHOLD",
            _int_value(consolidation.get("procedure_success_threshold"), 3),
        ),
        cognition_consolidation_context_foreground_max=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_CONTEXT_FOREGROUND_MAX",
            _int_value(consolidation.get("context_foreground_max"), 8),
        ),
        cognition_consolidation_context_absorb_batch=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_CONTEXT_ABSORB_BATCH",
            _int_value(consolidation.get("context_absorb_batch"), 4),
        ),
        cognition_consolidation_context_summary_chars=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_CONTEXT_SUMMARY_CHARS",
            _int_value(consolidation.get("context_summary_chars"), 480),
        ),
        cognition_consolidation_counterpart_digest_min_beliefs=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_COUNTERPART_DIGEST_MIN_BELIEFS",
            _int_value(consolidation.get("counterpart_digest_min_beliefs"), 5),
        ),
        cognition_consolidation_counterpart_digest_min_new_beliefs=_int_env(
            "ALPHA_COGNITION_CONSOLIDATION_COUNTERPART_DIGEST_MIN_NEW_BELIEFS",
            _int_value(consolidation.get("counterpart_digest_min_new_beliefs"), 3),
        ),
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
            or os.getenv("ALPHA_CODEX_API_KEY")
            or _string_setting(config_data, "codex", "access_token")
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
    positive_values = (
        (
            "cognition.consolidation.interval_seconds",
            config.cognition_consolidation_interval_seconds,
        ),
        (
            "cognition.consolidation.judgment_repeat_window",
            config.cognition_consolidation_judgment_repeat_window,
        ),
        (
            "cognition.consolidation.judgment_repeat_threshold",
            config.cognition_consolidation_judgment_repeat_threshold,
        ),
        (
            "cognition.consolidation.procedure_success_threshold",
            config.cognition_consolidation_procedure_success_threshold,
        ),
        (
            "cognition.consolidation.context_foreground_max",
            config.cognition_consolidation_context_foreground_max,
        ),
        (
            "cognition.consolidation.context_absorb_batch",
            config.cognition_consolidation_context_absorb_batch,
        ),
        (
            "cognition.consolidation.context_summary_chars",
            config.cognition_consolidation_context_summary_chars,
        ),
        (
            "cognition.consolidation.counterpart_digest_min_beliefs",
            config.cognition_consolidation_counterpart_digest_min_beliefs,
        ),
        (
            "cognition.consolidation.counterpart_digest_min_new_beliefs",
            config.cognition_consolidation_counterpart_digest_min_new_beliefs,
        ),
        ("cognition.drive.interval_seconds", config.cognition_drive_interval_seconds),
        (
            "cognition.drive.goal_cooldown_seconds",
            config.cognition_drive_goal_cooldown_seconds,
        ),
        ("cognition.drive.active_goal_limit", config.cognition_drive_active_goal_limit),
    )
    for key, value in positive_values:
        if value <= 0:
            raise ValueError(f"{key} must be greater than 0")
    return config


def _write_toml_config(path: Path, config_data: dict[str, Any]) -> None:
    sections = (
        "runtime",
        "llm",
        "llm.context",
        "llm.providers.openai-compatible",
        "llm.providers.deepseek",
        "compatible",
        "cognition.consolidation",
        "cognition.drive",
        "deepseek",
        "codex",
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
