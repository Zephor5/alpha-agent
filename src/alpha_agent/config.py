"""Configuration loading for Alpha Agent."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
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

[memory]
retrieval_limit = 8
capture_mode = "auto_approve_explicit"
consolidation_mode = "manual"
consolidation_after_turns = 20

[context]
max_prompt_tokens = 6000
compression_threshold_ratio = 0.85
recent_tail_messages = 8
min_summary_tokens = 256
max_summary_tokens = 1024
semantic_memory_tokens = 512
episodic_memory_tokens = 512
procedural_memory_tokens = 512
session_context_tokens = 2048

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
    "compatible.base_url": str,
    "compatible.api_key": str,
    "memory.retrieval_limit": int,
    "memory.capture_mode": str,
    "memory.consolidation_mode": str,
    "memory.consolidation_after_turns": int,
    "context.max_prompt_tokens": int,
    "context.compression_threshold_ratio": float,
    "context.recent_tail_messages": int,
    "context.min_summary_tokens": int,
    "context.max_summary_tokens": int,
    "context.semantic_memory_tokens": int,
    "context.episodic_memory_tokens": int,
    "context.procedural_memory_tokens": int,
    "context.session_context_tokens": int,
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
    "memory.capture_mode": {"disabled", "candidate_only", "auto_approve_explicit"},
    "memory.consolidation_mode": {"manual", "after_n_turns", "scheduled"},
}

POSITIVE_INT_CONFIG_KEYS = {
    "memory.retrieval_limit",
    "memory.consolidation_after_turns",
    "context.max_prompt_tokens",
    "context.recent_tail_messages",
    "context.min_summary_tokens",
    "context.max_summary_tokens",
    "context.semantic_memory_tokens",
    "context.episodic_memory_tokens",
    "context.procedural_memory_tokens",
    "context.session_context_tokens",
}

RATIO_CONFIG_KEYS = {
    "context.compression_threshold_ratio",
}

SECRET_CONFIG_KEYS = {
    "compatible.api_key",
    "deepseek.api_key",
    "codex.access_token",
}


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
    compatible_base_url: str | None = None
    compatible_api_key: str | None = None
    retrieval_limit: int = 8
    memory_capture_mode: str = "auto_approve_explicit"
    memory_consolidation_mode: str = "manual"
    memory_consolidation_after_turns: int = 20
    context_max_prompt_tokens: int = 6000
    context_compression_threshold_ratio: float = 0.85
    context_recent_tail_messages: int = 8
    context_min_summary_tokens: int = 256
    context_max_summary_tokens: int = 1024
    context_semantic_memory_tokens: int = 512
    context_episodic_memory_tokens: int = 512
    context_procedural_memory_tokens: int = 512
    context_session_context_tokens: int = 2048
    deepseek_api_key: str | None = None
    deepseek_reasoning_enabled: bool = True
    deepseek_reasoning_effort: str | None = None
    codex_access_token: str | None = None


def default_config_path() -> Path:
    """Return the default user config path."""

    return Path(os.getenv("ALPHA_CONFIG_PATH", "~/.alpha-agent/config.toml")).expanduser()


def write_default_config(path: str | Path | None = None, *, overwrite: bool = False) -> bool:
    """Write a default TOML config file.

    Returns True when the file was written and False when it already existed.
    """

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
    expected_type = CONFIG_KEY_TYPES[key]
    parsed_value = _parse_config_value(raw_value, expected_type)
    parsed_value = _validate_config_value(key, parsed_value)
    path = Path(config_path).expanduser() if config_path is not None else default_config_path()
    write_default_config(path)
    config_data = _load_toml_config(path)
    section_name, field_name = key.split(".", 1)
    section = config_data.setdefault(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Config section {section_name!r} is not editable")
    section[field_name] = parsed_value
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
    section_name, field_name = key.split(".", 1)
    value = _section(config_data, section_name).get(field_name, "")
    if key in SECRET_CONFIG_KEYS and not reveal_secret:
        return "***" if value else ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


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
            raise ValueError(f"Expected decimal value, got: {raw_value}") from exc
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
    if key in RATIO_CONFIG_KEYS and isinstance(value, float) and not 0 < value <= 1:
        raise ValueError(f"{key} must be greater than 0 and less than or equal to 1")
    return value


def _write_toml_config(path: Path, config_data: dict[str, Any]) -> None:
    sections = ("runtime", "llm", "compatible", "memory", "context", "deepseek", "codex")
    lines = [
        "# Alpha Agent local configuration.",
        "# Environment variables still override these values for one-off runs or deploys.",
        "",
    ]
    for section_name in sections:
        section = _section(config_data, section_name)
        if not section:
            continue
        lines.append(f"[{section_name}]")
        for field_name, value in section.items():
            lines.append(f"{field_name} = {_toml_literal(value)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
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
        raise ValueError(f"Expected decimal value for {name}, got: {raw}") from exc


def _float_value(value: Any, default: float) -> float:
    if value is None:
        return default
    if type(value) in {int, float}:
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"Expected decimal value, got: {value}") from exc
    raise ValueError(f"Expected decimal value, got: {value}")


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


def _memory_capture_mode_value(value: str | None) -> str:
    normalized = (value or "auto_approve_explicit").strip().lower()
    allowed = CONFIG_KEY_ALLOWED_VALUES["memory.capture_mode"]
    if normalized not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid value for memory.capture_mode: {value}. Allowed values: {options}"
        )
    return normalized


def _memory_consolidation_mode_value(value: str | None) -> str:
    normalized = (value or "manual").strip().lower()
    allowed = CONFIG_KEY_ALLOWED_VALUES["memory.consolidation_mode"]
    if normalized not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValueError(
            f"Invalid value for memory.consolidation_mode: {value}. "
            f"Allowed values: {options}"
        )
    return normalized


def _validate_loaded_config(config: AlphaConfig) -> AlphaConfig:
    values = {
        "llm.provider": config.llm_provider,
        "deepseek.reasoning_effort": config.deepseek_reasoning_effort or "",
        "memory.retrieval_limit": config.retrieval_limit,
        "memory.capture_mode": config.memory_capture_mode,
        "memory.consolidation_mode": config.memory_consolidation_mode,
        "memory.consolidation_after_turns": config.memory_consolidation_after_turns,
        "context.max_prompt_tokens": config.context_max_prompt_tokens,
        "context.compression_threshold_ratio": config.context_compression_threshold_ratio,
        "context.recent_tail_messages": config.context_recent_tail_messages,
        "context.min_summary_tokens": config.context_min_summary_tokens,
        "context.max_summary_tokens": config.context_max_summary_tokens,
        "context.semantic_memory_tokens": config.context_semantic_memory_tokens,
        "context.episodic_memory_tokens": config.context_episodic_memory_tokens,
        "context.procedural_memory_tokens": config.context_procedural_memory_tokens,
        "context.session_context_tokens": config.context_session_context_tokens,
    }
    for key, value in values.items():
        _validate_config_value(key, value)
    return config


def _load_toml_config(config_file: str | Path | None) -> dict[str, Any]:
    path = Path(config_file).expanduser() if config_file is not None else default_config_path()
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        loaded = tomllib.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _section(config_data: dict[str, Any], name: str) -> dict[str, Any]:
    value = config_data.get(name)
    return value if isinstance(value, dict) else {}


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


def load_config(
    env_file: str | Path | None = ".env",
    config_file: str | Path | None = None,
) -> AlphaConfig:
    """Load configuration from TOML, optional .env, and environment variables."""

    if env_file:
        load_dotenv(env_file, override=False)

    config_data = _load_toml_config(config_file)
    memory = _section(config_data, "memory")
    context = _section(config_data, "context")
    deepseek = _section(config_data, "deepseek")

    db_path = Path(
        _env_or_config(
            "ALPHA_DB_PATH",
            config_data,
            "runtime",
            "db_path",
            "~/.alpha-agent/alpha.db",
        )
        or "~/.alpha-agent/alpha.db"
    ).expanduser()
    log_dir = Path(
        _env_or_config("ALPHA_LOG_DIR", config_data, "runtime", "log_dir", "~/.alpha-agent/logs")
        or "~/.alpha-agent/logs"
    ).expanduser()
    gateway_status_path = Path(
        _env_or_config(
            "ALPHA_GATEWAY_STATUS_PATH",
            config_data,
            "runtime",
            "gateway_status_path",
            "~/.alpha-agent/gateway-status.json",
        )
        or "~/.alpha-agent/gateway-status.json"
    ).expanduser()
    daemon_socket_path = Path(
        _env_or_config(
            "ALPHA_DAEMON_SOCKET_PATH",
            config_data,
            "runtime",
            "daemon_socket_path",
            "~/.alpha-agent/daemon.sock",
        )
        or "~/.alpha-agent/daemon.sock"
    ).expanduser()
    daemon_status_path = Path(
        _env_or_config(
            "ALPHA_DAEMON_STATUS_PATH",
            config_data,
            "runtime",
            "daemon_status_path",
            "~/.alpha-agent/daemon-status.json",
        )
        or "~/.alpha-agent/daemon-status.json"
    ).expanduser()
    return _validate_loaded_config(AlphaConfig(
        db_path=db_path,
        log_dir=log_dir,
        gateway_status_path=gateway_status_path,
        daemon_socket_path=daemon_socket_path,
        daemon_status_path=daemon_status_path,
        llm_provider=(
            _env_or_config("ALPHA_LLM_PROVIDER", config_data, "llm", "provider", "mock") or "mock"
        ).strip().lower(),
        llm_model=_env_or_config("ALPHA_LLM_MODEL", config_data, "llm", "model", "") or "",
        llm_debug_logging=_bool_env(
            "ALPHA_LLM_DEBUG_LOGGING",
            _bool_value(_section(config_data, "llm").get("debug_logging"), False),
        ),
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
        retrieval_limit=_int_env(
            "ALPHA_RETRIEVAL_LIMIT",
            _int_value(memory.get("retrieval_limit"), 8),
        ),
        memory_capture_mode=_memory_capture_mode_value(
            _env_or_config(
                "ALPHA_MEMORY_CAPTURE_MODE",
                config_data,
                "memory",
                "capture_mode",
                "auto_approve_explicit",
            )
        ),
        memory_consolidation_mode=_memory_consolidation_mode_value(
            _env_or_config(
                "ALPHA_MEMORY_CONSOLIDATION_MODE",
                config_data,
                "memory",
                "consolidation_mode",
                "manual",
            )
        ),
        memory_consolidation_after_turns=_int_env(
            "ALPHA_MEMORY_CONSOLIDATION_AFTER_TURNS",
            _int_value(memory.get("consolidation_after_turns"), 20),
        ),
        context_max_prompt_tokens=_int_env(
            "ALPHA_CONTEXT_MAX_PROMPT_TOKENS",
            _int_value(context.get("max_prompt_tokens"), 6000),
        ),
        context_compression_threshold_ratio=_float_env(
            "ALPHA_CONTEXT_COMPRESSION_THRESHOLD_RATIO",
            _float_value(context.get("compression_threshold_ratio"), 0.85),
        ),
        context_recent_tail_messages=_int_env(
            "ALPHA_CONTEXT_RECENT_TAIL_MESSAGES",
            _int_value(context.get("recent_tail_messages"), 8),
        ),
        context_min_summary_tokens=_int_env(
            "ALPHA_CONTEXT_MIN_SUMMARY_TOKENS",
            _int_value(context.get("min_summary_tokens"), 256),
        ),
        context_max_summary_tokens=_int_env(
            "ALPHA_CONTEXT_MAX_SUMMARY_TOKENS",
            _int_value(context.get("max_summary_tokens"), 1024),
        ),
        context_semantic_memory_tokens=_int_env(
            "ALPHA_CONTEXT_SEMANTIC_MEMORY_TOKENS",
            _int_value(context.get("semantic_memory_tokens"), 512),
        ),
        context_episodic_memory_tokens=_int_env(
            "ALPHA_CONTEXT_EPISODIC_MEMORY_TOKENS",
            _int_value(context.get("episodic_memory_tokens"), 512),
        ),
        context_procedural_memory_tokens=_int_env(
            "ALPHA_CONTEXT_PROCEDURAL_MEMORY_TOKENS",
            _int_value(context.get("procedural_memory_tokens"), 512),
        ),
        context_session_context_tokens=_int_env(
            "ALPHA_CONTEXT_SESSION_CONTEXT_TOKENS",
            _int_value(context.get("session_context_tokens"), 2048),
        ),
        deepseek_api_key=_env_or_config(
            "ALPHA_DEEPSEEK_API_KEY",
            config_data,
            "deepseek",
            "api_key",
        ),
        deepseek_reasoning_enabled=(
            _bool_env(
                "ALPHA_DEEPSEEK_REASONING_ENABLED",
                _bool_value(deepseek.get("reasoning_enabled"), True),
            )
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
    ))
