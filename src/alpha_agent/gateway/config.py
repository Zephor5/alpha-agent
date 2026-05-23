"""Gateway runtime configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpha_agent.config import AlphaConfig
from alpha_agent.gateway.adapters import PlatformAdapter

GATEWAY_LOG_FILENAMES = ("agent.log", "gateway.log", "errors.log")


@dataclass(frozen=True, slots=True)
class GatewayRuntimeConfig:
    """Filesystem paths used by the gateway operational shell."""

    log_dir: Path
    status_path: Path
    log_paths: dict[str, Path]


def gateway_runtime_config(config: AlphaConfig) -> GatewayRuntimeConfig:
    """Return resolved gateway runtime paths from the loaded Alpha config."""

    log_dir = config.log_dir.expanduser()
    return GatewayRuntimeConfig(
        log_dir=log_dir,
        status_path=config.gateway_status_path.expanduser(),
        log_paths={filename: log_dir / filename for filename in GATEWAY_LOG_FILENAMES},
    )


def ensure_gateway_runtime_files(runtime: GatewayRuntimeConfig) -> None:
    """Create the gateway runtime directory and empty log files if absent."""

    runtime.log_dir.mkdir(parents=True, exist_ok=True)
    runtime.status_path.parent.mkdir(parents=True, exist_ok=True)
    for path in runtime.log_paths.values():
        path.touch(exist_ok=True)


def configured_adapter_names() -> tuple[str, ...]:
    """Return configured platform adapters.

    P0 intentionally ships no real Feishu or WeChat adapters, so this remains empty until
    a concrete adapter configuration exists.
    """

    return tuple(adapter_name(adapter) for adapter in configured_adapters())


def configured_adapters() -> tuple[PlatformAdapter, ...]:
    """Return configured platform adapter instances.

    No real platform adapters ship in this package yet. Tests and downstream deployments can
    supply concrete adapters by replacing this factory without changing the adapter contract.
    """

    return ()


def adapter_name(adapter: PlatformAdapter) -> str:
    """Return a stable display name for an adapter instance."""

    name = getattr(adapter, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    return adapter.__class__.__name__
