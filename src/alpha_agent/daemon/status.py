"""Daemon runtime path and status helpers."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alpha_agent.config import AlphaConfig
from alpha_agent.utils.time import utc_now_iso

INVALID_LOCK_STALE_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class DaemonRuntimeConfig:
    """Filesystem paths used by the daemon runtime."""

    socket_path: Path
    status_path: Path
    log_dir: Path


@dataclass(frozen=True, slots=True)
class DaemonStatus:
    """Serializable daemon runtime status."""

    state: str
    running: bool
    pid: int | None
    socket_path: str
    status_path: str
    updated_at: str
    adapters: list[str]
    db_path: str = ""
    log_dir: str = ""
    message: str = ""
    started_at: str | None = None
    background_enabled: bool = False
    background_state: str = "disabled"
    background_last_tick: str | None = None
    background_last_success: str | None = None
    background_last_error: str | None = None
    background_next_tick: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def daemon_runtime_config(config: AlphaConfig) -> DaemonRuntimeConfig:
    """Return resolved daemon runtime paths from the loaded Alpha config."""

    return DaemonRuntimeConfig(
        socket_path=config.daemon_socket_path.expanduser(),
        status_path=config.daemon_status_path.expanduser(),
        log_dir=config.log_dir.expanduser(),
    )


def running_status(
    *,
    config: AlphaConfig,
    runtime: DaemonRuntimeConfig,
    adapter_names: tuple[str, ...] = (),
    state: str = "running",
    message: str = "Daemon is running.",
    background_status: Any | None = None,
) -> DaemonStatus:
    """Build a running status for the current daemon process."""

    now = utc_now_iso()
    return DaemonStatus(
        state=state,
        running=True,
        pid=os.getpid(),
        socket_path=str(runtime.socket_path),
        status_path=str(runtime.status_path),
        db_path=str(config.db_path),
        log_dir=str(runtime.log_dir),
        updated_at=now,
        started_at=now,
        adapters=list(adapter_names),
        message=message,
        **_background_fields(config, background_status),
    )


def idle_status(
    *,
    config: AlphaConfig,
    runtime: DaemonRuntimeConfig,
    adapter_names: tuple[str, ...] = (),
    message: str = "Daemon is idle; not running.",
    background_status: Any | None = None,
) -> DaemonStatus:
    """Build an idle status when no daemon process is active."""

    return DaemonStatus(
        state="idle",
        running=False,
        pid=None,
        socket_path=str(runtime.socket_path),
        status_path=str(runtime.status_path),
        db_path=str(config.db_path),
        log_dir=str(runtime.log_dir),
        updated_at=utc_now_iso(),
        adapters=list(adapter_names),
        message=message,
        **_background_fields(config, background_status),
    )


def error_status(
    *,
    config: AlphaConfig,
    runtime: DaemonRuntimeConfig,
    adapter_names: tuple[str, ...] = (),
    message: str = "Daemon stopped after an error.",
    background_status: Any | None = None,
) -> DaemonStatus:
    """Build an error status after daemon startup or runtime failure."""

    return DaemonStatus(
        state="error",
        running=False,
        pid=None,
        socket_path=str(runtime.socket_path),
        status_path=str(runtime.status_path),
        db_path=str(config.db_path),
        log_dir=str(runtime.log_dir),
        updated_at=utc_now_iso(),
        adapters=list(adapter_names),
        message=message,
        **_background_fields(config, background_status),
    )


def write_daemon_status(path: Path, status: DaemonStatus) -> None:
    """Persist daemon status as inspectable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def read_daemon_status(path: Path) -> DaemonStatus | None:
    """Read daemon status JSON, returning None when absent or invalid."""

    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return DaemonStatus(
            state=str(raw.get("state", "unknown")),
            running=bool(raw.get("running", False)),
            pid=_optional_int(raw.get("pid")),
            socket_path=str(raw.get("socket_path", "")),
            status_path=str(raw.get("status_path", "")),
            db_path=str(raw.get("db_path", "")),
            log_dir=str(raw.get("log_dir", "")),
            updated_at=str(raw.get("updated_at", "")),
            adapters=[str(adapter) for adapter in raw.get("adapters", [])],
            message=str(raw.get("message", "")),
            started_at=_optional_str(raw.get("started_at")),
            background_enabled=bool(raw.get("background_enabled", False)),
            background_state=str(raw.get("background_state", "disabled")),
            background_last_tick=_optional_str(raw.get("background_last_tick")),
            background_last_success=_optional_str(raw.get("background_last_success")),
            background_last_error=_optional_str(raw.get("background_last_error")),
            background_next_tick=_optional_str(raw.get("background_next_tick")),
        )
    except (TypeError, ValueError):
        return None


def is_pid_running(pid: int | None) -> bool:
    """Return whether a PID appears alive on the local machine."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def cleanup_runtime_files(runtime: DaemonRuntimeConfig) -> None:
    """Remove socket file after daemon shutdown."""

    try:
        runtime.socket_path.unlink()
    except FileNotFoundError:
        return


def _background_fields(config: AlphaConfig, status: Any | None) -> dict[str, Any]:
    if status is None:
        enabled = config.cognition_background.enabled
        return {
            "background_enabled": enabled,
            "background_state": "stopped" if enabled else "disabled",
            "background_last_tick": None,
            "background_last_success": None,
            "background_last_error": None,
            "background_next_tick": None,
        }
    return {
        "background_enabled": bool(getattr(status, "enabled", False)),
        "background_state": str(getattr(status, "state", "unknown")),
        "background_last_tick": _optional_str(getattr(status, "last_tick", None)),
        "background_last_success": _optional_str(getattr(status, "last_success", None)),
        "background_last_error": _optional_str(getattr(status, "last_error", None)),
        "background_next_tick": _optional_str(getattr(status, "next_tick", None)),
    }


def daemon_lock_path(runtime: DaemonRuntimeConfig) -> Path:
    """Return the atomic daemon ownership lock path for this runtime."""

    return runtime.status_path.with_name(f"{runtime.status_path.name}.lock")


@dataclass(slots=True)
class DaemonRuntimeLock:
    """Atomic file lock representing ownership of daemon runtime paths."""

    path: Path
    pid: int
    _released: bool = False

    @classmethod
    def acquire(cls, runtime: DaemonRuntimeConfig) -> DaemonRuntimeLock:
        """Acquire daemon ownership with an atomic lock file."""

        path = daemon_lock_path(runtime)
        path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                pid = os.getpid()
                temp_path.write_text(f"{pid}\n", encoding="utf-8")
                os.link(temp_path, path)
            except FileExistsError as exc:
                existing_pid = _read_lock_pid(path)
                if existing_pid is not None and not is_pid_running(existing_pid):
                    _unlink_lock_if_pid_matches(path, existing_pid)
                    continue
                if existing_pid is None and _invalid_lock_is_stale(path):
                    _unlink_invalid_lock_if_still_invalid(path)
                    continue
                raise FileExistsError(f"Daemon runtime lock already exists: {path}") from exc
            else:
                return cls(path=path, pid=pid)
            finally:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def release(self) -> None:
        """Release the daemon runtime lock if this process still owns it."""

        if self._released:
            return
        if _read_lock_pid(self.path) == self.pid:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._released = True


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _read_lock_pid(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _unlink_lock_if_pid_matches(path: Path, pid: int) -> None:
    if _read_lock_pid(path) != pid:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _unlink_invalid_lock_if_still_invalid(path: Path) -> None:
    if _read_lock_pid(path) is not None:
        return
    if not _invalid_lock_is_stale(path):
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _invalid_lock_is_stale(path: Path) -> bool:
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return False
    return time.time() - mtime >= INVALID_LOCK_STALE_SECONDS
