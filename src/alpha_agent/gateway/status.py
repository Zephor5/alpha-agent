"""Gateway status file and diagnostic helpers."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from alpha_agent.utils.time import utc_now_iso

GATEWAY_TABLES = ("gateway_session_mappings", "gateway_dedup")


@dataclass(frozen=True, slots=True)
class GatewayStatus:
    """Serializable gateway runtime status."""

    state: str
    running: bool
    pid: int | None
    updated_at: str
    db_path: str
    log_dir: str
    adapter_count: int
    adapters: list[str]
    message: str
    started_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


def idle_status(
    *,
    db_path: Path,
    log_dir: Path,
    message: str = "Gateway is idle; not running.",
) -> GatewayStatus:
    """Build an idle gateway status for missing or stopped runtime state."""

    return GatewayStatus(
        state="idle",
        running=False,
        pid=None,
        updated_at=utc_now_iso(),
        db_path=str(db_path),
        log_dir=str(log_dir),
        adapter_count=0,
        adapters=[],
        message=message,
    )


def running_status(
    *,
    db_path: Path,
    log_dir: Path,
    adapter_names: tuple[str, ...],
    message: str,
) -> GatewayStatus:
    """Build a running gateway status for the current process."""

    now = utc_now_iso()
    return GatewayStatus(
        state="running",
        running=True,
        pid=os.getpid(),
        started_at=now,
        updated_at=now,
        db_path=str(db_path),
        log_dir=str(log_dir),
        adapter_count=len(adapter_names),
        adapters=list(adapter_names),
        message=message,
    )


def write_gateway_status(path: Path, status: GatewayStatus) -> None:
    """Persist gateway status as inspectable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status.to_json(), indent=2, sort_keys=True), encoding="utf-8")


def read_gateway_status(path: Path) -> GatewayStatus | None:
    """Read gateway status JSON, returning None when the file is absent or invalid."""

    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return GatewayStatus(
            state=str(raw.get("state", "unknown")),
            running=bool(raw.get("running", False)),
            pid=_optional_int(raw.get("pid")),
            started_at=_optional_str(raw.get("started_at")),
            updated_at=str(raw.get("updated_at", "")),
            db_path=str(raw.get("db_path", "")),
            log_dir=str(raw.get("log_dir", "")),
            adapter_count=int(raw.get("adapter_count", 0)),
            adapters=[str(adapter) for adapter in raw.get("adapters", [])],
            message=str(raw.get("message", "")),
        )
    except (TypeError, ValueError):
        return None


def gateway_tables_available(db_path: Path) -> dict[str, bool]:
    """Return availability for the P0 gateway tables."""

    if not db_path.exists():
        return {table: False for table in GATEWAY_TABLES}
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name IN (?, ?)
            """,
            GATEWAY_TABLES,
        ).fetchall()
    found = {str(row[0]) for row in rows}
    return {table: table in found for table in GATEWAY_TABLES}


def is_pid_running(pid: int | None) -> bool:
    """Return whether a PID appears alive on the local machine."""

    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
