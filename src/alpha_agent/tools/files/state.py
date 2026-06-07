"""Turn-scoped file tool state helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alpha_agent.tools.base import ToolExecutionContext


def read_ledger(context: ToolExecutionContext) -> dict[Any, Any] | None:
    state = getattr(context, "turn_state", None)
    if state is None:
        state = context.extensions.get("turn_state") if context.extensions else None
    if state is None:
        return None
    ledger = getattr(state, "file_read_ledger", None)
    if ledger is None:
        try:
            ledger = {}
            state.file_read_ledger = ledger
        except Exception:
            return None
    if isinstance(ledger, dict):
        return ledger
    return None


def invalidate_read_ledger(context: ToolExecutionContext, paths: list[Path]) -> None:
    ledger = read_ledger(context)
    if not ledger:
        return
    resolved = {path.resolve(strict=False) for path in paths}
    for key in list(ledger):
        if isinstance(key, tuple) and key and key[0] in resolved:
            del ledger[key]
