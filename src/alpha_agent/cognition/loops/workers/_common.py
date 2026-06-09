"""Shared helpers for retained consolidation workers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from alpha_agent.cognition.loops.scheduler import WorkerCheckpoint, WorkerReport
from alpha_agent.cognition.processing_ledger import BackgroundSourceWindow, BackgroundStage


def report(
    worker: str,
    checkpoint: WorkerCheckpoint,
    *,
    inspected: int,
    emitted: int,
    notes: list[str] | None = None,
    yielded: bool = False,
    metadata: dict[str, object] | None = None,
) -> WorkerReport:
    return WorkerReport(
        worker=worker,
        inspected=inspected,
        emitted=emitted,
        notes=notes or [],
        yielded_to_higher_priority=yielded,
        new_checkpoint=WorkerCheckpoint(
            worker_name=worker,
            last_run_at=checkpoint.last_run_at,
            last_processed_event_id=checkpoint.last_processed_event_id,
            last_status="yielded" if yielded else "ok",
            metadata=metadata if metadata is not None else checkpoint.metadata,
        ),
    )


def after_cursor_wrap[T](items: list[T], cursor: str, key: Any) -> list[T]:
    """Return sorted work after cursor, then wrap to the lower/equal prefix."""

    if not cursor:
        return items
    greater = [item for item in items if str(key(item)) > cursor]
    lower_or_equal = [item for item in items if str(key(item)) <= cursor]
    return greater + lower_or_equal


def background_llm_trace_metadata(
    *,
    worker_name: str,
    worker_id: str,
    stage: BackgroundStage,
    window: BackgroundSourceWindow,
    run_id: str,
    session_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return stable context attached to background LLM debug traces."""

    worker: dict[str, Any] = {
        "name": worker_name,
        "worker_id": worker_id,
        "stage": stage.value,
        "target_unit": window.target_unit,
        "window_id": window.window_id,
        "run_id": run_id,
    }
    resolved_session_id = session_id or _session_id_from_target_unit(window.target_unit)
    if resolved_session_id is not None:
        worker["session_id"] = resolved_session_id
    return {"worker": {**worker, **dict(extra or {})}}


def _session_id_from_target_unit(target_unit: str) -> str | None:
    prefix = "session:"
    if not target_unit.startswith(prefix):
        return None
    session_id = target_unit[len(prefix) :]
    return session_id or None
