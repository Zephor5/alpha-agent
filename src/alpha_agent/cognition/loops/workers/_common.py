"""Shared helpers for retained consolidation workers."""

from __future__ import annotations

from typing import Any

from alpha_agent.cognition.loops.scheduler import WorkerCheckpoint, WorkerReport


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
