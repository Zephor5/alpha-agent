"""Consolidation workers."""

from alpha_agent.cognition.loops.scheduler import ScheduledWorker
from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker


def default_workers() -> list[ScheduledWorker]:
    return [ArchiveExpiredWorker()]


__all__ = [
    "ArchiveExpiredWorker",
    "default_workers",
]
