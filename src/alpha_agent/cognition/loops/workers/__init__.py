"""Consolidation workers."""

from alpha_agent.cognition.loops.scheduler import ScheduledWorker
from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker
from alpha_agent.cognition.loops.workers.memory_consolidation import (
    MemoryConflictReviewWorker,
    MemoryConsolidationWorker,
)
from alpha_agent.cognition.loops.workers.memory_extraction import MemoryExtractionWorker


def default_workers() -> list[ScheduledWorker]:
    return [ArchiveExpiredWorker()]


__all__ = [
    "ArchiveExpiredWorker",
    "MemoryConflictReviewWorker",
    "MemoryConsolidationWorker",
    "MemoryExtractionWorker",
    "default_workers",
]
