"""Consolidation workers."""

from alpha_agent.cognition.loops.scheduler import ScheduledWorker
from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker
from alpha_agent.cognition.loops.workers.compress_context import CompressContextWorker
from alpha_agent.cognition.loops.workers.expire_strategies import ExpireStrategiesWorker
from alpha_agent.cognition.loops.workers.learn_value_lens import LearnValueLensWorker
from alpha_agent.cognition.loops.workers.merge_beliefs import MergeBeliefsWorker
from alpha_agent.cognition.loops.workers.resolve_queued_conflicts import (
    ResolveQueuedConflictsWorker,
)
from alpha_agent.cognition.loops.workers.summarize_counterpart import SummarizeCounterpartWorker


def default_workers() -> list[ScheduledWorker]:
    from alpha_agent.cognition.reflectors.l3 import ReflectorL3

    return [
        MergeBeliefsWorker(),
        ArchiveExpiredWorker(),
        CompressContextWorker(),
        SummarizeCounterpartWorker(),
        ResolveQueuedConflictsWorker(),
        LearnValueLensWorker(),
        ExpireStrategiesWorker(),
        ReflectorL3(),
    ]


__all__ = [
    "ArchiveExpiredWorker",
    "CompressContextWorker",
    "ExpireStrategiesWorker",
    "LearnValueLensWorker",
    "MergeBeliefsWorker",
    "ResolveQueuedConflictsWorker",
    "SummarizeCounterpartWorker",
    "default_workers",
]
