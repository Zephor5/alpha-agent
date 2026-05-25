"""Consolidation workers."""

from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker
from alpha_agent.cognition.loops.workers.compress_context import CompressContextWorker
from alpha_agent.cognition.loops.workers.learn_procedure import LearnProcedureWorker
from alpha_agent.cognition.loops.workers.learn_value_lens import LearnValueLensWorker
from alpha_agent.cognition.loops.workers.merge_beliefs import MergeBeliefsWorker
from alpha_agent.cognition.loops.workers.promote_judgment import PromoteJudgmentWorker
from alpha_agent.cognition.loops.workers.resolve_queued_conflicts import (
    ResolveQueuedConflictsWorker,
)
from alpha_agent.cognition.loops.workers.summarize_counterpart import SummarizeCounterpartWorker


def default_workers():
    return [
        PromoteJudgmentWorker(),
        MergeBeliefsWorker(),
        ArchiveExpiredWorker(),
        LearnProcedureWorker(),
        CompressContextWorker(),
        SummarizeCounterpartWorker(),
        ResolveQueuedConflictsWorker(),
        LearnValueLensWorker(),
    ]


__all__ = [
    "ArchiveExpiredWorker",
    "CompressContextWorker",
    "LearnProcedureWorker",
    "LearnValueLensWorker",
    "MergeBeliefsWorker",
    "PromoteJudgmentWorker",
    "ResolveQueuedConflictsWorker",
    "SummarizeCounterpartWorker",
    "default_workers",
]
