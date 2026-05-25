"""Consolidation workers."""

from alpha_agent.cognition.loops.workers.archive_expired import ArchiveExpiredWorker
from alpha_agent.cognition.loops.workers.compress_context import CompressContextWorker
from alpha_agent.cognition.loops.workers.learn_procedure import LearnProcedureWorker
from alpha_agent.cognition.loops.workers.merge_beliefs import MergeBeliefsWorker
from alpha_agent.cognition.loops.workers.promote_judgment import PromoteJudgmentWorker
from alpha_agent.cognition.loops.workers.summarize_counterpart import SummarizeCounterpartWorker


def default_workers():
    return [
        PromoteJudgmentWorker(),
        MergeBeliefsWorker(),
        ArchiveExpiredWorker(),
        LearnProcedureWorker(),
        CompressContextWorker(),
        SummarizeCounterpartWorker(),
    ]


__all__ = [
    "ArchiveExpiredWorker",
    "CompressContextWorker",
    "LearnProcedureWorker",
    "MergeBeliefsWorker",
    "PromoteJudgmentWorker",
    "SummarizeCounterpartWorker",
    "default_workers",
]
