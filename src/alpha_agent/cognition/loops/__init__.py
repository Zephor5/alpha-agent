"""Cognition loop infrastructure."""

from alpha_agent.cognition.loops.background_service import (
    BackgroundCognitionService,
    BackgroundCognitionStatus,
    SourceIntakeWorker,
)
from alpha_agent.cognition.loops.compact_extraction import DirectCompactExtractionService
from alpha_agent.cognition.loops.consolidation import ConsolidationConfig, ConsolidationLoop
from alpha_agent.cognition.loops.drive import DriveConfig, DriveLoop, DriveReport
from alpha_agent.cognition.loops.scheduler import (
    CheckpointStore,
    InMemoryCheckpointStore,
    Scheduler,
    ScheduleTrigger,
    WorkerCheckpoint,
    WorkerReport,
)

__all__ = [
    "CheckpointStore",
    "BackgroundCognitionService",
    "BackgroundCognitionStatus",
    "ConsolidationConfig",
    "ConsolidationLoop",
    "DirectCompactExtractionService",
    "DriveConfig",
    "DriveLoop",
    "DriveReport",
    "InMemoryCheckpointStore",
    "ScheduleTrigger",
    "Scheduler",
    "SourceIntakeWorker",
    "WorkerCheckpoint",
    "WorkerReport",
]
