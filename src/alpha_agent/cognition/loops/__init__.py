"""Cognition loop infrastructure."""

from alpha_agent.cognition.loops.background_service import (
    BackgroundCognitionService,
    BackgroundCognitionStatus,
    SourceIntakeWorker,
)
from alpha_agent.cognition.loops.compact_extraction import DirectCompactExtractionService
from alpha_agent.cognition.loops.consolidation import ConsolidationConfig, ConsolidationLoop
from alpha_agent.cognition.loops.drive import DriveConfig, DriveLoop, DriveReport
from alpha_agent.cognition.loops.feedback_attribution import (
    FeedbackAttributionJob,
    RealtimeFeedbackAttributionService,
    RecalledBeliefHandle,
    claim_feedback_attribution_sources,
    complete_feedback_attribution_sources,
    fail_feedback_attribution_sources,
    feedback_attribution_idempotency_key,
    feedback_attribution_target_unit,
    recalled_beliefs_for_previous_turn,
)
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
    "FeedbackAttributionJob",
    "InMemoryCheckpointStore",
    "RealtimeFeedbackAttributionService",
    "RecalledBeliefHandle",
    "ScheduleTrigger",
    "Scheduler",
    "SourceIntakeWorker",
    "WorkerCheckpoint",
    "WorkerReport",
    "claim_feedback_attribution_sources",
    "complete_feedback_attribution_sources",
    "fail_feedback_attribution_sources",
    "feedback_attribution_idempotency_key",
    "feedback_attribution_target_unit",
    "recalled_beliefs_for_previous_turn",
]
