"""Enumerations for the cognition runtime."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class CognitiveType(StrEnum):
    """Shape of a belief's cognitive content."""

    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    PREFERENCE = "preference"
    VALUE = "value"
    CAUSAL = "causal"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    META = "meta"
    CONCEPT = "concept"


class ValueKind(StrEnum):
    """Value dimensions used by profiles and lenses."""

    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    SAFETY = "safety"
    AUTONOMY = "autonomy"
    EFFICIENCY = "efficiency"
    LEARNING = "learning"


class ThreadKind(StrEnum):
    """Context thread category."""

    CONVERSATION = "conversation"
    COGNITION = "cognition"


class CognitiveEventKind(StrEnum):
    """First event-kind vocabulary for the event log."""

    PERCEIVED = "perceived"
    ATTENDED = "attended"
    INTERPRETED = "interpreted"
    JUDGED = "judged"
    DECIDED = "decided"
    ACTED = "acted"
    RECEIVED_FEEDBACK = "received_feedback"
    REFLECTED = "reflected"
    REVISED = "revised"
    BELIEF_FORMED = "belief_formed"
    BELIEF_STRENGTHENED = "belief_strengthened"
    BELIEF_WEAKENED = "belief_weakened"
    BELIEF_SUPERSEDED = "belief_superseded"
    BELIEF_RETRACTED = "belief_retracted"
    BIAS_DETECTED = "bias_detected"
    STRATEGY_CHANGED = "strategy_changed"
    STRATEGY_EXPIRED = "strategy_expired"
    SELF_MODEL_UPDATED = "self_model_updated"
    PROCEDURE_LEARNED = "procedure_learned"
    PROCEDURE_STRENGTHENED = "procedure_strengthened"
    PROCEDURE_WEAKENED = "procedure_weakened"
    PROCEDURE_MATCHED = "procedure_matched"
    VALUE_LENS_SHIFTED = "value_lens_shifted"
    CONTEXT_COMPRESSED = "context_compressed"
    CONSOLIDATION_CONFLICT_QUEUED = "consolidation_conflict_queued"
    CONFLICT_KEPT_FOR_HUMAN_REVIEW = "conflict_kept_for_human_review"
    BELIEF_ARCHIVED = "belief_archived"
    BELIEF_FORM_PENDING_CONFIRMATION = "belief_form_pending_confirmation"
    CONTEXT_ANCHOR_SET = "context_anchor_set"
    CONTEXT_ANCHOR_CLEARED = "context_anchor_cleared"
    COUNTERPART_FIRST_OBSERVED = "counterpart_first_observed"
    COUNTERPART_IDENTIFIED = "counterpart_identified"
    COUNTERPART_RELATIONSHIP_CHANGED = "counterpart_relationship_changed"
    TURN_SOURCES_RECORDED = "turn_sources_recorded"
    SERVICE_COMMITTED = "service_committed"
    SERVICE_FULFILLED = "service_fulfilled"
    SERVICE_FAILED = "service_failed"
    TRUST_UPDATED = "trust_updated"
    LOOP_ACQUIRED = "loop_acquired"
    LOOP_RELEASED = "loop_released"
    LOOP_YIELDED = "loop_yielded"
    GOAL_SET = "goal_set"
    GOAL_SATISFIED = "goal_satisfied"
    GOAL_ABANDONED = "goal_abandoned"
    GOAL_PROGRESSED = "goal_progressed"
    EXTERNAL_SIGNAL_RECEIVED = "external_signal_received"


class CounterpartRole(StrEnum):
    """Role played by a counterpart."""

    USER = "user"
    OPERATOR = "operator"
    PEER_AGENT = "peer_agent"
    SYSTEM = "system"
    ANONYMOUS = "anonymous"


class StimulusKind(StrEnum):
    """External or internal stimulus source."""

    USER_MESSAGE = "user_message"
    TOOL_RESULT = "tool_result"
    CLOCK_TICK = "clock_tick"
    SELF_SIGNAL = "self_signal"
    WEBHOOK = "webhook"
    INTER_AGENT = "inter_agent"


class LoopPriority(IntEnum):
    """Single-subject loop scheduling priority. Lower numbers run first."""

    REACTIVE = 0
    L2 = 1
    DRIVE = 2
    CONSOLIDATION = 3
    L3 = 4
