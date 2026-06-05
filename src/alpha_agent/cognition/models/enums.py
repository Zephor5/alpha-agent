"""Enumerations for the cognition runtime."""

from __future__ import annotations

from enum import IntEnum, StrEnum


class MemoryKind(StrEnum):
    """Atomic memory assertion kinds."""

    FACT = "fact"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    PROCEDURE = "procedure"
    VALUE = "value"
    RELATIONSHIP = "relationship"


class SummaryKind(StrEnum):
    """Summary belief kinds."""

    COUNTERPART_PROFILE = "counterpart_profile"
    PROJECT_PROFILE = "project_profile"
    DOMAIN_SUMMARY = "domain_summary"
    SELF_MEMORY_SUMMARY = "self_memory_summary"


class DerivationStage(StrEnum):
    """How a belief was produced."""

    TOOL_WRITTEN = "tool_written"
    BACKGROUND_EXTRACTED = "background_extracted"
    BACKGROUND_CONSOLIDATED = "background_consolidated"
    BACKGROUND_SUMMARIZED = "background_summarized"
    HUMAN_CONFIRMED = "human_confirmed"


class BeliefScope(StrEnum):
    """Where a belief applies."""

    GLOBAL = "global"
    COUNTERPART = "counterpart"
    SELF = "self"
    PROJECT = "project"
    SESSION = "session"


class Authority(StrEnum):
    """Categorical source-trust signal for a belief."""

    SYSTEM_DEFINED = "system_defined"
    HUMAN_CONFIRMED = "human_confirmed"
    USER_ASSERTED = "user_asserted"
    BACKGROUND_SYNTHESIZED = "background_synthesized"
    LLM_INTERPRETED = "llm_interpreted"


class BeliefLifecycle(StrEnum):
    """Lifecycle state for a belief entity."""

    PENDING_CONFIRMATION = "pending_confirmation"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETRACTED = "retracted"
    ARCHIVED = "archived"


class BeliefRelationKind(StrEnum):
    """Typed relationship between belief entities."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    DERIVED_FROM = "derived_from"
    ABOUT_SAME_ENTITY_AS = "about_same_entity_as"


class CognitiveEventKind(StrEnum):
    """Cognition event vocabulary."""

    PERCEIVED = "perceived"
    ACTED = "acted"
    RECEIVED_FEEDBACK = "received_feedback"
    MEMORY_PROPOSED = "memory_proposed"
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
    DRIVE = 1
    CONSOLIDATION = 2
