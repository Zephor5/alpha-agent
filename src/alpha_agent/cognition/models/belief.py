"""Belief model."""

from __future__ import annotations

from dataclasses import dataclass, field

from alpha_agent.cognition.models._ids import (
    ActionHint,
    Applicability,
    BeliefId,
    BeliefRef,
    BeliefRelation,
    DerivationTrace,
    EvidenceRef,
    FeedbackEntry,
    Instant,
    Lifecycle,
    NLStatement,
    Reference,
    ReflectionRef,
    Role,
    SituationRef,
    StructuredClaim,
    SubjectRef,
    UpdatePolicy,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import CognitiveType
from alpha_agent.cognition.models.value import ValueProfile


@dataclass(frozen=True)
class Belief:
    """Immutable belief held by the subject."""

    id: BeliefId
    subject: SubjectRef
    about: list[Reference]
    object: str
    content: NLStatement
    cognitive_type: CognitiveType
    structure: StructuredClaim | None
    sources: list[EvidenceRef]
    confidence: float
    applicability: Applicability
    value_profile: ValueProfile
    relations: list[BeliefRelation]
    formed_in: SituationRef
    holder_role: Role
    action_orientation: list[ActionHint]
    update_policy: UpdatePolicy
    status: Lifecycle
    held_since: Instant
    derivation: DerivationTrace | None = None
    feedback_history: list[FeedbackEntry] = field(default_factory=list)
    held_until: Instant | None = None
    superseded_by: BeliefRef | None = None
    supersedes: BeliefRef | None = None
    self_audit: list[ReflectionRef] = field(default_factory=list)

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> Belief:
        return dataclass_from_record(cls, record)
