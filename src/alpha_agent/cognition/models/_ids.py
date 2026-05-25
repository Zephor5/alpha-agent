"""Identifiers and small reference records for the cognition runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import CounterpartRole

SubjectId = NewType("SubjectId", str)
CounterpartId = NewType("CounterpartId", str)
BeliefId = NewType("BeliefId", str)
EventId = NewType("EventId", str)
SituationId = NewType("SituationId", str)
PerceptionId = NewType("PerceptionId", str)
JudgmentId = NewType("JudgmentId", str)
DecisionId = NewType("DecisionId", str)
ReflectionId = NewType("ReflectionId", str)
ProcedureId = NewType("ProcedureId", str)

Capability = NewType("Capability", str)
Need = NewType("Need", str)
Role = NewType("Role", str)
GroupRef = NewType("GroupRef", str)
BiasMarker = NewType("BiasMarker", str)
ConfidenceCurve = NewType("ConfidenceCurve", str)
FailurePattern = NewType("FailurePattern", str)
ValueTradeoff = NewType("ValueTradeoff", str)
InteractionPattern = NewType("InteractionPattern", str)
BeliefRelation = NewType("BeliefRelation", str)
ActionHint = NewType("ActionHint", str)
FeedbackEntry = NewType("FeedbackEntry", str)
UpdatePolicy = NewType("UpdatePolicy", str)
Lifecycle = NewType("Lifecycle", str)
IntentMarker = NewType("IntentMarker", str)
ExpectedFeedback = NewType("ExpectedFeedback", str)
Action = NewType("Action", str)
Severity = NewType("Severity", str)
ReflectionKind = NewType("ReflectionKind", str)
ReflectionTarget = NewType("ReflectionTarget", str)
RemedyHint = NewType("RemedyHint", str)
TriggerPattern = NewType("TriggerPattern", str)
Step = NewType("Step", str)
NLStatement = NewType("NLStatement", str)
StructuredClaim = NewType("StructuredClaim", str)
DerivationTrace = NewType("DerivationTrace", str)
Applicability = NewType("Applicability", str)
CompressedSummary = NewType("CompressedSummary", str)
MetaEval = NewType("MetaEval", str)
PhysicalContext = NewType("PhysicalContext", str)
InstitutionalContext = NewType("InstitutionalContext", str)
InformationalContext = NewType("InformationalContext", str)
CulturalContext = NewType("CulturalContext", str)
HistoricalContext = NewType("HistoricalContext", str)
Instant = NewType("Instant", str)


@dataclass(frozen=True)
class Reference:
    """Typed pointer to another cognition object or external entity."""

    kind: str
    id: str

    def to_record(self) -> dict[str, str]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, str]) -> Reference:
        return dataclass_from_record(cls, record)


SubjectRef = Reference
CounterpartRef = Reference
BeliefRef = Reference
JudgmentRef = Reference
ProcedureRef = Reference
ReflectionRef = Reference
EvidenceRef = Reference
SituationRef = Reference
EntityRef = Reference
ActorRef = Reference
StrategyRef = Reference


@dataclass(frozen=True)
class SelfModel:
    """Stable self-model fields populated by later reflective phases."""

    capabilities_self_assessed: dict[Capability, ConfidenceCurve] = field(default_factory=dict)
    typical_failure_modes: list[FailurePattern] = field(default_factory=list)
    preferred_strategies: list[StrategyRef] = field(default_factory=list)
    stable_preferences: list[BeliefRef] = field(default_factory=list)
    typical_value_tradeoffs: list[ValueTradeoff] = field(default_factory=list)
    interaction_patterns_by_counterpart_role: dict[CounterpartRole, InteractionPattern] = field(
        default_factory=dict
    )

    def to_record(self) -> dict[str, object]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, object]) -> SelfModel:
        return dataclass_from_record(cls, record)


def subject_ref(subject_id: SubjectId) -> SubjectRef:
    return Reference("subject", str(subject_id))


def counterpart_ref(counterpart_id: CounterpartId) -> CounterpartRef:
    return Reference("counterpart", str(counterpart_id))


def belief_ref(belief_id: BeliefId) -> BeliefRef:
    return Reference("belief", str(belief_id))


def entity_ref(entity_id: str) -> EntityRef:
    return Reference("entity", entity_id)


def situation_ref(situation_id: SituationId) -> SituationRef:
    return Reference("situation", str(situation_id))


def actor_ref(actor_id: str) -> ActorRef:
    return Reference("actor", actor_id)
