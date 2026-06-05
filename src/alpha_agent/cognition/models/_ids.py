"""Identifiers and small reference records for the cognition runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record

SubjectId = NewType("SubjectId", str)
CounterpartId = NewType("CounterpartId", str)
BeliefId = NewType("BeliefId", str)
GoalId = NewType("GoalId", str)
EventId = NewType("EventId", str)
SituationId = NewType("SituationId", str)
PerceptionId = NewType("PerceptionId", str)

Capability = NewType("Capability", str)
Need = NewType("Need", str)
Role = NewType("Role", str)
GroupRef = NewType("GroupRef", str)
BeliefRelation = NewType("BeliefRelation", str)
ActionHint = NewType("ActionHint", str)
FeedbackEntry = NewType("FeedbackEntry", str)
IntentMarker = NewType("IntentMarker", str)
NLStatement = NewType("NLStatement", str)
StructuredClaim = NewType("StructuredClaim", str)
DerivationTrace = NewType("DerivationTrace", str)
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
EvidenceRef = Reference
SituationRef = Reference
EntityRef = Reference
ActorRef = Reference


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
