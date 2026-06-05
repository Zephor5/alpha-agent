"""Belief ontology models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

from alpha_agent.cognition.models._ids import (
    ActionHint,
    BeliefId,
    BeliefRef,
    DerivationTrace,
    EvidenceRef,
    FeedbackEntry,
    Instant,
    NLStatement,
    Reference,
    Role,
    SituationId,
    SituationRef,
    SubjectId,
    SubjectRef,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models._serialization import dataclass_from_record, dataclass_to_record
from alpha_agent.cognition.models.enums import (
    Authority,
    BeliefLifecycle,
    BeliefRelationKind,
    BeliefScope,
    DerivationStage,
    MemoryKind,
    SummaryKind,
)

_SCOPE_REFERENCE_KINDS: dict[BeliefScope, frozenset[str]] = {
    BeliefScope.COUNTERPART: frozenset({"counterpart"}),
    BeliefScope.SELF: frozenset({"subject", "self"}),
    BeliefScope.PROJECT: frozenset({"project"}),
    BeliefScope.SESSION: frozenset({"session"}),
}


@dataclass(frozen=True)
class ValidityWindow:
    """Explicit temporal applicability for any memory kind."""

    observed_at: Instant | None = None
    valid_from: Instant | None = None
    valid_until: Instant | None = None
    recurrence: str | None = None

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class BeliefRelationRecord:
    """Typed link from one belief to another belief or referenced entity."""

    kind: BeliefRelationKind
    target: Reference

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BeliefRelationKind(self.kind))
        _validate_reference(self.target, "belief relation target")

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class AtomicBelief:
    """Immutable first-order assertion held by the subject."""

    id: BeliefId
    subject: SubjectRef
    about: list[Reference]
    object: str
    content: NLStatement
    memory_kind: MemoryKind
    derivation_stage: DerivationStage
    scope: BeliefScope
    authority: Authority
    lifecycle: BeliefLifecycle = BeliefLifecycle.ACTIVE
    structure: dict[str, Any] | None = None
    sources: list[EvidenceRef] = field(default_factory=list)
    validity: ValidityWindow = field(default_factory=ValidityWindow)
    relations: list[BeliefRelationRecord] = field(default_factory=list)
    update_policy: dict[str, Any] = field(default_factory=dict)
    formed_in: SituationRef = field(
        default_factory=lambda: situation_ref(SituationId("situation:unknown"))
    )
    holder_role: Role = Role("agent")
    action_orientation: list[ActionHint] = field(default_factory=list)
    held_since: Instant = Instant("")
    derivation: DerivationTrace | None = None
    feedback_history: list[FeedbackEntry] = field(default_factory=list)
    held_until: Instant | None = None
    superseded_by: BeliefRef | None = None
    supersedes: BeliefRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_kind", MemoryKind(self.memory_kind))
        _validate_common(self)

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        if record.get("summary_kind") is not None:
            raise ValueError("atomic belief cannot include summary_kind")
        if "memory_kind" not in record or record.get("memory_kind") is None:
            raise ValueError("atomic belief requires memory_kind")
        return dataclass_from_record(cls, record)


@dataclass(frozen=True)
class SummaryBelief:
    """Immutable summary over a bounded set of belief entities."""

    id: BeliefId
    subject: SubjectRef
    about: list[Reference]
    object: str
    content: NLStatement
    summary_kind: SummaryKind
    derivation_stage: DerivationStage
    scope: BeliefScope
    authority: Authority
    lifecycle: BeliefLifecycle = BeliefLifecycle.ACTIVE
    structure: dict[str, Any] | None = None
    sources: list[EvidenceRef] = field(default_factory=list)
    validity: ValidityWindow = field(default_factory=ValidityWindow)
    relations: list[BeliefRelationRecord] = field(default_factory=list)
    update_policy: dict[str, Any] = field(default_factory=dict)
    source_belief_ids: list[BeliefId] = field(default_factory=list)
    formed_in: SituationRef = field(
        default_factory=lambda: situation_ref(SituationId("situation:unknown"))
    )
    holder_role: Role = Role("agent")
    action_orientation: list[ActionHint] = field(default_factory=list)
    held_since: Instant = Instant("")
    derivation: DerivationTrace | None = None
    feedback_history: list[FeedbackEntry] = field(default_factory=list)
    held_until: Instant | None = None
    superseded_by: BeliefRef | None = None
    supersedes: BeliefRef | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary_kind", SummaryKind(self.summary_kind))
        _validate_common(self)

    def to_record(self) -> dict[str, Any]:
        return dataclass_to_record(self)

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Self:
        if record.get("memory_kind") is not None:
            raise ValueError("summary belief cannot include memory_kind")
        if "summary_kind" not in record or record.get("summary_kind") is None:
            raise ValueError("summary belief requires summary_kind")
        return dataclass_from_record(cls, record)


Belief = AtomicBelief
BeliefRecord = AtomicBelief | SummaryBelief


def _validate_common(belief: AtomicBelief | SummaryBelief) -> None:
    object.__setattr__(belief, "derivation_stage", DerivationStage(belief.derivation_stage))
    object.__setattr__(belief, "scope", BeliefScope(belief.scope))
    object.__setattr__(belief, "authority", Authority(belief.authority))
    object.__setattr__(belief, "lifecycle", BeliefLifecycle(belief.lifecycle))
    _validate_reference(belief.subject, "belief subject")
    if not isinstance(belief.validity, ValidityWindow):
        raise TypeError("belief validity must be a ValidityWindow")
    _validate_scope_about(belief.scope, belief.about)
    for source in belief.sources:
        _validate_reference(source, "belief source")
    for relation in belief.relations:
        if not isinstance(relation, BeliefRelationRecord):
            raise TypeError("belief relation entries must be BeliefRelationRecord")
        _validate_reference(relation.target, "belief relation target")


def _validate_scope_about(scope: BeliefScope, about: list[Reference]) -> None:
    for ref in about:
        _validate_reference(ref, "belief about entry")
    expected_kinds = _SCOPE_REFERENCE_KINDS.get(scope)
    if expected_kinds is None:
        return
    if any(ref.kind in expected_kinds for ref in about):
        return
    expected = ", ".join(sorted(expected_kinds))
    raise ValueError(f"{scope.value}-scoped belief requires about reference kind: {expected}")


def _validate_reference(ref: object, label: str) -> None:
    if not isinstance(ref, Reference):
        raise TypeError(f"{label} must be a Reference")
    if not isinstance(ref.kind, str) or not ref.kind.strip():
        raise ValueError(f"{label} kind must be non-empty")
    if not isinstance(ref.id, str) or not ref.id.strip():
        raise ValueError(f"{label} id must be non-empty")


def unknown_subject_ref() -> SubjectRef:
    return subject_ref(SubjectId("subject:self"))
