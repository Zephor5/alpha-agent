from dataclasses import FrozenInstanceError, is_dataclass
from typing import Any, cast

import pytest

from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    AuthorityHint,
    Belief,
    BeliefId,
    BeliefRelationKind,
    BeliefRelationRecord,
    BeliefScope,
    CognitiveEvent,
    Counterpart,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Perception,
    Reference,
    Relationship,
    Role,
    ServiceCommitment,
    Situation,
    SituationId,
    SocialContext,
    Subject,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF


@pytest.mark.parametrize(
    "model_cls",
    [
        Reference,
        AuthorityHint,
        Subject,
        Counterpart,
        ServiceCommitment,
        Relationship,
        AtomicBelief,
        Belief,
        BeliefRelationRecord,
        SummaryBelief,
        ValidityWindow,
        Situation,
        SocialContext,
        Perception,
        CognitiveEvent,
    ],
)
def test_model_types_are_frozen_dataclasses(model_cls) -> None:
    assert is_dataclass(model_cls)
    assert cast(Any, model_cls).__dataclass_params__.frozen is True
    assert getattr(model_cls, "__hash__", None) is not None


def test_frozen_instances_reject_mutation() -> None:
    subject = Subject()
    with pytest.raises(FrozenInstanceError):
        cast(Any, subject).role = "other"


def test_belief_models_reject_unstructured_sources() -> None:
    with pytest.raises(TypeError, match="source"):
        _atomic_belief(sources=[cast(Any, {"kind": "session_message", "id": "msg-1"})])

    record = _atomic_belief().to_record()
    record["sources"] = ["msg-1"]
    with pytest.raises(TypeError, match="source"):
        AtomicBelief.from_record(record)

    summary_record = _summary_belief().to_record()
    summary_record["sources"] = [{"kind": "", "id": "msg-1"}]
    with pytest.raises(ValueError, match="source.*kind"):
        SummaryBelief.from_record(summary_record)


def test_belief_models_reject_invalid_relations() -> None:
    with pytest.raises(TypeError, match="relation"):
        _atomic_belief(
            relations=[
                cast(
                    Any,
                    {
                        "kind": BeliefRelationKind.SUPPORTS.value,
                        "target": Reference("belief", "belief:source"),
                    },
                )
            ]
        )

    record = _atomic_belief().to_record()
    record["relations"] = [
        {
            "kind": BeliefRelationKind.SUPPORTS.value,
            "target": "belief:source",
        }
    ]
    with pytest.raises(TypeError, match="relation target"):
        AtomicBelief.from_record(record)

    summary_record = _summary_belief().to_record()
    summary_record["relations"] = [
        {
            "kind": "near_duplicate",
            "target": {"kind": "belief", "id": "belief:source"},
        }
    ]
    with pytest.raises(ValueError, match="near_duplicate"):
        SummaryBelief.from_record(summary_record)


def _atomic_belief(
    *,
    sources: list[Reference] | None = None,
    relations: list[BeliefRelationRecord] | None = None,
) -> AtomicBelief:
    return AtomicBelief(
        id=BeliefId("belief:test"),
        subject=subject_ref(SUBJECT_SELF),
        about=[],
        object="python",
        content=NLStatement("Python uses indentation."),
        memory_kind=MemoryKind.FACT,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope.GLOBAL,
        authority=Authority.USER_ASSERTED,
        sources=sources or [],
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        relations=relations or [],
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


def _summary_belief() -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId("belief:summary"),
        subject=subject_ref(SUBJECT_SELF),
        about=[],
        object="profile",
        content=NLStatement("Python appears in active memories."),
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=BeliefScope.GLOBAL,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )
