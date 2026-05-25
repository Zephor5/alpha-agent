from dataclasses import FrozenInstanceError, is_dataclass

import pytest

from alpha_agent.cognition.models import (
    AuthorityHint,
    Belief,
    CognitiveEvent,
    ContextWindow,
    Counterpart,
    Decision,
    Judgment,
    Perception,
    Procedure,
    Reference,
    Reflection,
    Relationship,
    ServiceCommitment,
    Situation,
    SocialContext,
    Stimulus,
    Subject,
    ThreadId,
    ValueLens,
    ValueProfile,
)


@pytest.mark.parametrize(
    "model_cls",
    [
        Reference,
        AuthorityHint,
        Subject,
        Counterpart,
        ServiceCommitment,
        Relationship,
        Belief,
        Situation,
        SocialContext,
        Stimulus,
        Perception,
        Judgment,
        Decision,
        Reflection,
        Procedure,
        ContextWindow,
        ValueProfile,
        ValueLens,
        CognitiveEvent,
        ThreadId,
    ],
)
def test_model_types_are_frozen_dataclasses(model_cls) -> None:
    assert is_dataclass(model_cls)
    assert model_cls.__dataclass_params__.frozen is True
    assert getattr(model_cls, "__hash__", None) is not None


def test_frozen_instances_reject_mutation() -> None:
    subject = Subject()
    with pytest.raises(FrozenInstanceError):
        subject.role = "other"
