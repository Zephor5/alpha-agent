from alpha_agent.cognition.models import (
    Applicability,
    Belief,
    BeliefId,
    CognitiveType,
    Lifecycle,
    NLStatement,
    Role,
    SituationId,
    UpdatePolicy,
    ValueProfile,
    counterpart_ref,
    entity_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_belief_about_field_round_trips_reference_order() -> None:
    about = [counterpart_ref("counterpart:user-a"), entity_ref("repo:alpha-agent")]
    belief = Belief(
        id=BeliefId("belief:1"),
        subject=subject_ref(SUBJECT_SELF),
        about=about,
        object="user preference",
        content=NLStatement("User A prefers concise answers."),
        cognitive_type=CognitiveType.PREFERENCE,
        structure=None,
        sources=[],
        confidence=0.8,
        applicability=Applicability("always"),
        value_profile=ValueProfile(),
        relations=[],
        formed_in=situation_ref(SituationId("situation:1")),
        holder_role=Role("agent"),
        action_orientation=[],
        update_policy=UpdatePolicy("revise_on_conflict"),
        status=Lifecycle("active"),
        held_since="2026-01-01T00:00:00+00:00",
    )

    round_tripped = Belief.from_record(belief.to_record())

    assert round_tripped == belief
    assert round_tripped.about == about
