from alpha_agent.cognition.models import (
    Authority,
    Belief,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    CounterpartId,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Role,
    SituationId,
    ValidityWindow,
    counterpart_ref,
    entity_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF


def test_belief_about_field_round_trips_reference_order() -> None:
    about = [counterpart_ref(CounterpartId("counterpart:user-a")), entity_ref("repo:alpha-agent")]
    belief = Belief(
        id=BeliefId("belief:1"),
        subject=subject_ref(SUBJECT_SELF),
        about=about,
        object="user preference",
        content=NLStatement("User A prefers concise answers."),
        memory_kind=MemoryKind.PREFERENCE,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope.COUNTERPART,
        authority=Authority.USER_ASSERTED,
        structure=None,
        sources=[],
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        relations=[],
        formed_in=situation_ref(SituationId("situation:1")),
        holder_role=Role("agent"),
        action_orientation=[],
        update_policy={"updates": "revise_on_conflict"},
        lifecycle=BeliefLifecycle.ACTIVE,
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )

    round_tripped = Belief.from_record(belief.to_record())

    assert round_tripped == belief
    assert round_tripped.about == about
