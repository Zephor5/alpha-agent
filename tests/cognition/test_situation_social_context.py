from alpha_agent.cognition.models import (
    AuthorityHint,
    CounterpartId,
    Situation,
    SituationId,
    SocialContext,
    counterpart_ref,
)


def test_social_context_authority_hints_round_trip_counterpart_refs() -> None:
    counterpart = counterpart_ref(CounterpartId("counterpart:user-a"))
    situation = Situation(
        id=SituationId("situation:1"),
        social=SocialContext(
            present_counterparts=[counterpart],
            authority_hints=[AuthorityHint(counterpart=counterpart, authority="operator")],
            group_dynamics=["private-chat"],
        ),
    )

    round_tripped = Situation.from_record(situation.to_record())

    assert round_tripped == situation
    assert round_tripped.social.authority_hints[0].counterpart == counterpart
    assert type(round_tripped.social.authority_hints[0].counterpart) is type(counterpart)
