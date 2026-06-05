from __future__ import annotations

from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.state.store import StateStore
from tests.cognition.test_belief_projection_apply import belief, counterpart_a, counterpart_b


def test_recall_scopes_to_current_counterpart_and_global_beliefs(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    projection = BeliefProjection(store)
    for item in [
        belief("belief:a", "User A prefers Python.", about=[counterpart_a()]),
        belief("belief:b", "User B prefers Go.", about=[counterpart_b()], object_="go"),
        belief("belief:global", "Python is a programming language.", about=[]),
    ]:
        projection.upsert_atomic(item)

    recalled = projection.recall(
        BeliefRecallParams(
            counterpart=counterpart_a(),
        )
    )

    assert [item.id for item in recalled] == ["belief:a", "belief:global"]
