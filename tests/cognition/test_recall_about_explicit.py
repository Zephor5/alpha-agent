from __future__ import annotations

from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.state.store import StateStore
from tests.cognition.test_belief_projection_apply import belief, counterpart_a, counterpart_b


def test_recall_about_returns_active_beliefs_for_explicit_ref_without_entity_filter(
    tmp_path,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    projection = BeliefProjection(store)
    for item in [
        belief(
            "belief:a-python",
            "User A prefers Python.",
            about=[counterpart_a()],
            object_="python",
        ),
        belief("belief:a-rust", "User A prefers Rust.", about=[counterpart_a()], object_="rust"),
        belief("belief:b-go", "User B prefers Go.", about=[counterpart_b()], object_="go"),
    ]:
        projection.upsert_atomic(item)

    recalled = projection.recall_about(counterpart_a())

    assert [item.id for item in recalled] == ["belief:a-python", "belief:a-rust"]
