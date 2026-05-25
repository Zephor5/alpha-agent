from __future__ import annotations

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind, entity_ref
from alpha_agent.cognition.projections.belief import BeliefProjection, BeliefRecallParams
from alpha_agent.cognition.stages.types import AttentionFocus
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import clock_factory, emit, id_factory
from tests.cognition.test_belief_projection_apply import belief, counterpart_a, python_entity


def test_recall_with_focus_entities_requires_entity_overlap(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    projection = BeliefProjection(store)
    event_ids = id_factory()
    clock = clock_factory()
    for item in [
        belief("belief:python", "User A prefers Python.", about=[counterpart_a()]),
        belief("belief:rust", "User A prefers Rust.", about=[counterpart_a()], object_="rust"),
        belief(
            "belief:global-python",
            "Python uses indentation.",
            about=[],
            object_="python",
        ),
    ]:
        projection.apply(
            emit(
                log,
                CognitiveEventKind.BELIEF_FORMED,
                payload={"belief": item.to_record()},
                event_ids=event_ids,
                clock=clock,
            )
        )

    recalled = projection.recall(
        BeliefRecallParams(
            focus=AttentionFocus(
                entities=[python_entity(), entity_ref("unrelated")],
                salient_claims=[],
                value_signals={},
            ),
            counterpart=counterpart_a(),
        )
    )

    assert [item.id for item in recalled] == ["belief:python", "belief:global-python"]
