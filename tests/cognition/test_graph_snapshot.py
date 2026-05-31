from __future__ import annotations

from alpha_agent.cognition.render import GraphSnapshotRenderer, RenderBudget
from tests.cognition.render_helpers import view
from tests.cognition.test_belief_projection_apply import belief, counterpart_a


def test_graph_snapshot_renders_mermaid() -> None:
    rendered = GraphSnapshotRenderer(format="mermaid").render(
        view(),
        RenderBudget(),
        beliefs=[belief("belief:1", "User prefers Python.", about=[counterpart_a()])],
    )

    assert str(rendered.payload).startswith("graph TD")
    assert "belief_1" in rendered.payload
    assert "counterpart:user-a" in rendered.payload


def test_graph_snapshot_renders_dot() -> None:
    rendered = GraphSnapshotRenderer(format="dot").render(
        view(),
        RenderBudget(),
        beliefs=[belief("belief:1", "User prefers Python.")],
    )

    assert str(rendered.payload).startswith("digraph cognition {")
    assert '"subject:agent:self" -> "belief:belief:1";' in rendered.payload
