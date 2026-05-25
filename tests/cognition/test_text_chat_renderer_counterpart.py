from __future__ import annotations

from alpha_agent.cognition.models import CounterpartRole
from alpha_agent.cognition.render import RenderBudget, TextChatRenderer
from tests.cognition.render_helpers import counterpart, view
from tests.cognition.test_belief_projection_apply import belief


def test_counterpart_role_changes_system_template() -> None:
    user_prompt = TextChatRenderer().render(
        view(counterpart=counterpart(role=CounterpartRole.USER)),
        RenderBudget(),
    ).payload[0]["content"]
    operator_prompt = TextChatRenderer().render(
        view(counterpart=counterpart(role=CounterpartRole.OPERATOR)),
        RenderBudget(),
    ).payload[0]["content"]

    assert user_prompt != operator_prompt
    assert "protocol-oriented" in operator_prompt


def test_communication_style_hint_appears_in_system_prompt() -> None:
    prompt = TextChatRenderer().render(
        view(counterpart=counterpart(style_value="brief and direct")),
        RenderBudget(),
    ).payload[0]["content"]

    assert "Communication style:" in prompt
    assert "brief and direct" in prompt


def test_low_trust_prefixes_recalled_beliefs() -> None:
    rendered = TextChatRenderer().render(
        view(
            counterpart=counterpart(trust_level=0.2),
            recalled_beliefs=[belief("belief:1", "User prefers Python.")],
        ),
        RenderBudget(),
    )

    assert "User-reported, not verified by agent" in str(rendered.payload)


def test_no_counterpart_uses_default_template_without_style_segment() -> None:
    prompt = TextChatRenderer().render(view(counterpart=None), RenderBudget()).payload[0]["content"]

    assert "Identity: Alpha Agent" in prompt
    assert "Communication style:" not in prompt
