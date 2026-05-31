from __future__ import annotations

from alpha_agent.cognition.models import CounterpartRole
from alpha_agent.cognition.render import RenderBudget, TextChatRenderer
from tests.cognition.render_helpers import counterpart, view


def test_counterpart_role_does_not_change_system_prompt() -> None:
    user_prompt = TextChatRenderer().render(
        view(counterpart=counterpart(role=CounterpartRole.USER)),
        RenderBudget(),
    ).payload[0]["content"]
    operator_prompt = TextChatRenderer().render(
        view(counterpart=counterpart(role=CounterpartRole.OPERATOR)),
        RenderBudget(),
    ).payload[0]["content"]

    assert user_prompt == operator_prompt
    assert "Identity: Alpha Agent" in user_prompt


def test_communication_style_hint_appears_in_system_prompt() -> None:
    prompt = TextChatRenderer().render(
        view(counterpart=counterpart(style_value="brief and direct")),
        RenderBudget(),
    ).payload[0]["content"]

    assert "Communication style:" in prompt
    assert "brief and direct" in prompt


def test_counterpart_profile_renders_before_chat_history() -> None:
    rendered = TextChatRenderer().render(
        view(
            counterpart=counterpart(trust_level=0.2),
            counterpart_profile="User prefers Python.",
            chat_history=[{"role": "assistant", "content": "previous answer"}],
            current_query="next question",
        ),
        RenderBudget(),
    )

    assert "Counterpart profile: User prefers Python." in str(rendered.payload[1]["content"])
    assert rendered.payload[2]["content"] == "previous answer"
    assert "User-reported, not verified by agent" not in str(rendered.payload)


def test_counterpart_profile_is_not_pruned_by_over_budget_history() -> None:
    rendered = TextChatRenderer().render(
        view(
            counterpart_profile="Stable profile.",
            chat_history=[
                {"role": "user", "content": f"old source {index} " + ("long text " * 20)}
                for index in range(8)
            ],
            current_query="next question",
        ),
        RenderBudget(max_tokens=1),
    )

    assert "counterpart_profile" not in rendered.dropped_sections
    assert "Counterpart profile: Stable profile." in str(rendered.payload[1]["content"])
    assert rendered.payload[-1]["content"] == "next question"


def test_no_counterpart_uses_default_template_without_style_segment() -> None:
    prompt = TextChatRenderer().render(view(counterpart=None), RenderBudget()).payload[0]["content"]

    assert "Identity: Alpha Agent" in prompt
    assert "Communication style:" not in prompt
