from __future__ import annotations

from alpha_agent.cognition.render import (
    RenderBudget,
    TextChatRenderer,
    conversation_message_to_chat,
    wrap_system_reminder,
)
from alpha_agent.state.models import ConversationMessage
from tests.cognition.render_helpers import view
from tests.cognition.test_belief_projection_apply import belief


def _message(
    *,
    message_id: str,
    ordinal: int,
    role: str,
    content: str,
    tool_call_id: str | None = None,
    tool_calls: list[dict] | None = None,
) -> ConversationMessage:
    return ConversationMessage(
        id=message_id,
        session_id="s1",
        ordinal=ordinal,
        role=role,  # type: ignore[arg-type]
        raw_content=content,
        model_content=None,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls or [],
        tool_result_id=None,
        provider_metadata={},
        source_metadata={},
        created_at="2026-01-01T00:00:00+00:00",
    )


def test_renderer_orders_system_sections_and_current_user_message() -> None:
    rendered = TextChatRenderer().render(
        view(
            recalled_beliefs=[belief("belief:1", "User prefers Python.")],
            current_query="what now?",
        ),
        RenderBudget(),
    )

    messages = rendered.payload
    assert [message["role"] for message in messages] == ["system", "system", "system", "user"]
    assert "Identity: Alpha Agent" in messages[0]["content"]
    assert "Recalled beliefs:" in messages[1]["content"]
    assert "Foreground:" in messages[2]["content"]
    assert messages[-1]["content"] == "what now?"


def test_renderer_preserves_history_and_duplicate_current_input() -> None:
    history = [
        conversation_message_to_chat(
            _message(message_id="msg_1", ordinal=1, role="user", content="hello")
        )
    ]

    rendered = TextChatRenderer().render(
        view(chat_history=history, current_query="hello"),
        RenderBudget(),
    )

    messages = rendered.payload
    assert [message["role"] for message in messages[:2]] == ["system", "user"]
    assert messages[1]["content"] == "hello"
    assert messages[-1]["content"] == "hello"


def test_conversation_tool_round_converts_to_chat_messages() -> None:
    assistant = _message(
        message_id="msg_1",
        ordinal=1,
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    )
    tool = _message(
        message_id="msg_2",
        ordinal=2,
        role="tool",
        content='{"ok": true}',
        tool_call_id="call_1",
    )

    assert conversation_message_to_chat(assistant)["tool_calls"][0]["id"] == "call_1"
    assert conversation_message_to_chat(tool) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }


def test_budget_drops_zero_budget_section() -> None:
    rendered = TextChatRenderer().render(
        view(recalled_beliefs=[belief("belief:1", "User prefers Python.")]),
        RenderBudget(per_section_tokens={"recalled_beliefs": 0}),
    )

    assert "recalled_beliefs" in rendered.dropped_sections
    contents = "\n".join(str(message.get("content", "")) for message in rendered.payload)
    assert "Recalled beliefs:" not in contents


def test_wrap_system_reminder() -> None:
    assert wrap_system_reminder("hello") == "<system-reminder>\nhello\n</system-reminder>"
