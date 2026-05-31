from __future__ import annotations

from typing import cast

from alpha_agent.cognition.render import (
    RenderBudget,
    TextChatRenderer,
    source_message_to_chat,
    wrap_system_reminder,
)
from alpha_agent.llm.base import AssistantChatMessage
from alpha_agent.state.models import SessionMessage
from tests.cognition.render_helpers import view


def _message(
    *,
    message_id: str,
    ordinal: int,
    role: str,
    content: str,
    reasoning_content: str | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict] | None = None,
) -> SessionMessage:
    return SessionMessage(
        id=message_id,
        session_id="s1",
        ordinal=ordinal,
        kind=f"{role}_message",  # type: ignore[arg-type]
        llm_role=role,  # type: ignore[arg-type]
        raw_content=content,
        model_content=None,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls or [],
        tool_result_id=None,
        provider_metadata={},
        source_metadata={},
        compression_point_ordinal=None,
        compression_version=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at=None,
        reasoning_content=reasoning_content,
    )


def test_renderer_orders_system_sections_and_current_user_message() -> None:
    history = [
        source_message_to_chat(
            _message(message_id="msg_1", ordinal=1, role="user", content="hello")
        )
    ]
    rendered = TextChatRenderer().render(
        view(
            counterpart_profile="User prefers Python.",
            chat_history=history,
            current_query="what now?",
        ),
        RenderBudget(),
    )

    messages = rendered.payload
    assert [message["role"] for message in messages] == ["system", "user", "user", "user"]
    assert "Identity: Alpha Agent" in messages[0]["content"]
    assert "Counterpart profile:" in messages[1]["content"]
    assert "<system-reminder>" in messages[1]["content"]
    assert "<context-reminder>" not in messages[1]["content"]
    assert messages[2]["content"] == "hello"
    assert "Foreground:" not in "\n".join(str(message["content"]) for message in messages)
    assert messages[-1]["content"] == "what now?"


def test_renderer_preserves_history_and_appends_current_input_once() -> None:
    history = [
        source_message_to_chat(
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
    assert [message["content"] for message in messages].count("hello") == 2


def test_renderer_budget_does_not_prune_source_history_messages() -> None:
    history = [
        source_message_to_chat(
            _message(
                message_id=f"msg_{index}",
                ordinal=index,
                role="user" if index % 2 else "assistant",
                content=f"source history {index} " + ("long text " * 40),
            )
        )
        for index in range(1, 7)
    ]

    rendered = TextChatRenderer().render(
        view(chat_history=history, current_query="current user source message"),
        RenderBudget(max_tokens=1),
    )

    contents = [str(message.get("content", "")) for message in rendered.payload]
    for index in range(1, 7):
        assert any(f"source history {index}" in content for content in contents)
    assert contents[-1] == "current user source message"
    assert "unknown" not in rendered.dropped_sections


def test_renderer_uses_user_role_for_non_transcript_context() -> None:
    rendered = TextChatRenderer().render(
        view(
            counterpart_profile="User prefers Python.",
            current_query="what now?",
        ),
        RenderBudget(),
    )

    context_messages = rendered.payload[1:-1]
    assert context_messages
    assert {message["role"] for message in context_messages} == {"user"}
    assert all("<system-reminder>" in str(message["content"]) for message in context_messages)
    assert all("<context-reminder>" not in str(message["content"]) for message in context_messages)


def test_source_tool_round_converts_to_chat_messages() -> None:
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

    assistant_chat = cast(AssistantChatMessage, source_message_to_chat(assistant))
    assert assistant_chat["tool_calls"][0]["id"] == "call_1"
    assert source_message_to_chat(tool) == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }


def test_source_assistant_message_preserves_reasoning_content() -> None:
    assistant = _message(
        message_id="msg_1",
        ordinal=1,
        role="assistant",
        content="answer",
        reasoning_content="thinking trace",
    )

    assert source_message_to_chat(assistant) == {
        "role": "assistant",
        "content": "answer",
        "reasoning_content": "thinking trace",
    }


def test_budget_drops_zero_budget_section() -> None:
    rendered = TextChatRenderer().render(
        view(counterpart_profile="User prefers Python."),
        RenderBudget(per_section_tokens={"counterpart_profile": 0}),
    )

    assert "counterpart_profile" in rendered.dropped_sections
    contents = "\n".join(str(message.get("content", "")) for message in rendered.payload)
    assert "Counterpart profile:" not in contents


def test_wrap_system_reminder() -> None:
    assert wrap_system_reminder("hello") == "<system-reminder>\nhello\n</system-reminder>"
