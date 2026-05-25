from __future__ import annotations

from alpha_agent.runtime.prompt_builder import PromptBuilder, wrap_system_reminder
from alpha_agent.runtime.session_context import SessionContextManager
from alpha_agent.state.models import ConversationMessage
from alpha_agent.state.store import StateStore


def _message(
    *,
    message_id: str,
    session_id: str = "s1",
    ordinal: int,
    role: str,
    content: str,
    tool_call_id: str | None = None,
    tool_calls: list[dict] | None = None,
) -> ConversationMessage:
    return ConversationMessage(
        id=message_id,
        session_id=session_id,
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


def test_prompt_contains_system_recent_transcript_and_current_user_message() -> None:
    builder = PromptBuilder()
    history = [
        _message(message_id="msg_1", ordinal=1, role="user", content="hello"),
        _message(message_id="msg_2", ordinal=2, role="assistant", content="hi"),
    ]

    messages = builder.build("what now?", history)

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert "Reactive cognition is active" in messages[0]["content"]
    assert messages[1]["content"] == "hello"
    assert messages[2]["content"] == "hi"
    assert messages[-1]["content"] == "what now?"


def test_prompt_preserves_prior_user_message_even_when_text_matches_current_input() -> None:
    builder = PromptBuilder()
    history = [_message(message_id="msg_1", ordinal=1, role="user", content="hello")]

    messages = builder.build("hello", history)

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    assert messages[1]["content"] == "hello"
    assert messages[-1]["content"] == "hello"


def test_prompt_replays_tool_call_round() -> None:
    builder = PromptBuilder()
    history = [
        _message(
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
        ),
        _message(
            message_id="msg_2",
            ordinal=2,
            role="tool",
            content='{"ok": true}',
            tool_call_id="call_1",
        ),
    ]

    messages = builder.build("done?", history)

    assert messages[1]["role"] == "assistant"
    assert messages[1]["tool_calls"][0]["id"] == "call_1"  # type: ignore[index]
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"ok": true}',
    }


def test_wrap_system_reminder() -> None:
    assert wrap_system_reminder("hello") == "<system-reminder>\nhello\n</system-reminder>"


def test_session_context_loads_recent_tail_and_preserves_tool_pairs(tmp_path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    for role, content in [
        ("user", "one"),
        ("assistant", "two"),
        ("user", "three"),
    ]:
        store.append_conversation_message(session_id="s1", role=role, raw_content=content)
    assistant = store.append_conversation_message(
        session_id="s1",
        role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    )
    store.append_conversation_message(
        session_id="s1",
        role="tool",
        raw_content='{"ok": true}',
        tool_call_id="call_1",
    )

    context = SessionContextManager(store, recent_tail_messages=2).load("s1")

    assert [message.id for message in context.messages] == [
        assistant.id,
        store.list_conversation_messages("s1")[-1].id,
    ]
