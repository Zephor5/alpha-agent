from __future__ import annotations

from alpha_agent.runtime.session_context import SessionContextManager
from alpha_agent.state.store import StateStore


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
