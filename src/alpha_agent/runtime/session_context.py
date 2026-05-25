"""Session context projection built from append-only conversation messages."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.state.models import ConversationMessage
from alpha_agent.state.store import StateStore


@dataclass(frozen=True)
class SessionContextProjection:
    """LLM-visible session context projected from the recent transcript."""

    messages: list[ConversationMessage]
    before_ordinal: int | None = None


class SessionContextManager:
    """Load recent session context without mutating the transcript."""

    def __init__(self, store: StateStore, *, recent_tail_messages: int = 8):
        self.store = store
        self.recent_tail_messages = max(0, recent_tail_messages)

    def load(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> SessionContextProjection:
        """Load replay-safe recent transcript messages before an ordinal."""

        messages = self.store.list_conversation_messages(
            session_id,
            before_ordinal=before_ordinal,
        )
        if self.recent_tail_messages:
            messages = messages[-self.recent_tail_messages :]
        messages = self._repair_tool_replay_head(session_id, messages, before_ordinal)
        messages = self._drop_incomplete_tool_replay(messages)
        return SessionContextProjection(messages=messages, before_ordinal=before_ordinal)

    def _repair_tool_replay_head(
        self,
        session_id: str,
        messages: list[ConversationMessage],
        before_ordinal: int | None,
    ) -> list[ConversationMessage]:
        if not messages or messages[0].role != "tool":
            return messages

        assistant = self._find_assistant_for_tool_result(session_id, messages[0])
        if assistant is None:
            return self._drop_leading_tool_results(messages)

        repaired = self.store.list_conversation_messages(
            session_id,
            after_ordinal=assistant.ordinal - 1,
            before_ordinal=before_ordinal,
        )
        if self.recent_tail_messages:
            return repaired[-self.recent_tail_messages :]
        return repaired

    def _find_assistant_for_tool_result(
        self,
        session_id: str,
        tool_message: ConversationMessage,
    ) -> ConversationMessage | None:
        if not tool_message.tool_call_id:
            return None
        prior_messages = self.store.list_conversation_messages(
            session_id,
            before_ordinal=tool_message.ordinal,
        )
        for message in reversed(prior_messages):
            if (
                message.role == "assistant"
                and tool_message.tool_call_id in self._tool_call_ids(message)
            ):
                return message
        return None

    def _drop_leading_tool_results(
        self,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        index = 0
        while index < len(messages) and messages[index].role == "tool":
            index += 1
        return messages[index:]

    def _drop_incomplete_tool_replay(
        self,
        messages: list[ConversationMessage],
    ) -> list[ConversationMessage]:
        projected: list[ConversationMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "tool":
                index += 1
                continue
            if message.role != "assistant" or not message.tool_calls:
                projected.append(message)
                index += 1
                continue

            required_call_ids = self._tool_call_ids(message)
            replay_messages = [message]
            seen_call_ids: set[str] = set()
            index += 1
            while index < len(messages) and messages[index].role == "tool":
                tool_message = messages[index]
                if tool_message.tool_call_id in required_call_ids:
                    replay_messages.append(tool_message)
                    if tool_message.tool_call_id is not None:
                        seen_call_ids.add(tool_message.tool_call_id)
                    index += 1
                    continue
                break
            if seen_call_ids == required_call_ids:
                projected.extend(replay_messages)
        return projected

    def _tool_call_ids(self, message: ConversationMessage) -> set[str]:
        ids: set[str] = set()
        for tool_call in message.tool_calls:
            tool_call_id = tool_call.get("id")
            if tool_call_id is not None:
                ids.add(str(tool_call_id))
        return ids
