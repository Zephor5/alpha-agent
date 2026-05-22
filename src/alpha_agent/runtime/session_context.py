"""Session context projection built from append-only conversation messages."""

from __future__ import annotations

from dataclasses import dataclass

from alpha_agent.memory.models import ConversationMessage, SessionContextState
from alpha_agent.memory.store import MemoryStore


@dataclass(frozen=True)
class SessionContextProjection:
    """LLM-visible session context projected from durable transcript state."""

    state: SessionContextState | None
    uncompressed_messages: list[ConversationMessage]
    before_ordinal: int | None = None

    @property
    def compressed_until_ordinal(self) -> int:
        """Return the active compression boundary, or zero for uncompressed sessions."""

        return self.state.compressed_until_ordinal if self.state is not None else 0

    @property
    def summary(self) -> str:
        """Return the active compressed summary text, if any."""

        if self.state is None:
            return ""
        return self.state.summary.strip()


class SessionContextManager:
    """Load stable session context without mutating the original transcript."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def load(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> SessionContextProjection:
        """Load compressed state and uncompressed transcript before an ordinal."""

        state = self.store.get_session_context_state(session_id)
        compressed_until = state.compressed_until_ordinal if state is not None else 0
        if before_ordinal is not None and compressed_until >= before_ordinal:
            raise ValueError(
                "session context compressed_until_ordinal must be lower than "
                f"before_ordinal for session {session_id!r}: "
                f"{compressed_until} >= {before_ordinal}"
            )
        messages = self.store.list_conversation_messages(
            session_id,
            after_ordinal=compressed_until,
            before_ordinal=before_ordinal,
        )
        messages = self._repair_tool_replay_head(
            session_id=session_id,
            messages=messages,
            before_ordinal=before_ordinal,
        )
        messages = self._drop_incomplete_tool_replay(messages)
        return SessionContextProjection(
            state=state,
            uncompressed_messages=messages,
            before_ordinal=before_ordinal,
        )

    def _repair_tool_replay_head(
        self,
        *,
        session_id: str,
        messages: list[ConversationMessage],
        before_ordinal: int | None,
    ) -> list[ConversationMessage]:
        if not messages or messages[0].role != "tool":
            return messages

        assistant = self._find_assistant_for_tool_result(session_id, messages[0])
        if assistant is None:
            return self._drop_leading_tool_results(messages)

        return self.store.list_conversation_messages(
            session_id,
            after_ordinal=assistant.ordinal - 1,
            before_ordinal=before_ordinal,
        )

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
            if message.role != "assistant" or not message.tool_calls:
                continue
            if tool_message.tool_call_id in self._tool_call_ids(message):
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
