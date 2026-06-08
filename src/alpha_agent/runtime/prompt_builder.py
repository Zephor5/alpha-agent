"""Shared answer-path prompt construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from alpha_agent.llm.base import ChatMessage
from alpha_agent.runtime.chat_messages import (
    render_counterpart_profile,
    wrap_system_reminder,
)
from alpha_agent.state.models import SessionProfileSnapshot

_DEFAULT_RUNTIME_SYSTEM_MESSAGE: ChatMessage = {
    "role": "system",
    "content": (
        "Identity: Alpha Agent.\n"
        "Use the current turn and session context and answer concisely. "
        "Call tools only when they are useful. "
        "When stable counterpart profile context is present, it is already visible near "
        "the start of the prompt. Use memory_recall for explicit long-term belief "
        "lookups, and use memory_propose only for explicit long-term memory updates."
    ),
}


@dataclass(frozen=True)
class AnswerPromptFrame:
    """Stable answer prompt prefix shared by runtime and debug rendering."""

    system_message: ChatMessage
    profile_context_messages: list[ChatMessage] = field(default_factory=list)


def copy_chat_message(message: ChatMessage) -> ChatMessage:
    """Return a shallow copy of a chat message mapping."""

    return cast(ChatMessage, dict(message))


def default_runtime_system_message() -> ChatMessage:
    """Return the system message used by the real runtime answer path."""

    return copy_chat_message(_DEFAULT_RUNTIME_SYSTEM_MESSAGE)


def build_answer_prompt_frame(
    *,
    profile_snapshot: SessionProfileSnapshot | None,
    system_message: ChatMessage | None = None,
) -> AnswerPromptFrame:
    """Build the non-history prefix for an answer prompt."""

    return AnswerPromptFrame(
        system_message=copy_chat_message(
            system_message if system_message is not None else _DEFAULT_RUNTIME_SYSTEM_MESSAGE
        ),
        profile_context_messages=build_profile_context_messages(profile_snapshot),
    )


def build_profile_context_messages(
    snapshot: SessionProfileSnapshot | None,
) -> list[ChatMessage]:
    """Render the stable session profile snapshot, when one exists."""

    if snapshot is None:
        return []
    return [
        {
            "role": "user",
            "content": wrap_system_reminder(render_counterpart_profile(snapshot.content)),
        }
    ]


def build_answer_prompt_messages(
    *,
    profile_snapshot: SessionProfileSnapshot | None,
    session_history: Sequence[ChatMessage],
    current_turn_messages: Sequence[ChatMessage] = (),
    system_message: ChatMessage | None = None,
) -> list[ChatMessage]:
    """Build the complete answer prompt from visible answer-path inputs."""

    return build_answer_prompt_messages_from_frame(
        frame=build_answer_prompt_frame(
            profile_snapshot=profile_snapshot,
            system_message=system_message,
        ),
        session_history=session_history,
        current_turn_messages=current_turn_messages,
    )


def build_answer_prompt_messages_from_frame(
    *,
    frame: AnswerPromptFrame,
    session_history: Sequence[ChatMessage],
    current_turn_messages: Sequence[ChatMessage] = (),
) -> list[ChatMessage]:
    """Compose system, profile snapshot, session history, and current-turn messages."""

    return [
        copy_chat_message(frame.system_message),
        *[copy_chat_message(message) for message in frame.profile_context_messages],
        *[copy_chat_message(message) for message in session_history],
        *[copy_chat_message(message) for message in current_turn_messages],
    ]
