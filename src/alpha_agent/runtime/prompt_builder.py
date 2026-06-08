"""Shared answer-path prompt construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

from alpha_agent.cognition.models import SummaryKind
from alpha_agent.llm.base import ChatMessage
from alpha_agent.runtime.chat_messages import (
    render_counterpart_profile,
    render_self_memory_summary,
    wrap_system_reminder,
)
from alpha_agent.state.models import SessionSummarySnapshot

_DEFAULT_RUNTIME_SYSTEM_MESSAGE: ChatMessage = {
    "role": "system",
    "content": (
        "Identity: Alpha Agent.\n"
        "You are a direct, tool-capable personal agent. Use the current turn, session "
        "history, and stable summary context to help the user through the shortest "
        "reliable path.\n\n"
        "Context model:\n"
        "- Stable self-memory and counterpart profile reminders may appear near the "
        "start of the prompt. Treat them as stable summary context, not as the "
        "current user request and not live system state.\n"
        "- Dynamic long-term memory is not silently injected. Call memory_recall when "
        "durable facts, preferences, constraints, procedures, values, or relationships "
        "would materially improve the answer.\n"
        "- Use memory_propose only when the user explicitly provides, updates, or "
        "corrects durable long-term memory. Do not write task progress, transient "
        "outcomes, or guesses.\n\n"
        "Execution discipline:\n"
        "- Use tools when they improve correctness, grounding, or completion; do not "
        "describe tool work you can perform.\n"
        "- Resolve prerequisite lookups before acting, and verify results before "
        "finalizing when the task has observable effects.\n"
        "- If required context can be retrieved with tools, retrieve it. Ask for "
        "clarification only when the missing decision changes the action and cannot "
        "be discovered safely.\n"
        "- Keep answers concise and match the user's requested format."
    ),
}
_SUMMARY_PROMPT_ORDER = {
    SummaryKind.SELF_MEMORY_SUMMARY.value: 0,
    SummaryKind.COUNTERPART_PROFILE.value: 1,
}


@dataclass(frozen=True)
class AnswerPromptFrame:
    """Stable answer prompt prefix shared by runtime and debug rendering."""

    system_message: ChatMessage
    summary_context_messages: list[ChatMessage] = field(default_factory=list)


def copy_chat_message(message: ChatMessage) -> ChatMessage:
    """Return a shallow copy of a chat message mapping."""

    return cast(ChatMessage, dict(message))


def default_runtime_system_message() -> ChatMessage:
    """Return the system message used by the real runtime answer path."""

    return copy_chat_message(_DEFAULT_RUNTIME_SYSTEM_MESSAGE)


def build_answer_prompt_frame(
    *,
    summary_snapshots: Sequence[SessionSummarySnapshot] = (),
    system_message: ChatMessage | None = None,
) -> AnswerPromptFrame:
    """Build the non-history prefix for an answer prompt."""

    return AnswerPromptFrame(
        system_message=copy_chat_message(
            system_message if system_message is not None else _DEFAULT_RUNTIME_SYSTEM_MESSAGE
        ),
        summary_context_messages=build_summary_context_messages(summary_snapshots),
    )


def build_summary_context_messages(
    snapshots: Sequence[SessionSummarySnapshot],
) -> list[ChatMessage]:
    """Render stable session summary snapshots as separate prompt messages."""

    messages: list[ChatMessage] = []
    ordered = sorted(
        snapshots,
        key=lambda snapshot: (
            _SUMMARY_PROMPT_ORDER.get(snapshot.summary_kind, len(_SUMMARY_PROMPT_ORDER)),
            snapshot.summary_kind,
        ),
    )
    for snapshot in ordered:
        rendered = _render_summary_snapshot(snapshot)
        if rendered is None:
            continue
        messages.append({"role": "user", "content": wrap_system_reminder(rendered)})
    return messages


def _render_summary_snapshot(snapshot: SessionSummarySnapshot) -> str | None:
    if snapshot.summary_kind == SummaryKind.SELF_MEMORY_SUMMARY.value:
        return render_self_memory_summary(snapshot.content)
    if snapshot.summary_kind == SummaryKind.COUNTERPART_PROFILE.value:
        return render_counterpart_profile(snapshot.content)
    return None


def build_answer_prompt_messages(
    *,
    summary_snapshots: Sequence[SessionSummarySnapshot] = (),
    session_history: Sequence[ChatMessage],
    current_turn_messages: Sequence[ChatMessage] = (),
    system_message: ChatMessage | None = None,
) -> list[ChatMessage]:
    """Build the complete answer prompt from visible answer-path inputs."""

    return build_answer_prompt_messages_from_frame(
        frame=build_answer_prompt_frame(
            summary_snapshots=summary_snapshots,
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
    """Compose system, summary snapshots, session history, and current-turn messages."""

    return [
        copy_chat_message(frame.system_message),
        *[copy_chat_message(message) for message in frame.summary_context_messages],
        *[copy_chat_message(message) for message in session_history],
        *[copy_chat_message(message) for message in current_turn_messages],
    ]
