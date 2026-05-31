"""Chat-completions renderer for reactive cognition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView
from alpha_agent.llm.base import ChatCompletionToolCall, ChatMessage, LLMToolDefinitionInput
from alpha_agent.runtime.context_budget import estimate_context_budget
from alpha_agent.state.models import SessionMessage

SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"
COUNTERPART_PROFILE_LABEL = "Counterpart profile:"


def wrap_system_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content.strip()}\n{SYSTEM_REMINDER_CLOSE}"


def render_counterpart_profile(content: str) -> str:
    return f"{COUNTERPART_PROFILE_LABEL} {content}"


def estimate_chat_tokens(
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
) -> int:
    estimate = estimate_context_budget(messages, tools=tools, max_context_tokens=0)
    return estimate.message_tokens + estimate.tool_schema_tokens


def source_message_to_chat(message: SessionMessage) -> ChatMessage:
    """Convert a durable source message into a replayable chat message."""

    content = message.model_content if message.model_content is not None else message.raw_content
    if message.kind == "compressed_message":
        raise ValueError("compressed_message must be projected by SessionContextAssembler")
    if message.llm_role == "user":
        return {"role": "user", "content": content}
    if message.llm_role == "assistant":
        assistant_content = (content or None) if message.tool_calls else content
        assistant_message: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_content,
        }
        if message.reasoning_content is not None:
            assistant_message["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            assistant_message["tool_calls"] = [
                _source_tool_call(tool_call) for tool_call in message.tool_calls
            ]
        return cast(ChatMessage, assistant_message)
    if message.llm_role != "tool":
        raise ValueError(f"session message {message.id!r} is missing llm_role")
    if not message.tool_call_id:
        raise ValueError(f"tool session message {message.id!r} is missing tool_call_id")
    return {"role": "tool", "tool_call_id": message.tool_call_id, "content": content}


class TextChatRenderer:
    """Render a cognition view into OpenAI-style chat messages."""

    name = "text_chat"

    def render(self, view: CognitionView, budget: RenderBudget) -> RenderResult:
        dropped: list[str] = []
        messages: list[ChatMessage] = [
            {"role": "system", "content": self._system_prompt(view, budget)}
        ]
        context_sections: list[tuple[int, str]] = []
        self._append_section(
            messages,
            context_sections,
            dropped,
            "counterpart_profile",
            self._counterpart_profile(view),
            budget,
            droppable=False,
        )
        messages.extend(view.chat_history)
        sections = [
            ("strategy_reminders", self._strategy_reminders(view)),
            ("background", self._background(view)),
        ]
        for section, content in sections:
            self._append_section(
                messages,
                context_sections,
                dropped,
                section,
                content,
                budget,
            )
        messages.append({"role": "user", "content": self._current_query(view)})
        used_tokens = estimate_chat_tokens(messages)
        while used_tokens > budget.max_tokens and context_sections:
            index, section = context_sections.pop()
            messages.pop(index)
            dropped.append(section)
            context_sections = [
                (item_index - 1 if item_index > index else item_index, item_section)
                for item_index, item_section in context_sections
            ]
            used_tokens = estimate_chat_tokens(messages)
        return RenderResult(
            payload=messages,
            used_tokens=used_tokens,
            dropped_sections=dropped,
        )

    def _system_prompt(self, view: CognitionView, budget: RenderBudget) -> str:
        parts = [_SYSTEM_PROMPT]
        hints = _style_hints(view, budget)
        if hints:
            parts.append("Communication style: " + "; ".join(hints))
        return "\n\n".join(parts)

    def _strategy_reminders(self, view: CognitionView) -> str:
        if not view.active_strategies:
            return ""
        return "Strategy reminders:\n" + "\n".join(f"- {item}" for item in view.active_strategies)

    def _counterpart_profile(self, view: CognitionView) -> str:
        if not view.counterpart_profile:
            return ""
        return render_counterpart_profile(view.counterpart_profile)

    def _background(self, view: CognitionView) -> str:
        return f"Context background: {view.window.background}" if view.window.background else ""

    def _current_query(self, view: CognitionView) -> str:
        if view.current_query:
            return view.current_query
        for perception in reversed(view.window.foreground):
            if perception.raw is not None:
                return str(perception.raw)
        return ""

    def _append_section(
        self,
        messages: list[ChatMessage],
        context_sections: list[tuple[int, str]],
        dropped: list[str],
        section: str,
        content: str,
        budget: RenderBudget,
        *,
        droppable: bool = True,
    ) -> None:
        if not content:
            return
        clipped = _clip_to_budget(content, budget.per_section_tokens.get(section))
        if clipped is None:
            dropped.append(section)
            return
        if droppable:
            context_sections.append((len(messages), section))
        messages.append({"role": "user", "content": wrap_system_reminder(clipped)})


_SYSTEM_PROMPT = (
    "Identity: Alpha Agent.\n"
    "Use the current reactive context and answer concisely. "
    "Call tools only when they are useful.\n"
    "When counterpart profile context is present, treat it as already-visible stable "
    "relationship context near the start of the prompt. Use memory_recall for explicit "
    "long-term belief lookup when details are needed beyond the visible context.\n"
    "Use memory_propose only for explicit long-term user preferences, stable "
    "constraints, reusable procedures, or direct corrections to remembered cognition. "
    "Do not call memory_propose for ordinary facts, transient session context, or guesses. "
    "Corrections are proposals only; do not claim memory was changed unless asked."
)


def _style_hints(view: CognitionView, budget: RenderBudget) -> list[str]:
    hints = (
        [f"{hint.kind}={hint.value}" for hint in view.counterpart.communication_style]
        if view.counterpart
        else []
    )
    hints.extend(f"{key}={value}" for key, value in budget.style_hints.items())
    return hints


def _clip_to_budget(content: str, token_budget: int | None) -> str | None:
    if token_budget is None:
        return content
    if token_budget <= 0:
        return None
    max_chars = token_budget * 4
    if len(content) <= max_chars:
        return content
    return content[: max(0, max_chars - 3)].rstrip() + "..."


def _source_tool_call(payload: Mapping[str, Any]) -> ChatCompletionToolCall:
    function = payload.get("function")
    if not isinstance(function, Mapping):
        function = {}
    return {
        "id": str(payload.get("id") or ""),
        "type": "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": str(function.get("arguments") or "{}"),
        },
    }
