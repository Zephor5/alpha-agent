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


def wrap_system_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content.strip()}\n{SYSTEM_REMINDER_CLOSE}"


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
        messages.extend(view.chat_history)
        first_reminder_index = len(messages)
        sections = [
            ("strategy_reminders", self._strategy_reminders(view)),
            ("counterpart_digest", self._counterpart_digest(view)),
            ("recalled_beliefs", self._recalled_beliefs(view)),
            ("background", self._background(view)),
        ]
        for section, content in sections:
            if not content:
                continue
            clipped = _clip_to_budget(content, budget.per_section_tokens.get(section))
            if clipped is None:
                dropped.append(section)
                continue
            messages.append({"role": "user", "content": wrap_system_reminder(clipped)})
        messages.append({"role": "user", "content": self._current_query(view)})
        used_tokens = estimate_chat_tokens(messages)
        while used_tokens > budget.max_tokens and len(messages) - 1 > first_reminder_index:
            removed = messages.pop(-2)
            dropped.append(_section_name_from_message(removed))
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

    def _counterpart_digest(self, view: CognitionView) -> str:
        if view.counterpart_digest is None:
            return ""
        return f"Counterpart digest: {view.counterpart_digest.content}"

    def _recalled_beliefs(self, view: CognitionView) -> str:
        if not view.recalled_beliefs:
            return ""
        prefix = ""
        if view.counterpart is not None and view.counterpart.trust_level < 0.5:
            prefix = "User-reported, not verified by agent: "
        lines = [
            f"- {prefix}{belief.content} (confidence={belief.confidence:.2f}, id={belief.id})"
            for belief in view.recalled_beliefs
        ]
        return "Recalled beliefs:\n" + "\n".join(lines)

    def _background(self, view: CognitionView) -> str:
        return f"Context background: {view.window.background}" if view.window.background else ""

    def _current_query(self, view: CognitionView) -> str:
        if view.current_query:
            return view.current_query
        for perception in reversed(view.window.foreground):
            if perception.raw is not None:
                return str(perception.raw)
        return ""


_SYSTEM_PROMPT = (
    "Identity: Alpha Agent.\n"
    "Use the current reactive context and answer concisely. "
    "Call tools only when they are useful.\n"
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


def _section_name_from_message(message: ChatMessage) -> str:
    content = str(message.get("content") or "")
    if "Recalled beliefs:" in content:
        return "recalled_beliefs"
    if "Context background:" in content:
        return "background"
    if "Counterpart digest:" in content:
        return "counterpart_digest"
    if "Strategy reminders:" in content:
        return "strategy_reminders"
    return "unknown"


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
