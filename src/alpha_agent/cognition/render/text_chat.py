"""Chat-completions renderer for reactive cognition."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from alpha_agent.cognition.models import CounterpartRole
from alpha_agent.cognition.render.base import RenderBudget, RenderResult
from alpha_agent.cognition.render.view import CognitionView
from alpha_agent.llm.base import ChatCompletionToolCall, ChatMessage, LLMToolDefinitionInput
from alpha_agent.state.models import ConversationMessage

SYSTEM_REMINDER_OPEN = "<system-reminder>"
SYSTEM_REMINDER_CLOSE = "</system-reminder>"


def wrap_system_reminder(content: str) -> str:
    return f"{SYSTEM_REMINDER_OPEN}\n{content.strip()}\n{SYSTEM_REMINDER_CLOSE}"


def estimate_chat_tokens(
    messages: Sequence[ChatMessage],
    *,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
) -> int:
    character_count = sum(_message_character_count(message) for message in messages)
    if tools is not None:
        character_count += sum(len(_stable_json(_tool_payload(tool))) for tool in tools)
    return character_count // 4


def conversation_message_to_chat(message: ConversationMessage) -> ChatMessage:
    """Convert a durable transcript message into a replayable chat message."""

    content = message.model_content if message.model_content is not None else message.raw_content
    if message.role == "user":
        return {"role": "user", "content": content}
    if message.role == "assistant":
        if message.tool_calls:
            return {
                "role": "assistant",
                "content": content or None,
                "tool_calls": [
                    _conversation_tool_call(tool_call) for tool_call in message.tool_calls
                ],
            }
        return {"role": "assistant", "content": content}
    if not message.tool_call_id:
        raise ValueError(f"tool conversation message {message.id!r} is missing tool_call_id")
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
        sections = [
            ("strategy_reminders", self._strategy_reminders(view)),
            ("counterpart_digest", self._counterpart_digest(view)),
            ("recalled_beliefs", self._recalled_beliefs(view)),
            ("background", self._background(view)),
            ("foreground", self._foreground(view)),
        ]
        for section, content in sections:
            if not content:
                continue
            clipped = _clip_to_budget(content, budget.per_section_tokens.get(section))
            if clipped is None:
                dropped.append(section)
                continue
            messages.append({"role": "system", "content": wrap_system_reminder(clipped)})
        messages.append({"role": "user", "content": self._current_query(view)})
        used_tokens = estimate_chat_tokens(messages)
        while used_tokens > budget.max_tokens and len(messages) > 2:
            removed = messages.pop(-2)
            dropped.append(_section_name_from_message(removed))
            used_tokens = estimate_chat_tokens(messages)
        return RenderResult(
            payload=messages,
            used_tokens=used_tokens,
            dropped_sections=dropped,
        )

    def _system_prompt(self, view: CognitionView, budget: RenderBudget) -> str:
        role = view.counterpart.role if view.counterpart is not None else CounterpartRole.ANONYMOUS
        template = _SYSTEM_PROMPTS.get(role, _SYSTEM_PROMPTS[CounterpartRole.ANONYMOUS])
        parts = [template]
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

    def _foreground(self, view: CognitionView) -> str:
        lines = [str(item.raw) for item in view.window.foreground if item.raw is not None]
        return "Foreground:\n" + "\n".join(f"- {line}" for line in lines) if lines else ""

    def _current_query(self, view: CognitionView) -> str:
        if view.current_query:
            return view.current_query
        for perception in reversed(view.window.foreground):
            if perception.raw is not None:
                return str(perception.raw)
        return ""


_SYSTEM_PROMPTS = {
    CounterpartRole.USER: (
        "Identity: Alpha Agent.\n"
        "Use the current reactive context and answer naturally, concisely, and usefully. "
        "Call tools only when they are useful."
    ),
    CounterpartRole.OPERATOR: (
        "Identity: Alpha Agent.\n"
        "Respond to the operator in a compact, protocol-oriented form. Surface decisions, "
        "constraints, and blockers directly. Call tools only when they are useful."
    ),
    CounterpartRole.PEER_AGENT: (
        "Identity: Alpha Agent.\n"
        "Coordinate with the peer agent using explicit state, assumptions, and next actions. "
        "Call tools only when they are useful."
    ),
    CounterpartRole.SYSTEM: (
        "Identity: Alpha Agent.\n"
        "Handle system-originated input conservatively and report operational state clearly. "
        "Call tools only when they are useful."
    ),
    CounterpartRole.ANONYMOUS: (
        "Identity: Alpha Agent.\n"
        "Use the current reactive context and answer concisely. "
        "Call tools only when they are useful."
    ),
}


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
    if "Foreground:" in content:
        return "foreground"
    if "Context background:" in content:
        return "background"
    if "Counterpart digest:" in content:
        return "counterpart_digest"
    if "Strategy reminders:" in content:
        return "strategy_reminders"
    return "unknown"


def _conversation_tool_call(payload: Mapping[str, Any]) -> ChatCompletionToolCall:
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


def _message_character_count(message: ChatMessage) -> int:
    count = len(str(message.get("role", "")))
    content = message.get("content")
    if isinstance(content, str):
        count += len(content)
    tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(tool_calls, list):
        count += len(_stable_json(tool_calls))
    tool_call_id = message.get("tool_call_id") if isinstance(message, dict) else None
    if isinstance(tool_call_id, str):
        count += len(tool_call_id)
    return count


def _tool_payload(tool: LLMToolDefinitionInput) -> dict[str, Any]:
    if hasattr(tool, "name"):
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": tool.strict,
        }
    return dict(tool)


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
