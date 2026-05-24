"""Session context compression primitives."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from alpha_agent.llm.base import LLMProvider
from alpha_agent.memory.models import ConversationMessage

DETERMINISTIC_COMPRESSION_VERSION = "structured-session-state-v1"


@dataclass(frozen=True)
class CompressionBudget:
    """Prompt budget and compression policy for one model request."""

    max_prompt_tokens: int = 6000
    threshold_ratio: float = 0.85
    recent_tail_messages: int = 8
    min_summary_tokens: int = 128
    max_summary_tokens: int = 512

    @property
    def threshold_tokens(self) -> int:
        """Return the token estimate where compression should start."""

        return max(1, int(self.max_prompt_tokens * self.threshold_ratio))

    @property
    def effective_recent_tail_messages(self) -> int:
        """Preserve at least one prior message whenever compression is possible."""

        return max(1, self.recent_tail_messages)


@dataclass(frozen=True)
class CompressionContext:
    """Compression decision input for a projected prompt."""

    session_id: str
    prompt_token_estimate: int
    uncompressed_message_count: int
    has_previous_summary: bool


@dataclass(frozen=True)
class CompressionFocus:
    """Turn-specific context supplied to a compressor implementation."""

    session_id: str
    current_user_message: str
    prompt_token_estimate: int
    budget: CompressionBudget
    compressed_until_ordinal: int
    previous_summary_source_message_ids: list[str] = field(default_factory=list)
    previous_projection: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionResult:
    """Structured result produced by a context compressor."""

    summary: str
    summary_source_message_ids: list[str]
    compressed_until_ordinal: int
    compression_version: str
    input_token_estimate: int
    output_token_estimate: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompressionSelection:
    """Prefix selected for compression and tail selected for raw replay."""

    messages_to_compress: list[ConversationMessage]
    preserved_messages: list[ConversationMessage]
    split_index: int


@dataclass(frozen=True)
class StructuredSessionState:
    """Stable task state projected from earlier transcript messages."""

    current_goal: str
    decisions: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    user_constraints: list[str] = field(default_factory=list)
    relevant_files_entities: list[str] = field(default_factory=list)
    last_action: str = ""

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-compatible projection record."""

        return {
            "current_goal": self.current_goal,
            "decisions": list(self.decisions),
            "open_questions": list(self.open_questions),
            "pending_tasks": list(self.pending_tasks),
            "user_constraints": list(self.user_constraints),
            "relevant_files_entities": list(self.relevant_files_entities),
            "last_action": self.last_action,
        }


class ContextCompressor(Protocol):
    """Interface for deterministic and LLM-backed context compressors."""

    compression_version: str

    def should_compress(
        self,
        context: CompressionContext,
        budget: CompressionBudget,
    ) -> bool:
        """Return whether the projected prompt should be compressed."""

    def compress(
        self,
        messages: Sequence[ConversationMessage],
        previous_summary: str,
        focus: CompressionFocus,
    ) -> CompressionResult:
        """Compress older transcript messages into a replacement summary."""


class LLMContextCompressor:
    """Extension point for provider-backed compression."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        compression_version: str = "llm-context-compressor-unimplemented",
    ):
        self.provider = provider
        self.compression_version = compression_version

    def should_compress(
        self,
        context: CompressionContext,
        budget: CompressionBudget,
    ) -> bool:
        """Use the same deterministic trigger; only summary generation needs an LLM."""

        return context.prompt_token_estimate > budget.threshold_tokens

    def compress(
        self,
        messages: Sequence[ConversationMessage],
        previous_summary: str,
        focus: CompressionFocus,
    ) -> CompressionResult:
        """Implement provider-backed summarization here when wiring cost is justified."""

        raise NotImplementedError(
            "LLMContextCompressor.compress is an explicit extension point; "
            "use DeterministicContextCompressor as the fallback implementation."
        )


class DeterministicContextCompressor:
    """Local deterministic compressor for tests and fallback operation."""

    compression_version = DETERMINISTIC_COMPRESSION_VERSION

    def should_compress(
        self,
        context: CompressionContext,
        budget: CompressionBudget,
    ) -> bool:
        """Compress when the prompt estimate crosses the configured threshold."""

        return context.prompt_token_estimate > budget.threshold_tokens

    def compress(
        self,
        messages: Sequence[ConversationMessage],
        previous_summary: str,
        focus: CompressionFocus,
    ) -> CompressionResult:
        """Project older transcript messages into bounded structured session state."""

        if not messages:
            raise ValueError("cannot compress an empty message set")

        message_list = list(messages)
        source_ids = [
            *focus.previous_summary_source_message_ids,
            *[message.id for message in message_list],
        ]
        projection = self._project_session_state(
            messages=message_list,
            previous_summary=previous_summary,
            previous_projection=focus.previous_projection,
        )
        summary = self._summary_text(
            projection=projection,
            max_summary_tokens=focus.budget.max_summary_tokens,
        )
        input_token_estimate = _estimate_messages_tokens(message_list) + _estimate_text_tokens(
            previous_summary
        )
        output_token_estimate = _estimate_text_tokens(summary)
        return CompressionResult(
            summary=summary,
            summary_source_message_ids=source_ids,
            compressed_until_ordinal=message_list[-1].ordinal,
            compression_version=self.compression_version,
            input_token_estimate=input_token_estimate,
            output_token_estimate=output_token_estimate,
            metadata={
                "compressor": self.compression_version,
                "input_message_count": len(message_list),
                "input_first_ordinal": message_list[0].ordinal,
                "input_last_ordinal": message_list[-1].ordinal,
                "previous_summary_included": bool(previous_summary.strip()),
                "projection": projection.to_record(),
            },
        )

    def _project_session_state(
        self,
        *,
        messages: Sequence[ConversationMessage],
        previous_summary: str,
        previous_projection: dict[str, Any] | None = None,
    ) -> StructuredSessionState:
        prior = _projection_from_record(previous_projection)
        if not prior:
            prior = _projection_from_previous_summary(previous_summary)
        current_goal = str(prior.get("current_goal") or "").strip()
        decisions = _list_from_record(prior.get("decisions"))
        open_questions = _list_from_record(prior.get("open_questions"))
        pending_tasks = _list_from_record(prior.get("pending_tasks"))
        user_constraints = _list_from_record(prior.get("user_constraints"))
        relevant_files_entities = _list_from_record(prior.get("relevant_files_entities"))
        last_action = str(prior.get("last_action") or "").strip()

        for message in messages:
            text = _message_content(message)
            if not text:
                continue
            sentences = _sentence_units(text)
            if message.role == "user":
                current_goal = _latest_goal(sentences, fallback=current_goal)
                for item in _matching_units(
                    sentences,
                    ("must", "do not", "don't", "avoid", "prefer", "constraint"),
                ):
                    _append_unique(user_constraints, item)
                for item in _matching_units(
                    sentences,
                    ("todo", "task", "need to", "please", "implement", "fix", "finish"),
                ):
                    _append_unique(pending_tasks, item)
                for item in [sentence for sentence in sentences if "?" in sentence]:
                    _append_unique(open_questions, item)
            elif message.role == "assistant":
                for item in _matching_units(
                    sentences,
                    ("decision", "decided", "choose", "chosen", "use ", "store ", "replace"),
                ):
                    _append_unique(decisions, item)
                for item in _matching_units(
                    sentences,
                    ("todo", "next", "pending", "need to", "will"),
                ):
                    _append_unique(pending_tasks, item)
            for item in _extract_files_and_entities(text):
                _append_unique(relevant_files_entities, item)
            last_action = _message_summary(message)

        if not current_goal:
            current_goal = _latest_goal(
                _sentence_units(_message_content(messages[-1])),
                fallback="",
            )
        return StructuredSessionState(
            current_goal=current_goal or "Continue from the uncompressed tail.",
            decisions=decisions[-8:],
            open_questions=open_questions[-8:],
            pending_tasks=pending_tasks[-8:],
            user_constraints=user_constraints[-8:],
            relevant_files_entities=relevant_files_entities[-12:],
            last_action=last_action or _message_summary(messages[-1]),
        )

    def _summary_text(
        self,
        *,
        projection: StructuredSessionState,
        max_summary_tokens: int,
    ) -> str:
        lines = [
            "## Structured Session State",
            "",
            "### Current Goal",
            f"- {projection.current_goal}",
            "",
            "### Decisions",
        ]
        lines.extend(_render_items(projection.decisions))
        lines.extend(["", "### Open Questions"])
        lines.extend(_render_items(projection.open_questions))
        lines.extend(["", "### Pending Tasks"])
        lines.extend(_render_items(projection.pending_tasks))
        lines.extend(["", "### User Constraints"])
        lines.extend(_render_items(projection.user_constraints))
        lines.extend(["", "### Relevant Files / Entities"])
        lines.extend(_render_items(projection.relevant_files_entities))
        lines.extend(["", "### Last Action", f"- {projection.last_action}"])
        summary = "\n".join(lines)
        return _clip_to_token_budget(summary, max_summary_tokens)


def select_compression_window(
    messages: Sequence[ConversationMessage],
    *,
    recent_tail_messages: int,
) -> CompressionSelection:
    """Select older messages to summarize while preserving valid tool replay."""

    message_list = list(messages)
    if not message_list:
        return CompressionSelection([], [], 0)

    tail_count = max(1, recent_tail_messages)
    split_index = max(0, len(message_list) - tail_count)
    split_index = _rewind_split_for_tool_replay(message_list, split_index)
    if split_index <= 0:
        return CompressionSelection([], message_list, 0)
    return CompressionSelection(
        messages_to_compress=message_list[:split_index],
        preserved_messages=message_list[split_index:],
        split_index=split_index,
    )


def _rewind_split_for_tool_replay(
    messages: Sequence[ConversationMessage],
    split_index: int,
) -> int:
    if split_index <= 0 or split_index >= len(messages):
        return split_index

    adjusted = split_index
    changed = True
    while changed and adjusted > 0:
        changed = False
        if messages[adjusted].role == "tool":
            assistant_index = _find_assistant_index_for_tool(messages, adjusted)
            if assistant_index is None:
                return 0
            if assistant_index < adjusted:
                adjusted = assistant_index
                changed = True
                continue
        for start, end in _tool_replay_spans(messages):
            if start < adjusted < end:
                adjusted = start
                changed = True
                break
    return adjusted


def _find_assistant_index_for_tool(
    messages: Sequence[ConversationMessage],
    tool_index: int,
) -> int | None:
    tool_call_id = messages[tool_index].tool_call_id
    if tool_call_id is None:
        return None
    for index in range(tool_index - 1, -1, -1):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            continue
        if tool_call_id in _tool_call_ids(message):
            return index
    return None


def _tool_replay_spans(
    messages: Sequence[ConversationMessage],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if message.role != "assistant" or not message.tool_calls:
            continue
        required_call_ids = _tool_call_ids(message)
        if not required_call_ids:
            continue
        seen_call_ids: set[str] = set()
        end = index + 1
        while end < len(messages) and messages[end].role == "tool":
            tool_call_id = messages[end].tool_call_id
            if tool_call_id in required_call_ids:
                seen_call_ids.add(tool_call_id)
            end += 1
            if seen_call_ids == required_call_ids:
                break
        if end > index + 1:
            spans.append((index, end))
    return spans


def _tool_call_ids(message: ConversationMessage) -> set[str]:
    ids: set[str] = set()
    for tool_call in message.tool_calls:
        tool_call_id = tool_call.get("id")
        if tool_call_id is not None:
            ids.add(str(tool_call_id))
    return ids


def _message_summary(message: ConversationMessage) -> str:
    content = message.model_content if message.model_content is not None else message.raw_content
    if message.role == "assistant" and message.tool_calls:
        tool_names = [
            str(tool_call.get("function", {}).get("name") or tool_call.get("name"))
            for tool_call in message.tool_calls
        ]
        content = f"requested tools: {', '.join(name for name in tool_names if name)}"
    elif message.role == "tool":
        content = f"tool result {message.tool_call_id}: {content}"
    return f"{message.ordinal}. {message.role}: {_clip(content, 80)}"


def _message_content(message: ConversationMessage) -> str:
    return message.model_content if message.model_content is not None else message.raw_content


def _projection_from_previous_summary(summary: str) -> dict[str, Any]:
    if not summary.strip():
        return {}
    sections: dict[str, list[str]] = {}
    current_title = ""
    for raw_line in summary.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            current_title = line[4:].casefold()
            sections[current_title] = []
            continue
        if current_title and line.startswith("- "):
            sections[current_title].append(line[2:].strip())
    return {
        "current_goal": _first(sections.get("current goal", [])),
        "decisions": sections.get("decisions", []),
        "open_questions": sections.get("open questions", []),
        "pending_tasks": sections.get("pending tasks", []),
        "user_constraints": sections.get("user constraints", []),
        "relevant_files_entities": sections.get("relevant files / entities", []),
        "last_action": _first(sections.get("last action", [])),
    }


def _projection_from_record(record: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {
        "current_goal": str(record.get("current_goal") or "").strip(),
        "decisions": _list_from_record(record.get("decisions")),
        "open_questions": _list_from_record(record.get("open_questions")),
        "pending_tasks": _list_from_record(record.get("pending_tasks")),
        "user_constraints": _list_from_record(record.get("user_constraints")),
        "relevant_files_entities": _list_from_record(
            record.get("relevant_files_entities")
        ),
        "last_action": str(record.get("last_action") or "").strip(),
    }


def _list_from_record(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _sentence_units(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized)
    return [_clip(part.strip(), 180) for part in parts if part.strip()]


def _latest_goal(sentences: Sequence[str], *, fallback: str) -> str:
    for sentence in reversed(sentences):
        lower = sentence.casefold()
        if lower.startswith("goal:"):
            return sentence.split(":", 1)[-1].strip()
        if any(marker in lower for marker in ("goal", "task", "implement", "finish", "fix")):
            return sentence
    return fallback


def _matching_units(sentences: Sequence[str], markers: Sequence[str]) -> list[str]:
    return [
        sentence
        for sentence in sentences
        if any(marker in sentence.casefold() for marker in markers)
    ]


def _extract_files_and_entities(text: str) -> list[str]:
    items = re.findall(r"\b(?:src|tests|docs)/[A-Za-z0-9_./-]+", text)
    items.extend(re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,3}\b", text))
    return [_clip(item.rstrip(".,:;"), 120) for item in items]


def _append_unique(values: list[str], value: str) -> None:
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return
    if normalized.casefold() in {item.casefold() for item in values}:
        return
    values.append(normalized)


def _render_items(values: Sequence[str]) -> list[str]:
    if not values:
        return ["- None captured."]
    return [f"- {value}" for value in values]


def _first(values: Sequence[str]) -> str:
    return values[0] if values else ""


def _estimate_messages_tokens(messages: Sequence[ConversationMessage]) -> int:
    return sum(_estimate_text_tokens(_message_token_text(message)) for message in messages)


def _message_token_text(message: ConversationMessage) -> str:
    pieces = [
        message.role,
        message.model_content if message.model_content is not None else message.raw_content,
        str(message.tool_call_id or ""),
        str(message.tool_calls),
    ]
    return "\n".join(pieces)


def _estimate_text_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _clip_to_token_budget(text: str, max_tokens: int) -> str:
    max_characters = max(80, max_tokens * 4)
    stripped = text.strip()
    if len(stripped) <= max_characters:
        return stripped
    return stripped[: max(0, max_characters - 3)].rstrip() + "..."


def _clip(text: str, max_characters: int) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_characters:
        return normalized
    return normalized[: max(0, max_characters - 3)].rstrip() + "..."
