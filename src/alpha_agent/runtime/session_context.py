"""Answer-path session context projection built from source messages."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_agent.config import LLMContextConfig
from alpha_agent.llm.base import ChatMessage, LLMToolDefinitionInput
from alpha_agent.runtime.chat_messages import (
    TOOL_TRUNCATION_MARKER,
    session_message_to_chat,
)
from alpha_agent.runtime.context_budget import (
    ContextBudgetEstimate,
    estimate_context_budget,
)
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore


@dataclass(frozen=True)
class SessionContextProjection:
    """LLM-visible answer context projected from the session message stream."""

    source_messages: list[SessionMessage]
    chat_messages: list[ChatMessage]
    compressed_message: SessionMessage | None = None
    before_ordinal: int | None = None


@dataclass(frozen=True)
class ToolContextTruncationResult:
    """Result of one explicit tool replay payload truncation maintenance pass."""

    triggered: bool
    checked_message_ids: list[str]
    truncated_message_ids: list[str]
    before_estimate: ContextBudgetEstimate
    after_estimate: ContextBudgetEstimate


@dataclass(frozen=True)
class _ReplayPayloadUpdate:
    message_id: str
    raw_content: str
    model_content: str | None
    tool_calls: list[dict[str, Any]]
    metadata: dict[str, Any]
    truncated: bool


class SessionContextAssembler:
    """Assemble answer-path LLM context without mutating the message stream."""

    def __init__(self, store: StateStore):
        self.store = store

    def load(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> SessionContextProjection:
        """Load runtime handover continuity plus later source messages.

        Background cognition ledgers, stage runs, and audit rows are maintenance
        artifacts; they are not part of this answer-path projection.
        """

        compressed = self.store.find_latest_compressed_message(
            session_id,
            before_ordinal=before_ordinal,
        )
        after_ordinal = compressed.ordinal if compressed is not None else None
        ordinary_messages = [
            message
            for message in self.store.list_session_messages(
                session_id,
                after_ordinal=after_ordinal,
                before_ordinal=before_ordinal,
            )
            if message.kind != "compressed_message"
        ]
        source_messages = [compressed, *ordinary_messages] if compressed else ordinary_messages
        return SessionContextProjection(
            source_messages=source_messages,
            chat_messages=[session_message_to_chat(message) for message in source_messages],
            compressed_message=compressed,
            before_ordinal=before_ordinal,
        )

    def assemble_messages_for_llm(
        self,
        session_id: str,
        *,
        before_ordinal: int | None = None,
    ) -> list[ChatMessage]:
        """Return ChatMessage-shaped source context for an LLM call."""

        return self.load(session_id, before_ordinal=before_ordinal).chat_messages

    def truncate_tool_context_if_needed(
        self,
        session_id: str,
        *,
        context_config: LLMContextConfig | None = None,
        max_context_tokens: int,
        tools: Sequence[LLMToolDefinitionInput | Mapping[str, Any]] | None = None,
        before_ordinal: int | None = None,
        planning_messages: Sequence[ChatMessage | Mapping[str, Any]] | None = None,
    ) -> ToolContextTruncationResult:
        """Truncate unchecked replay tool payloads after the latest compression boundary."""

        config = context_config or LLMContextConfig()
        projection = self.load(session_id, before_ordinal=before_ordinal)
        budget_messages = [*projection.chat_messages, *(planning_messages or ())]
        before_estimate = estimate_context_budget(
            budget_messages,
            tools=tools,
            context_config=config,
            max_context_tokens=max_context_tokens,
        )
        threshold_tokens = max_context_tokens * config.tool_truncate_threshold_ratio
        if before_estimate.used_context_tokens <= threshold_tokens:
            return ToolContextTruncationResult(
                triggered=False,
                checked_message_ids=[],
                truncated_message_ids=[],
                before_estimate=before_estimate,
                after_estimate=before_estimate,
            )

        updates = [
            update
            for message in projection.source_messages
            if (update := _prepare_tool_payload_update(message, config.tool_string_truncate_chars))
            is not None
        ]
        if updates:
            with self.store.immediate_transaction() as conn:
                for update in updates:
                    self.store.update_session_message_replay_payload(
                        update.message_id,
                        raw_content=update.raw_content,
                        model_content=update.model_content,
                        tool_calls=update.tool_calls,
                        metadata=update.metadata,
                        conn=conn,
                    )

        after_projection = self.load(session_id, before_ordinal=before_ordinal)
        after_budget_messages = [
            *after_projection.chat_messages,
            *(planning_messages or ()),
        ]
        after_estimate = estimate_context_budget(
            after_budget_messages,
            tools=tools,
            context_config=config,
            max_context_tokens=max_context_tokens,
        )
        return ToolContextTruncationResult(
            triggered=True,
            checked_message_ids=[update.message_id for update in updates],
            truncated_message_ids=[update.message_id for update in updates if update.truncated],
            before_estimate=before_estimate,
            after_estimate=after_estimate,
        )


def _prepare_tool_payload_update(
    message: SessionMessage,
    tool_string_truncate_chars: int,
) -> _ReplayPayloadUpdate | None:
    if message.metadata.get("truncate_checked") is True:
        return None
    if message.kind == "assistant_message" and message.tool_calls:
        return _prepare_assistant_tool_call_update(message, tool_string_truncate_chars)
    if message.kind == "tool_message":
        return _prepare_tool_message_update(message, tool_string_truncate_chars)
    return None


def _prepare_assistant_tool_call_update(
    message: SessionMessage,
    tool_string_truncate_chars: int,
) -> _ReplayPayloadUpdate:
    tool_calls = copy.deepcopy(message.tool_calls)
    original_lengths: dict[str, int] = {}
    for index, tool_call in enumerate(tool_calls):
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise ValueError(f"assistant tool_call {index} is missing function payload")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise ValueError(
                f"assistant tool_call {index} function.arguments must be a JSON string"
            )
        parsed_arguments = json.loads(arguments)
        truncated_arguments = _truncate_json_strings(
            parsed_arguments,
            tool_string_truncate_chars,
            f"tool_calls[{index}].function.arguments",
            original_lengths,
        )
        function["arguments"] = _dump_replay_json(truncated_arguments)
    return _ReplayPayloadUpdate(
        message_id=message.id,
        raw_content=message.raw_content,
        model_content=message.model_content,
        tool_calls=tool_calls,
        metadata=_truncate_checked_metadata(message.metadata, original_lengths),
        truncated=bool(original_lengths),
    )


def _prepare_tool_message_update(
    message: SessionMessage,
    tool_string_truncate_chars: int,
) -> _ReplayPayloadUpdate:
    original_lengths: dict[str, int] = {}
    raw_content = _truncate_tool_message_content(
        message.raw_content,
        tool_string_truncate_chars,
        "raw_content",
        original_lengths,
        message.metadata,
    )
    model_content = message.model_content
    if model_content is not None:
        model_content = _truncate_tool_message_content(
            model_content,
            tool_string_truncate_chars,
            "model_content",
            original_lengths,
            message.metadata,
        )
    return _ReplayPayloadUpdate(
        message_id=message.id,
        raw_content=raw_content,
        model_content=model_content,
        tool_calls=message.tool_calls,
        metadata=_truncate_checked_metadata(message.metadata, original_lengths),
        truncated=bool(original_lengths),
    )


def _truncate_tool_message_content(
    content: str,
    tool_string_truncate_chars: int,
    path: str,
    original_lengths: dict[str, int],
    metadata: Mapping[str, Any],
) -> str:
    if metadata.get("tool_output_kind") == "text":
        return _truncate_json_strings(
            content,
            tool_string_truncate_chars,
            path,
            original_lengths,
        )
    payload = json.loads(content)
    return _dump_replay_json(
        _truncate_json_strings(
            payload,
            tool_string_truncate_chars,
            path,
            original_lengths,
        )
    )


def _truncate_checked_metadata(
    metadata: Mapping[str, Any],
    original_lengths: Mapping[str, int],
) -> dict[str, Any]:
    updated = dict(metadata)
    updated["truncate_checked"] = True
    updated["original_lengths"] = dict(original_lengths)
    return updated


def _truncate_json_strings(
    value: Any,
    tool_string_truncate_chars: int,
    path: str,
    original_lengths: dict[str, int],
) -> Any:
    if tool_string_truncate_chars < 0:
        raise ValueError("tool_string_truncate_chars must be non-negative")
    if isinstance(value, str):
        if len(value) <= tool_string_truncate_chars:
            return value
        original_lengths[path] = len(value)
        return value[:tool_string_truncate_chars] + TOOL_TRUNCATION_MARKER
    if isinstance(value, list):
        return [
            _truncate_json_strings(
                item,
                tool_string_truncate_chars,
                f"{path}[{index}]",
                original_lengths,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _truncate_json_strings(
                item,
                tool_string_truncate_chars,
                f"{path}.{key}",
                original_lengths,
            )
            for key, item in value.items()
        }
    return value


def _dump_replay_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "SessionContextAssembler",
    "SessionContextProjection",
    "ToolContextTruncationResult",
    "session_message_to_chat",
]
