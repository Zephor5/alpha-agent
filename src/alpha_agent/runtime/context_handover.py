"""LLM handover compression over projected session context."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from alpha_agent.llm.base import (
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.context_budget import stable_json
from alpha_agent.runtime.session_context import (
    SessionContextAssembler,
    SessionContextProjection,
    wrap_system_reminder,
)
from alpha_agent.state.models import SessionMessage

DEFAULT_HANDOVER_COMPRESSION_VERSION = "handover-compression-v1"
DEFAULT_MEMORY_EXTRACTION_VERSION = "memory-extraction-v1"
DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION = (
    """Create an operational continuity handover for the next context holder.

Preserve the active task state, decisions, constraints, user preferences,
completed actions, open questions, and supporting source context needed to
continue accurately. Treat this as continuity context, not a short summary.
Do not describe or include this compression instruction as session content."""
)


@dataclass(frozen=True)
class HandoverCompressionPrompt:
    """Transient provider prompt plus the source ordinal it covers."""

    messages: list[ChatMessage]
    compression_point_ordinal: int


@dataclass(frozen=True)
class HandoverCompressionResult:
    """Successful compression append result."""

    message: SessionMessage
    response: LLMResponse
    compression_point_ordinal: int


def build_handover_compression_prompt(
    messages: Sequence[ChatMessage],
    *,
    compression_point_ordinal: int,
    instruction: str = DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION,
) -> HandoverCompressionPrompt:
    """Append the transient compression instruction to current LLM-visible messages."""

    if compression_point_ordinal < 1:
        raise ValueError("cannot compress session context without a compression point")
    prompt_messages = list(messages)
    prompt_messages.append({"role": "user", "content": wrap_system_reminder(instruction)})
    return HandoverCompressionPrompt(
        messages=prompt_messages,
        compression_point_ordinal=compression_point_ordinal,
    )


def handover_prompt_prefix_hash(messages: Sequence[ChatMessage]) -> str:
    """Hash the provider-visible message prefix before the compression suffix."""

    return _stable_hash({"messages": list(messages), "shape": "handover_prompt_prefix_v1"})


def handover_tools_schema_hash(
    tools: Sequence[LLMToolDefinitionInput] | None,
) -> str:
    """Hash the stable provider-visible tool schema used by handover compression."""

    return _stable_hash({"shape": "handover_tools_schema_v1", "tools": list(tools or [])})


def build_handover_compression_prompt_from_projection_with_prefix(
    projection: SessionContextProjection,
    *,
    prefix_messages: Sequence[ChatMessage],
    instruction: str = DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION,
) -> HandoverCompressionPrompt:
    """Build a projection-based compression prompt only with an explicit runtime prefix."""

    prefix = list(prefix_messages)
    if not prefix:
        raise ValueError("projection compression helper requires runtime prefix messages")
    if prefix[0].get("role") != "system":
        raise ValueError("runtime prefix must start with a system message")
    if not projection.source_messages:
        raise ValueError("cannot compress session context without a compression point")

    compression_point_ordinal = max(message.ordinal for message in projection.source_messages)
    return build_handover_compression_prompt(
        [*prefix, *projection.chat_messages],
        compression_point_ordinal=compression_point_ordinal,
        instruction=instruction,
    )


def compress_session_context(
    *,
    session_id: str,
    assembler: SessionContextAssembler,
    llm_provider: LLMProvider,
    llm_messages: Sequence[ChatMessage] | None,
    tools: Sequence[LLMToolDefinitionInput] | None = None,
    tool_choice: LLMToolChoice | None = None,
    before_ordinal: int | None = None,
    instruction: str = DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION,
    compression_version: str = DEFAULT_HANDOVER_COMPRESSION_VERSION,
    extraction_version: str = DEFAULT_MEMORY_EXTRACTION_VERSION,
    trace_metadata: Mapping[str, Any] | None = None,
) -> HandoverCompressionResult:
    """Call the LLM for handover compression and append the returned source record."""

    projection = assembler.load(session_id, before_ordinal=before_ordinal)
    if not projection.source_messages:
        raise ValueError("cannot compress session context without a compression point")
    if llm_messages is None:
        raise ValueError("runtime compression requires explicit llm_messages")
    compression_point_ordinal = max(message.ordinal for message in projection.source_messages)
    prompt = build_handover_compression_prompt(
        llm_messages,
        compression_point_ordinal=compression_point_ordinal,
        instruction=instruction,
    )

    compression_trace_metadata = {
        **dict(trace_metadata or {}),
        "compression_point_ordinal": prompt.compression_point_ordinal,
        "compression_version": compression_version,
        "extraction_version": extraction_version,
        "before_ordinal": before_ordinal,
        "provider": _provider_name(llm_provider),
        "model": _provider_model(llm_provider),
        "prompt_prefix_hash": handover_prompt_prefix_hash(llm_messages),
        "tools_schema_hash": handover_tools_schema_hash(tools),
        "prompt_prefix_message_count": len(llm_messages),
        "prompt_message_count": len(prompt.messages),
        "tool_count": len(tools) if tools is not None else 0,
        "tool_choice": tool_choice,
        **_covered_source_metadata(projection, prompt.compression_point_ordinal),
    }
    _append_compression_trace(
        assembler,
        session_id=session_id,
        event_type="handover_compression.started",
        content="Handover compression started.",
        metadata=compression_trace_metadata,
    )
    try:
        response = llm_provider.complete(
            list(prompt.messages),
            tools=tools,
            tool_choice=tool_choice,
        )
        compressed = assembler.store.append_compressed_message(
            session_id=session_id,
            raw_content=response.content,
            compression_point_ordinal=prompt.compression_point_ordinal,
            compression_version=compression_version,
            provider_metadata={
                "provider": response.provider,
                "model": response.model,
                "finish_reason": response.finish_reason,
            },
            metadata={"source": "handover_compression"},
        )
    except Exception as exc:
        _append_compression_trace(
            assembler,
            session_id=session_id,
            event_type="handover_compression.failed",
            content="Handover compression failed.",
            metadata={
                **compression_trace_metadata,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise

    _append_compression_trace(
        assembler,
        session_id=session_id,
        event_type="handover_compression.completed",
        content="Handover compression completed.",
        metadata={
            **compression_trace_metadata,
            "provider": response.provider,
            "model": response.model,
            "response_provider": response.provider,
            "response_model": response.model,
            "finish_reason": response.finish_reason,
            "compressed_message_id": compressed.id,
            "compressed_message_ordinal": compressed.ordinal,
        },
    )
    return HandoverCompressionResult(
        message=compressed,
        response=response,
        compression_point_ordinal=prompt.compression_point_ordinal,
    )


def _provider_name(llm_provider: LLMProvider) -> str:
    return str(getattr(llm_provider, "name", type(llm_provider).__name__))


def _provider_model(llm_provider: LLMProvider) -> str | None:
    model = getattr(llm_provider, "model", None)
    return str(model) if model is not None else None


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def _covered_source_metadata(
    projection: SessionContextProjection,
    compression_point_ordinal: int,
) -> dict[str, Any]:
    context_refs = [
        _session_message_ref_record(message)
        for message in projection.source_messages
        if message.ordinal <= compression_point_ordinal
    ]
    covered_refs = [
        _session_message_ref_record(message)
        for message in projection.source_messages
        if message.kind != "compressed_message"
        and message.ordinal <= compression_point_ordinal
    ]
    ordinals = [int(item["ordinal"]) for item in covered_refs]
    return {
        "context_source_message_refs": context_refs,
        "covered_source_message_refs": covered_refs,
        "covered_source_message_ids": [str(item["source_id"]) for item in covered_refs],
        "covered_ordinal_start": min(ordinals) if ordinals else None,
        "covered_ordinal_end": max(ordinals) if ordinals else None,
    }


def _session_message_ref_record(message: SessionMessage) -> dict[str, Any]:
    return {
        "source_type": "session_message",
        "source_id": message.id,
        "ordinal": message.ordinal,
        "kind": message.kind,
    }


def _append_compression_trace(
    assembler: SessionContextAssembler,
    *,
    session_id: str,
    event_type: str,
    content: str,
    metadata: dict[str, Any],
) -> None:
    try:
        assembler.store.append_runtime_trace(
            session_id=session_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )
    except Exception:
        return


__all__ = [
    "DEFAULT_HANDOVER_COMPRESSION_INSTRUCTION",
    "DEFAULT_HANDOVER_COMPRESSION_VERSION",
    "DEFAULT_MEMORY_EXTRACTION_VERSION",
    "HandoverCompressionPrompt",
    "HandoverCompressionResult",
    "build_handover_compression_prompt",
    "build_handover_compression_prompt_from_projection_with_prefix",
    "compress_session_context",
    "handover_prompt_prefix_hash",
    "handover_tools_schema_hash",
]
