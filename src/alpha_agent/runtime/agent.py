"""Explicit personal agent runtime."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import httpx

from alpha_agent.llm.base import (
    ChatCompletionToolCall,
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.memory.episodic import EpisodicMemoryManager
from alpha_agent.memory.extractor import MemoryExtractor
from alpha_agent.memory.models import (
    ConversationMessage,
    ExtractedMemoryCandidate,
    RetrievedContext,
    RuntimeTrace,
    SessionContextState,
)
from alpha_agent.memory.persistence import persist_candidates
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.context_compression import (
    CompressionBudget,
    CompressionContext,
    CompressionFocus,
    CompressionResult,
    ContextCompressor,
    DeterministicContextCompressor,
    select_compression_window,
)
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.runtime.prompt_builder import PromptBuilder, wrap_system_reminder
from alpha_agent.runtime.session_context import SessionContextManager, SessionContextProjection
from alpha_agent.runtime.tools import ExecutedToolResult, ToolExecutionError, ToolExecutor
from alpha_agent.tools.base import ToolCall
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of one agent turn."""

    response: str
    session_id: str
    debug: dict[str, Any] = field(default_factory=dict)


class AgentCanceledError(RuntimeError):
    """Raised when a session cancellation flag is observed during a turn."""

    def __init__(self, session_id: str, stage: str):
        super().__init__(f"Agent turn canceled for session {session_id} at {stage}")
        self.session_id = session_id
        self.stage = stage


class LLMCallError(RuntimeError):
    """Raised when the provider call fails, with retry metadata preserved."""

    def __init__(self, message: str, retry_count: int):
        super().__init__(message)
        self.retry_count = retry_count


class LLMRetryExhausted(LLMCallError):
    """Raised when transient LLM failures exceed the bounded retry policy."""


class ToolLoopLimitExceeded(RuntimeError):
    """Raised when the bounded official tool-call round would require another loop."""


class ToolProtocolError(RuntimeError):
    """Raised when provider-returned tool-call protocol data is incomplete."""


@dataclass(frozen=True)
class RetriedLLMCompletion:
    """LLM response plus retry metadata."""

    response: LLMResponse
    retry_count: int


@dataclass(frozen=True)
class AgentLoopResult:
    """Final model response plus accounting from the bounded agent loop."""

    response: LLMResponse
    provider_tool_messages: list[ConversationMessage]
    llm_round_count: int
    llm_retry_count: int
    tool_iteration_count: int
    tool_call_count: int
    provider_tool_call_count: int


class LLMCompletionService:
    """Bounded retry wrapper for a synchronous provider call."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        max_retries: int = 2,
        retry_sleep_seconds: float = 0.0,
    ):
        self.provider = provider
        self.max_retries = max(0, max_retries)
        self.retry_sleep_seconds = max(0.0, retry_sleep_seconds)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> RetriedLLMCompletion:
        """Complete once, retrying only known transient HTTP failures."""

        kwargs: dict[str, Any] = {}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                return RetriedLLMCompletion(
                    response=self.provider.complete(messages, **kwargs),
                    retry_count=attempt,
                )
            except Exception as exc:
                is_transient = self._is_transient(exc)
                if not is_transient:
                    raise LLMCallError(str(exc), retry_count=attempt) from exc
                if attempt >= self.max_retries:
                    raise LLMRetryExhausted(str(exc), retry_count=attempt) from exc
                if self.retry_sleep_seconds:
                    time.sleep(self.retry_sleep_seconds)
        raise LLMRetryExhausted("LLM retry policy exhausted", retry_count=self.max_retries)

    def _is_transient(self, exc: Exception) -> bool:
        if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
            return True
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            return status_code == 429 or status_code >= 500
        return False


class AlphaAgent:
    """Controllable synchronous agent runtime with explicit memory steps."""

    def __init__(
        self,
        store: MemoryStore,
        llm_provider: LLMProvider,
        retriever: MemoryRetriever,
        retrieval_limit: int = 8,
        prompt_builder: PromptBuilder | None = None,
        extractor: MemoryExtractor | None = None,
        tool_registry: ToolRegistry | None = None,
        max_llm_retries: int = 2,
        llm_retry_sleep_seconds: float = 0.0,
        max_tool_iterations: int = 8,
        max_llm_rounds: int | None = None,
        llm_debug_logging: bool = False,
        llm_trace_log_path: str | Path | None = None,
        context_compressor: ContextCompressor | None = None,
        context_max_prompt_tokens: int = 6000,
        context_compression_threshold_ratio: float = 0.85,
        context_recent_tail_messages: int = 8,
        context_min_summary_tokens: int = 128,
        context_max_summary_tokens: int = 512,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.retriever = retriever
        self.retrieval_limit = retrieval_limit
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.session_context = SessionContextManager(store)
        self.context_compressor = context_compressor or DeterministicContextCompressor()
        self.context_compression_budget = CompressionBudget(
            max_prompt_tokens=context_max_prompt_tokens,
            threshold_ratio=context_compression_threshold_ratio,
            recent_tail_messages=context_recent_tail_messages,
            min_summary_tokens=context_min_summary_tokens,
            max_summary_tokens=context_max_summary_tokens,
        )
        self.extractor = extractor or MemoryExtractor()
        self.episodic = EpisodicMemoryManager(store)
        self.semantic = SemanticMemoryManager(store)
        self.procedural = ProceduralMemoryManager(store)
        self.tool_registry = tool_registry or ToolRegistry()
        self.tool_executor = ToolExecutor(self.tool_registry)
        self.max_tool_iterations = max(0, max_tool_iterations)
        self.max_llm_rounds = (
            max(1, max_llm_rounds)
            if max_llm_rounds is not None
            else self.max_tool_iterations + 2
        )
        self.llm_completion = LLMCompletionService(
            llm_provider,
            max_retries=max_llm_retries,
            retry_sleep_seconds=llm_retry_sleep_seconds,
        )
        self.llm_debug_logging = llm_debug_logging
        self.llm_trace_log_path = (
            Path(llm_trace_log_path).expanduser() if llm_trace_log_path else None
        )
        self._canceled_sessions: set[str] = set()

    def cancel(self, session_id: str) -> None:
        """Request cancellation for an in-flight or next turn in a session."""

        self._canceled_sessions.add(session_id)

    def clear_cancel(self, session_id: str) -> None:
        """Clear a session cancellation flag."""

        self._canceled_sessions.discard(session_id)

    def is_canceled(self, session_id: str) -> bool:
        """Return whether a session has a pending cancellation flag."""

        return session_id in self._canceled_sessions

    def respond(
        self,
        user_message: str,
        session_id: str,
    ) -> AgentTurnResult:
        """Run one explicit agent turn."""

        turn_id = new_id("turn")
        debug: dict[str, Any] = {
            "turn_id": turn_id,
            "llm_retry_count": 0,
            "llm_round_count": 0,
            "tool_iteration_count": 0,
            "tool_call_count": 0,
            "provider_tool_call_count": 0,
        }
        try:
            self._check_canceled(session_id, "before_user_event")
            user_record = self._write_user_message(session_id, user_message)
            debug["user_message_id"] = user_record.id
            debug["user_message_ordinal"] = user_record.ordinal

            self._check_canceled(session_id, "before_retrieval")
            context = self._retrieve_memory(user_message, session_id)
            retrieved_ids = self._retrieved_ids(context)
            debug["retrieved_memory_ids"] = retrieved_ids

            session_context = self.session_context.load(
                session_id,
                before_ordinal=user_record.ordinal,
            )
            debug["session_context_compressed_until_ordinal"] = (
                session_context.compressed_until_ordinal
            )
            debug["session_context_message_count"] = len(
                session_context.uncompressed_messages
            )
            debug["session_context_has_summary"] = bool(session_context.summary)

            messages = self._build_prompt(
                user_message,
                context,
                session_context=session_context,
            )
            model_tools = self.tool_registry.to_llm_tool_definitions()
            model_tool_choice: LLMToolChoice | None = "auto" if model_tools else None
            prompt_token_estimate = self.prompt_builder.estimate_prompt_tokens(
                messages,
                tools=model_tools or None,
            )
            session_context, messages, prompt_token_estimate = (
                self._maybe_compress_initial_context(
                    session_id=session_id,
                    user_message=user_message,
                    user_ordinal=user_record.ordinal,
                    retrieved_context=context,
                    session_context=session_context,
                    messages=messages,
                    model_tools=model_tools or None,
                    prompt_token_estimate=prompt_token_estimate,
                    debug=debug,
                )
            )
            debug["prompt_token_estimate"] = prompt_token_estimate

            loop_result = self._run_agent_loop(
                session_id=session_id,
                messages=messages,
                model_tools=model_tools or None,
                model_tool_choice=model_tool_choice,
                initial_prompt_token_estimate=prompt_token_estimate,
                debug=debug,
            )
            llm_response = loop_result.response
            provider_tool_messages = loop_result.provider_tool_messages
            debug["provider"] = llm_response.provider
            debug["final_provider"] = llm_response.provider
            debug["llm_round_count"] = loop_result.llm_round_count
            debug["llm_retry_count"] = loop_result.llm_retry_count
            debug["tool_iteration_count"] = loop_result.tool_iteration_count
            debug["tool_call_count"] = loop_result.tool_call_count
            debug["provider_tool_call_count"] = loop_result.provider_tool_call_count
            debug["final_finish_reason"] = llm_response.finish_reason

            assistant_record = self._write_assistant_message(session_id, llm_response)
            debug["assistant_message_id"] = assistant_record.id
            debug["assistant_message_ordinal"] = assistant_record.ordinal

            extraction_source_message_ids = [
                user_record.id,
                assistant_record.id,
                *[message.id for message in provider_tool_messages],
            ]
            candidates = self._extract_memory(
                session_id=session_id,
                user_message=user_message,
                assistant_response=llm_response.content,
                source_message_ids=extraction_source_message_ids,
            )
            self._persist_extracted_memories(candidates)
            debug["extracted_memory_count"] = len(candidates)
            debug["consolidation"] = self._decide_consolidation_trigger(candidates)

            return AgentTurnResult(
                response=llm_response.content,
                session_id=session_id,
                debug=debug,
            )
        except AgentCanceledError as exc:
            self._emit_turn_failed(
                session_id=session_id,
                turn_id=turn_id,
                status="canceled",
                stage=exc.stage,
                error=exc,
                debug=debug,
            )
            self.clear_cancel(session_id)
            raise
        except LLMCallError as exc:
            debug["llm_retry_count"] = debug.get("llm_retry_count", 0) + exc.retry_count
            self._emit_turn_failed(
                session_id=session_id,
                turn_id=turn_id,
                status="failed",
                stage="llm",
                error=exc,
                debug=debug,
            )
            self.clear_cancel(session_id)
            raise
        except Exception as exc:
            self._emit_turn_failed(
                session_id=session_id,
                turn_id=turn_id,
                status="failed",
                stage=self._error_stage(exc),
                error=exc,
                debug=debug,
            )
            self.clear_cancel(session_id)
            raise
        finally:
            if session_id not in self._canceled_sessions:
                self.clear_cancel(session_id)

    def _write_user_message(
        self,
        session_id: str,
        user_message: str,
    ) -> ConversationMessage:
        return self.store.append_conversation_message(
            session_id=session_id,
            role="user",
            raw_content=user_message,
        )

    def _retrieve_memory(self, user_message: str, session_id: str) -> RetrievedContext:
        return self.retriever.retrieve_context(
            user_message,
            session_id,
            limit=self.retrieval_limit,
        )

    def _build_prompt(
        self,
        user_message: str,
        context: RetrievedContext,
        *,
        session_context: SessionContextProjection,
    ) -> list[ChatMessage]:
        return self.prompt_builder.build(
            user_message,
            context,
            session_context=session_context,
        )

    def _maybe_compress_initial_context(
        self,
        *,
        session_id: str,
        user_message: str,
        user_ordinal: int,
        retrieved_context: RetrievedContext,
        session_context: SessionContextProjection,
        messages: list[ChatMessage],
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        prompt_token_estimate: int,
        debug: dict[str, Any],
    ) -> tuple[SessionContextProjection, list[ChatMessage], int]:
        budget = self.context_compression_budget
        compression_context = CompressionContext(
            session_id=session_id,
            prompt_token_estimate=prompt_token_estimate,
            uncompressed_message_count=len(session_context.uncompressed_messages),
            has_previous_summary=bool(session_context.summary),
        )
        common_metadata = {
            "prompt_token_estimate_before": prompt_token_estimate,
            "threshold_tokens": budget.threshold_tokens,
            "max_prompt_tokens": budget.max_prompt_tokens,
            "threshold_ratio": budget.threshold_ratio,
            "recent_tail_messages": budget.effective_recent_tail_messages,
            "session_context_message_count": len(session_context.uncompressed_messages),
            "previous_compressed_until_ordinal": session_context.compressed_until_ordinal,
            "has_previous_summary": bool(session_context.summary),
        }
        try:
            should_compress = self.context_compressor.should_compress(
                compression_context,
                budget,
            )
        except Exception as exc:
            failed_metadata = {
                **common_metadata,
                "status": "failed",
                "stage": "decision",
                "error_type": type(exc).__name__,
            }
            self._emit_context_compression_trace(
                session_id=session_id,
                event_type="context_compression.failed",
                content=str(exc),
                metadata=failed_metadata,
            )
            self._record_context_compression_debug(
                debug,
                status="failed",
                metadata=failed_metadata,
            )
            raise

        if not should_compress:
            skipped_metadata = {
                **common_metadata,
                "status": "skipped",
                "reason": "below_threshold",
                "prompt_token_estimate_after": prompt_token_estimate,
            }
            self._emit_context_compression_trace(
                session_id=session_id,
                event_type="context_compression.skipped",
                content="Context compression skipped.",
                metadata=skipped_metadata,
            )
            self._record_context_compression_debug(
                debug,
                status="skipped",
                metadata=skipped_metadata,
            )
            return session_context, messages, prompt_token_estimate

        selection = select_compression_window(
            session_context.uncompressed_messages,
            recent_tail_messages=budget.recent_tail_messages,
        )
        if not selection.messages_to_compress:
            skipped_metadata = {
                **common_metadata,
                "status": "skipped",
                "reason": "no_compressible_messages",
                "prompt_token_estimate_after": prompt_token_estimate,
                "preserved_message_count": len(selection.preserved_messages),
            }
            self._emit_context_compression_trace(
                session_id=session_id,
                event_type="context_compression.skipped",
                content="Context compression skipped.",
                metadata=skipped_metadata,
            )
            self._record_context_compression_debug(
                debug,
                status="skipped",
                metadata=skipped_metadata,
            )
            return session_context, messages, prompt_token_estimate

        started_metadata = {
            **common_metadata,
            "status": "started",
            "compress_message_count": len(selection.messages_to_compress),
            "preserved_message_count": len(selection.preserved_messages),
            "candidate_compressed_until_ordinal": (
                selection.messages_to_compress[-1].ordinal
            ),
            "compressor": self.context_compressor.compression_version,
        }
        started_trace = self._emit_context_compression_trace(
            session_id=session_id,
            event_type="context_compression.started",
            content="Context compression started.",
            metadata=started_metadata,
        )
        previous_summary_source_message_ids = (
            session_context.state.summary_source_message_ids
            if session_context.state is not None
            else []
        )
        failure_stage = "compress"
        try:
            compression_result = self.context_compressor.compress(
                selection.messages_to_compress,
                previous_summary=session_context.summary,
                focus=CompressionFocus(
                    session_id=session_id,
                    current_user_message=user_message,
                    prompt_token_estimate=prompt_token_estimate,
                    budget=budget,
                    compressed_until_ordinal=session_context.compressed_until_ordinal,
                    previous_summary_source_message_ids=previous_summary_source_message_ids,
                ),
            )
            failure_stage = "validation"
            self._validate_context_compression_result(
                result=compression_result,
                selected_messages=selection.messages_to_compress,
                previous_summary_source_message_ids=previous_summary_source_message_ids,
                current_user_ordinal=user_ordinal,
            )
            failure_stage = "rebuild"
            now = utc_now_iso()
            provisional_state = SessionContextState(
                session_id=session_id,
                compressed_until_ordinal=compression_result.compressed_until_ordinal,
                summary=compression_result.summary,
                summary_source_message_ids=compression_result.summary_source_message_ids,
                compression_version=compression_result.compression_version,
                created_at=session_context.state.created_at
                if session_context.state is not None
                else now,
                updated_at=now,
                metadata=dict(compression_result.metadata),
            )
            provisional_projection = SessionContextProjection(
                state=provisional_state,
                uncompressed_messages=selection.preserved_messages,
                before_ordinal=user_ordinal,
            )
            rebuilt_messages = self._build_prompt(
                user_message,
                retrieved_context,
                session_context=provisional_projection,
            )
            rebuilt_prompt_token_estimate = self.prompt_builder.estimate_prompt_tokens(
                rebuilt_messages,
                tools=model_tools,
            )
            state_metadata = {
                **compression_result.metadata,
                "status": "completed",
                "reason": "prompt_budget_exceeded",
                "started_trace_id": started_trace.id,
                "prompt_token_estimate_before": prompt_token_estimate,
                "prompt_token_estimate_after": rebuilt_prompt_token_estimate,
                "threshold_tokens": budget.threshold_tokens,
                "max_prompt_tokens": budget.max_prompt_tokens,
                "threshold_ratio": budget.threshold_ratio,
                "recent_tail_messages": budget.effective_recent_tail_messages,
                "input_token_estimate": compression_result.input_token_estimate,
                "output_token_estimate": compression_result.output_token_estimate,
                "summary_source_message_count": len(
                    compression_result.summary_source_message_ids
                ),
                "preserved_message_count": len(selection.preserved_messages),
            }
            failure_stage = "persist"
            stored_state = self.store.upsert_session_context_state(
                SessionContextState(
                    session_id=session_id,
                    compressed_until_ordinal=compression_result.compressed_until_ordinal,
                    summary=compression_result.summary,
                    summary_source_message_ids=(
                        compression_result.summary_source_message_ids
                    ),
                    compression_version=compression_result.compression_version,
                    created_at=provisional_state.created_at,
                    updated_at=now,
                    metadata=state_metadata,
                )
            )
            failure_stage = "reload"
            rebuilt_context = self.session_context.load(
                session_id,
                before_ordinal=user_ordinal,
            )
            rebuilt_messages = self._build_prompt(
                user_message,
                retrieved_context,
                session_context=rebuilt_context,
            )
            rebuilt_prompt_token_estimate = self.prompt_builder.estimate_prompt_tokens(
                rebuilt_messages,
                tools=model_tools,
            )
            completed_metadata = {
                **state_metadata,
                "prompt_token_estimate_after": rebuilt_prompt_token_estimate,
                "compressed_until_ordinal": stored_state.compressed_until_ordinal,
                "compression_version": stored_state.compression_version,
            }
            self._emit_context_compression_trace(
                session_id=session_id,
                event_type="context_compression.completed",
                content="Context compression completed.",
                metadata=completed_metadata,
            )
            self._record_context_compression_debug(
                debug,
                status="completed",
                metadata=completed_metadata,
            )
            debug["session_context_compressed_until_ordinal"] = (
                rebuilt_context.compressed_until_ordinal
            )
            debug["session_context_message_count"] = len(
                rebuilt_context.uncompressed_messages
            )
            debug["session_context_has_summary"] = bool(rebuilt_context.summary)
            return rebuilt_context, rebuilt_messages, rebuilt_prompt_token_estimate
        except Exception as exc:
            failed_metadata = {
                **started_metadata,
                "status": "failed",
                "stage": failure_stage,
                "started_trace_id": started_trace.id,
                "error_type": type(exc).__name__,
            }
            self._emit_context_compression_trace(
                session_id=session_id,
                event_type="context_compression.failed",
                content=str(exc),
                metadata=failed_metadata,
            )
            self._record_context_compression_debug(
                debug,
                status="failed",
                metadata=failed_metadata,
            )
            raise

    def _validate_context_compression_result(
        self,
        *,
        result: CompressionResult,
        selected_messages: Sequence[ConversationMessage],
        previous_summary_source_message_ids: Sequence[str],
        current_user_ordinal: int,
    ) -> None:
        if not selected_messages:
            raise ValueError("context compression validation requires selected messages")

        selected_message_list = list(selected_messages)
        expected_compressed_until = selected_message_list[-1].ordinal
        if result.compressed_until_ordinal != expected_compressed_until:
            raise ValueError(
                "context compression compressed_until_ordinal must equal selected "
                "compressed prefix last ordinal: "
                f"expected {expected_compressed_until}, got "
                f"{result.compressed_until_ordinal}"
            )
        if result.compressed_until_ordinal >= current_user_ordinal:
            raise ValueError(
                "context compression compressed_until_ordinal must be lower than "
                f"current user ordinal {current_user_ordinal}, got "
                f"{result.compressed_until_ordinal}"
            )

        expected_source_message_ids = [
            *previous_summary_source_message_ids,
            *[message.id for message in selected_message_list],
        ]
        if result.summary_source_message_ids != expected_source_message_ids:
            raise ValueError(
                "context compression summary_source_message_ids must equal previous "
                "summary ids plus selected compressed message ids"
            )
        if not result.compression_version.strip():
            raise ValueError("context compression result must include compression_version")
        if not result.summary.strip():
            raise ValueError("context compression result summary must not be empty")

    def _emit_context_compression_trace(
        self,
        *,
        session_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=session_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def _record_context_compression_debug(
        self,
        debug: dict[str, Any],
        *,
        status: str,
        metadata: dict[str, Any],
    ) -> None:
        prompt_token_estimate_after = metadata.get(
            "prompt_token_estimate_after",
            metadata.get("prompt_token_estimate_before"),
        )
        debug["context_compression_status"] = status
        debug["context_compression"] = dict(metadata)
        debug["prompt_token_estimate_before_compression"] = metadata.get(
            "prompt_token_estimate_before"
        )
        debug["prompt_token_estimate_after_rebuild"] = prompt_token_estimate_after

    def _run_agent_loop(
        self,
        *,
        session_id: str,
        messages: list[ChatMessage],
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        initial_prompt_token_estimate: int,
        debug: dict[str, Any],
    ) -> AgentLoopResult:
        conversation_messages = list(messages)
        provider_tool_messages: list[ConversationMessage] = []
        llm_round_count = 0
        llm_retry_count = 0
        tool_iteration_count = 0
        tool_call_count = 0
        provider_tool_call_count = 0
        finalizing_reason: str | None = None

        while True:
            finalizing = finalizing_reason is not None
            tools = model_tools
            tool_choice = (
                "none" if finalizing and model_tools is not None else model_tool_choice
            )
            round_name = self._agent_loop_round_name(
                llm_round_count=llm_round_count,
                tool_iteration_count=tool_iteration_count,
                finalizing=finalizing,
            )
            prompt_token_estimate = (
                initial_prompt_token_estimate
                if llm_round_count == 0
                else self.prompt_builder.estimate_prompt_tokens(
                    conversation_messages,
                    tools=tools,
                )
            )
            completion = self._call_model(
                session_id=session_id,
                messages=conversation_messages,
                prompt_token_estimate=prompt_token_estimate,
                round_name=round_name,
                tools=tools,
                tool_choice=tool_choice,
            )
            llm_round_count += 1
            llm_retry_count += completion.retry_count
            response = completion.response
            if llm_round_count == 1:
                debug["initial_provider"] = response.provider
            debug["llm_retry_count"] = llm_retry_count
            debug["llm_round_count"] = llm_round_count
            debug["final_provider"] = response.provider
            debug["final_finish_reason"] = response.finish_reason

            if not self._response_requests_tools(response):
                return AgentLoopResult(
                    response=response,
                    provider_tool_messages=provider_tool_messages,
                    llm_round_count=llm_round_count,
                    llm_retry_count=llm_retry_count,
                    tool_iteration_count=tool_iteration_count,
                    tool_call_count=tool_call_count,
                    provider_tool_call_count=provider_tool_call_count,
                )

            if finalizing_reason is not None:
                self._emit_tool_loop_event(
                    session_id=session_id,
                    event_type="tool_loop.finalization_failed",
                    content="No-tools finalization returned tool calls.",
                    metadata={
                        "reason": finalizing_reason,
                        "finish_reason": response.finish_reason,
                        "llm_round_count": llm_round_count,
                        "tool_iteration_count": tool_iteration_count,
                        "tool_call_count": len(response.tool_calls),
                    },
                )
                raise ToolLoopLimitExceeded(
                    "tool_loop_limit_exceeded: finalization returned tool calls"
                )

            provider_tool_calls = self.tool_executor.normalize_calls(response.tool_calls)
            self._validate_provider_tool_calls(provider_tool_calls, response)
            if tool_iteration_count >= self.max_tool_iterations:
                finalizing_reason = "max_tool_iterations"
                conversation_messages.append(
                    self._begin_tool_loop_finalization(
                        session_id=session_id,
                        reason=finalizing_reason,
                        llm_round_count=llm_round_count,
                        tool_iteration_count=tool_iteration_count,
                    )
                )
                continue
            if llm_round_count >= self.max_llm_rounds:
                finalizing_reason = "max_llm_rounds"
                conversation_messages.append(
                    self._begin_tool_loop_finalization(
                        session_id=session_id,
                        reason=finalizing_reason,
                        llm_round_count=llm_round_count,
                        tool_iteration_count=tool_iteration_count,
                    )
                )
                continue

            provider_tool_call_count += len(provider_tool_calls)
            debug["provider_tool_call_count"] = provider_tool_call_count
            provider_tool_call_message = self._write_assistant_tool_call_message(
                session_id=session_id,
                calls=provider_tool_calls,
                llm_response=response,
            )
            provider_tool_messages.append(provider_tool_call_message)
            conversation_messages.append(
                self.prompt_builder.conversation_message_to_chat(provider_tool_call_message)
            )

            provider_results = self._execute_tool_calls(
                session_id=session_id,
                calls=provider_tool_calls,
                recover_errors=True,
            )
            provider_result_messages = self._write_tool_result_messages(
                session_id=session_id,
                results=provider_results,
            )
            provider_tool_messages.extend(provider_result_messages)
            tool_call_count += len(provider_results)
            tool_iteration_count += 1
            debug["tool_call_count"] = tool_call_count
            debug["tool_iteration_count"] = tool_iteration_count

            conversation_messages.extend(
                self.prompt_builder.conversation_message_to_chat(message)
                for message in provider_result_messages
            )

    def _agent_loop_round_name(
        self,
        *,
        llm_round_count: int,
        tool_iteration_count: int,
        finalizing: bool,
    ) -> str:
        if finalizing:
            return "finalize"
        if llm_round_count == 0:
            return "initial"
        return f"tool_result_{tool_iteration_count}"

    def _call_model(
        self,
        *,
        session_id: str,
        messages: list[ChatMessage],
        prompt_token_estimate: int,
        round_name: str,
        tools: Sequence[LLMToolDefinitionInput] | None,
        tool_choice: LLMToolChoice | None,
    ) -> RetriedLLMCompletion:
        self._check_canceled(session_id, "before_llm")
        llm_call_id = new_id("llm")
        request_log = (
            _llm_request_log(messages=messages, tools=tools, tool_choice=tool_choice)
            if self.llm_debug_logging
            else None
        )
        started_metadata: dict[str, Any] = {
            "llm_call_id": llm_call_id,
            "provider": self.llm_provider.name,
            "round": round_name,
            "prompt_token_estimate": prompt_token_estimate,
            "max_retries": self.llm_completion.max_retries,
            "tool_count": len(tools) if tools is not None else 0,
            "tool_choice": tool_choice,
        }
        if request_log is not None:
            started_metadata["request"] = request_log
        started_trace = self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.started",
            content="LLM call started.",
            metadata=started_metadata,
        )
        if request_log is not None:
            self._append_llm_trace(
                event="llm.request",
                metadata={
                    "llm_call_id": llm_call_id,
                    "session_id": session_id,
                    "round": round_name,
                    "provider": self.llm_provider.name,
                    "prompt_token_estimate": prompt_token_estimate,
                    "request": request_log,
                },
            )
        try:
            completion = self.llm_completion.complete(
                list(messages),
                tools=tools,
                tool_choice=tool_choice,
            )
        except LLMCallError as exc:
            self.store.append_runtime_trace(
                session_id=session_id,
                event_type="llm.failed",
                content=str(exc),
                metadata={
                    "llm_call_id": llm_call_id,
                    "provider": self.llm_provider.name,
                    "round": round_name,
                    "retry_count": exc.retry_count,
                    "error_type": type(exc).__name__,
                    "request": request_log,
                },
            )
            if request_log is not None:
                self._append_llm_trace(
                    event="llm.error",
                    metadata={
                        "llm_call_id": llm_call_id,
                        "session_id": session_id,
                        "round": round_name,
                        "provider": self.llm_provider.name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "retry_count": exc.retry_count,
                        "request": request_log,
                    },
                )
            raise
        self._check_canceled(session_id, "after_llm")
        response_log = (
            _llm_response_log(completion.response) if self.llm_debug_logging else None
        )
        response_metadata = _llm_metadata_for_event(
            completion.response.metadata,
            include_raw_payloads=self.llm_debug_logging,
        )
        completed_metadata: dict[str, Any] = {
            "llm_call_id": llm_call_id,
            "provider": completion.response.provider,
            "model": completion.response.model,
            "round": round_name,
            "retry_count": completion.retry_count,
            "started_trace_id": started_trace.id,
            "finish_reason": completion.response.finish_reason,
            "tool_call_count": len(completion.response.tool_calls),
            "response_metadata": response_metadata,
        }
        if response_log is not None:
            completed_metadata["response"] = response_log
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.completed",
            content="LLM call completed.",
            metadata=completed_metadata,
        )
        if response_log is not None:
            self._append_llm_trace(
                event="llm.response",
                metadata={
                    "llm_call_id": llm_call_id,
                    "retry_count": completion.retry_count,
                    "response": response_log,
                },
            )
        return completion

    def _append_llm_trace(self, *, event: str, metadata: dict[str, Any]) -> None:
        if not self.llm_debug_logging or self.llm_trace_log_path is None:
            return
        self.llm_trace_log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": utc_now_iso(),
            "level": "debug",
            "event": event,
            "metadata": _json_safe(metadata),
        }
        with self.llm_trace_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def _write_assistant_message(
        self,
        session_id: str,
        llm_response: LLMResponse,
    ) -> ConversationMessage:
        return self.store.append_conversation_message(
            session_id=session_id,
            role="assistant",
            raw_content=llm_response.content,
            provider_metadata={
                "provider": llm_response.provider,
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "metadata": _llm_metadata_for_event(
                    llm_response.metadata,
                    include_raw_payloads=self.llm_debug_logging,
                ),
            },
        )

    def _write_assistant_tool_call_message(
        self,
        *,
        session_id: str,
        calls: list[ToolCall],
        llm_response: LLMResponse | None = None,
    ) -> ConversationMessage:
        provider_metadata: dict[str, Any] = {}
        if llm_response is not None:
            provider_metadata.update(
                {
                    "provider": llm_response.provider,
                    "model": llm_response.model,
                    "finish_reason": llm_response.finish_reason,
                    "metadata": _llm_metadata_for_event(
                        llm_response.metadata,
                        include_raw_payloads=self.llm_debug_logging,
                    ),
                }
            )
        tool_calls = [dict(self._wire_tool_call(call)) for call in calls]
        return self.store.append_conversation_message(
            session_id=session_id,
            role="assistant",
            raw_content=llm_response.content if llm_response is not None else "",
            tool_calls=tool_calls,
            provider_metadata=provider_metadata,
            metadata={
                "tool_call_ids": [self._required_tool_call_id(call) for call in calls],
            },
        )

    def _write_tool_result_messages(
        self,
        *,
        session_id: str,
        results: list[ExecutedToolResult],
    ) -> list[ConversationMessage]:
        messages: list[ConversationMessage] = []
        for item in results:
            messages.append(
                self.store.append_conversation_message(
                    session_id=session_id,
                    role="tool",
                    raw_content=item.trace.content,
                    tool_call_id=self._required_tool_call_id(item.call),
                    tool_result_id=item.trace.id,
                    provider_metadata={
                        "tool_name": item.result.name,
                    },
                    metadata={
                        "trace_id": item.trace.id,
                        "result_metadata": dict(item.result.metadata),
                    },
                )
            )
        return messages

    def _extract_memory(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        source_message_ids: list[str],
    ) -> list[ExtractedMemoryCandidate]:
        candidates = self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
            source_event_ids=source_message_ids,
        )
        type_counts: dict[str, int] = {}
        for candidate in candidates:
            type_counts[candidate.type] = type_counts.get(candidate.type, 0) + 1
        self.store.append_runtime_trace(
            session_id=session_id,
            event_type="memory.extracted",
            content=deterministic_json(type_counts),
            metadata={
                "extracted_memory_count": len(candidates),
                "candidate_type_counts": type_counts,
                "source_message_ids": source_message_ids,
            },
        )
        return candidates

    def _persist_extracted_memories(self, candidates: list[ExtractedMemoryCandidate]) -> None:
        persist_candidates(self.store, candidates)

    def _decide_consolidation_trigger(
        self,
        candidates: list[ExtractedMemoryCandidate],
    ) -> dict[str, Any]:
        high_salience_count = sum(1 for candidate in candidates if candidate.salience >= 0.75)
        should_consolidate = high_salience_count >= 3
        return {
            "should_consolidate": should_consolidate,
            "reason": "high_salience_batch" if should_consolidate else "below_threshold",
            "high_salience_count": high_salience_count,
        }

    def _emit_turn_failed(
        self,
        *,
        session_id: str,
        turn_id: str,
        status: str,
        stage: str,
        error: Exception,
        debug: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=session_id,
            event_type="turn.failed",
            content=str(error),
            metadata={
                "turn_id": turn_id,
                "status": status,
                "stage": stage,
                "error_type": type(error).__name__,
                "error_code": self._error_code(error),
                "retry_count": debug.get("llm_retry_count", 0),
                "llm_round_count": debug.get("llm_round_count", 0),
                "tool_iteration_count": debug.get("tool_iteration_count", 0),
                "final_finish_reason": debug.get("final_finish_reason"),
                "tool_call_count": debug.get("tool_call_count", 0),
                "provider_tool_call_count": debug.get("provider_tool_call_count", 0),
            },
        )

    def _begin_tool_loop_finalization(
        self,
        *,
        session_id: str,
        reason: str,
        llm_round_count: int,
        tool_iteration_count: int,
    ) -> ChatMessage:
        self._emit_tool_loop_event(
            session_id=session_id,
            event_type="tool_loop.finalizing",
            content="Tool loop limit reached; requesting final answer with tools disabled.",
            metadata={
                "reason": reason,
                "llm_round_count": llm_round_count,
                "tool_iteration_count": tool_iteration_count,
                "max_llm_rounds": self.max_llm_rounds,
                "max_tool_iterations": self.max_tool_iterations,
            },
        )
        return self._tool_loop_finalization_message(reason=reason)

    def _tool_loop_finalization_message(self, *, reason: str) -> ChatMessage:
        return {
            "role": "user",
            "content": wrap_system_reminder(
                f"Tool loop stopped because {reason}. Summarize the current progress "
                "and provide the best final answer from available information. "
                "Do not call tools."
            ),
        }

    def _emit_tool_loop_event(
        self,
        *,
        session_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=session_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def _response_requests_tools(self, response: LLMResponse) -> bool:
        return response.finish_reason == "tool_calls" or bool(response.tool_calls)

    def _validate_provider_tool_calls(
        self,
        calls: list[ToolCall],
        response: LLMResponse,
    ) -> None:
        if not calls:
            raise ToolProtocolError(
                f"Provider returned finish_reason={response.finish_reason} "
                "but no normalized tool calls"
            )
        for call in calls:
            if not call.id:
                raise ToolProtocolError(f"Provider tool call for {call.name} is missing an id")

    def _wire_tool_call(self, call: ToolCall) -> ChatCompletionToolCall:
        raw_arguments = call.metadata.get("raw_arguments")
        arguments = (
            raw_arguments if isinstance(raw_arguments, str) else deterministic_json(call.arguments)
        )
        return {
            "id": self._required_tool_call_id(call),
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": arguments,
            },
        }

    def _required_tool_call_id(self, call: ToolCall) -> str:
        if not call.id:
            raise ToolExecutionError(call, f"Provider tool call for {call.name} is missing an id")
        return call.id

    def _execute_tool_calls(
        self,
        *,
        session_id: str,
        calls: list[ToolCall],
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        return self.tool_executor.execute(
            calls=calls,
            write_trace=lambda event_type, content, metadata: self._write_tool_trace(
                session_id=session_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            ),
            check_canceled=lambda stage: self._check_canceled(session_id, stage),
            recover_errors=recover_errors,
        )

    def _write_tool_trace(
        self,
        *,
        session_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=session_id,
            event_type=event_type,
            content=content,
            metadata=metadata,
        )

    def _check_canceled(self, session_id: str, stage: str) -> None:
        if self.is_canceled(session_id):
            raise AgentCanceledError(session_id, stage)

    def _error_stage(self, exc: Exception) -> str:
        if isinstance(exc, ToolExecutionError):
            return "tool"
        if isinstance(exc, ToolProtocolError):
            return "tool_protocol"
        if isinstance(exc, ToolLoopLimitExceeded):
            return "tool_loop_limit_exceeded"
        return "runtime"

    def _error_code(self, exc: Exception) -> str | None:
        if isinstance(exc, ToolProtocolError):
            return "tool_protocol_violation"
        if isinstance(exc, ToolLoopLimitExceeded):
            return "tool_loop_limit_exceeded"
        return None

    def _retrieved_ids(self, context: RetrievedContext) -> dict[str, list[str]]:
        return {
            "episodic": [item.id for item in context.episodic_memories],
            "semantic": [item.id for item in context.semantic_memories],
            "procedural": [item.id for item in context.procedural_memories],
        }


def _llm_request_log(
    *,
    messages: list[ChatMessage],
    tools: Sequence[LLMToolDefinitionInput] | None,
    tool_choice: LLMToolChoice | None,
) -> dict[str, Any]:
    return {
        "messages": _json_safe(messages),
        "tools": _json_safe(list(tools)) if tools is not None else None,
        "tool_choice": _json_safe(tool_choice),
    }


def _llm_response_log(response: LLMResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "model": response.model,
        "provider": response.provider,
        "metadata": _json_safe(
            {
                key: value
                for key, value in response.metadata.items()
                if key != "request_payload"
            }
        ),
        "tool_calls": [tool_call.to_dict() for tool_call in response.tool_calls],
        "finish_reason": response.finish_reason,
    }


def _llm_metadata_for_event(
    metadata: dict[str, Any],
    *,
    include_raw_payloads: bool,
) -> dict[str, Any]:
    if include_raw_payloads:
        return _json_safe(metadata)
    return _json_safe(
        {
            key: value
            for key, value in metadata.items()
            if key not in {"request_payload", "response_payload"}
        }
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_json_safe(item) for item in value]
    return str(value)
