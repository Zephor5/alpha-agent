"""Explicit personal agent runtime."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path
from threading import Lock, RLock
from typing import Any

import httpx

from alpha_agent.cognition.coordinator import (
    LockBusy,
    LoopAcquireRequest,
    LoopCoordinator,
)
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops.feedback_attribution import (
    FeedbackAttributionJob,
    recalled_beliefs_for_previous_turn,
)
from alpha_agent.cognition.models import (
    BeliefScope,
    CognitiveEvent,
    CognitiveEventKind,
    CounterpartId,
    EventId,
    Instant,
    LoopPriority,
    Reference,
    SituationId,
    StimulusKind,
    Subject,
    SummaryKind,
    counterpart_ref,
    situation_ref,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.config import (
    DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS,
    AlphaConfig,
    LLMContextConfig,
)
from alpha_agent.llm.base import (
    ChatCompletionToolCall,
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMResponseFormat,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.llm.tracing import LLMTraceLogger, traced_llm_complete
from alpha_agent.llm.tracing import llm_metadata_summary as _llm_metadata_summary
from alpha_agent.llm.tracing import llm_request_summary as _llm_request_summary
from alpha_agent.runtime.chat_messages import (
    estimate_chat_tokens,
    source_message_to_chat,
    wrap_system_reminder,
)
from alpha_agent.runtime.context_budget import ContextBudgetEstimate, estimate_context_budget
from alpha_agent.runtime.context_handover import (
    HandoverCompressionResult,
    HandoverExtractionJob,
    compress_session_context,
)
from alpha_agent.runtime.counterpart_router import CounterpartRouter
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.runtime.prompt_builder import (
    AnswerPromptFrame,
    build_answer_prompt_frame,
    build_answer_prompt_messages,
    build_answer_prompt_messages_from_frame,
    default_runtime_system_message,
)
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.runtime.tools import ExecutedToolResult, ToolExecutionError, ToolExecutor
from alpha_agent.state.models import RuntimeTrace, SessionMessage, SessionSummarySnapshot
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import ToolCall, TurnToolState, tool_output_kind
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_CONTEXT_KEY
from alpha_agent.tools.memory_recall import MEMORY_RECALL_CONTEXT_KEY
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso

_SELF_MEMORY_SUMMARY_REF = Reference("subject", "subject:self")


@dataclass(frozen=True)
class AgentTurnContext:
    """Runtime-owned identity and source refs for one accepted agent turn."""

    turn_id: str
    session_id: str
    started_at: Instant
    source: Reference | None
    counterpart: Reference | None
    user_message_id: str | None = None
    turn_received_event_id: str | None = None


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


class ContextWindowExceededError(RuntimeError):
    """Raised when a pending user message cannot fit in the configured context."""


CompactExtractionSubmitter = Callable[
    [HandoverExtractionJob, Sequence[LLMToolDefinitionInput] | None],
    object,
]
FeedbackAttributionSubmitter = Callable[[FeedbackAttributionJob], object]


_SESSION_TURN_LOCKS: dict[str, RLock] = {}
_SESSION_TURN_LOCKS_GUARD = Lock()
# Memory remains LLM-directed: the runtime exposes memory tools and context,
# but it does not perform hidden recall or hidden writes before the model acts.


@contextmanager
def _serialized_session_turn(session_id: str) -> Iterator[None]:
    with _SESSION_TURN_LOCKS_GUARD:
        lock = _SESSION_TURN_LOCKS.setdefault(session_id, RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


@dataclass(frozen=True)
class RetriedLLMCompletion:
    response: LLMResponse
    retry_count: int
    llm_call_id: str = ""
    started_trace_id: str = ""
    completed_trace_id: str = ""


@dataclass(frozen=True)
class AgentLoopResult:
    response: LLMResponse
    provider_tool_messages: list[SessionMessage]
    provider_tool_calls: list[ToolCall]
    tool_results: list[Any]
    llm_round_count: int
    llm_retry_count: int
    tool_iteration_count: int
    tool_call_count: int
    provider_tool_call_count: int
    provider_tool_trace_ids: list[str]
    llm_call_ids: list[str]
    llm_trace_ids: list[str]
    tool_cognitive_event_ids: list[str]


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
        response_format: LLMResponseFormat | None = None,
        trace_logger: LLMTraceLogger | None = None,
        trace_metadata: Mapping[str, Any] | None = None,
    ) -> RetriedLLMCompletion:
        for attempt in range(self.max_retries + 1):
            try:
                return RetriedLLMCompletion(
                    response=traced_llm_complete(
                        self.provider,
                        messages,
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                        trace_logger=trace_logger,
                        trace_metadata={
                            **dict(trace_metadata or {}),
                            "retry_attempt": attempt,
                        },
                    ),
                    retry_count=attempt,
                )
            except Exception as exc:
                if not self._is_transient(exc):
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
            return exc.response.status_code == 429 or exc.response.status_code >= 500
        return False


class AlphaAgent:
    """Controllable synchronous agent runtime with no long-term memory dependency."""

    def __init__(
        self,
        store: StateStore,
        llm_provider: LLMProvider,
        tool_registry: ToolRegistry | None = None,
        max_llm_retries: int = 2,
        llm_retry_sleep_seconds: float = 0.0,
        max_tool_iterations: int = 8,
        max_llm_rounds: int | None = None,
        tool_output_dir: str | Path | None = None,
        llm_context_config: LLMContextConfig | None = None,
        max_context_tokens: int | None = None,
        event_log: EventLog | None = None,
        coordinator: LoopCoordinator | None = None,
        compact_extraction_submitter: CompactExtractionSubmitter | None = None,
        llm_trace_logger: LLMTraceLogger | None = None,
        config: AlphaConfig | None = None,
        feedback_attribution_submitter: FeedbackAttributionSubmitter | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.config = config or _default_alpha_config(store)
        self.session_context = SessionContextAssembler(store)
        self.llm_context_config = llm_context_config or LLMContextConfig()
        self.max_context_tokens = max_context_tokens or _default_max_context_tokens(
            getattr(llm_provider, "name", None)
        )
        self.tool_registry = (
            build_tool_registry(self.config) if tool_registry is None else tool_registry
        )
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
        self.llm_trace_logger = llm_trace_logger or LLMTraceLogger.from_config(self.config)
        self.llm_trace_log_path = self.llm_trace_logger.trace_log_path
        self.tool_output_dir = (
            Path(tool_output_dir).expanduser()
            if tool_output_dir is not None
            else self.store.db_path.parent / "tool-results"
        )
        self.event_log = event_log or SQLiteEventLog(store)
        self.emitter = EventEmitter(self.event_log)
        self.counterpart_projection = CounterpartProjection(store)
        self.counterpart_router = CounterpartRouter(
            self.event_log,
            counterpart_projection=self.counterpart_projection,
        )
        self.coordinator = coordinator or LoopCoordinator(SUBJECT_SELF)
        self.compact_extraction_submitter = compact_extraction_submitter
        self.feedback_attribution_submitter = feedback_attribution_submitter
        self._canceled_sessions: set[str] = set()

    def cancel(self, session_id: str) -> None:
        self._canceled_sessions.add(session_id)

    def clear_cancel(self, session_id: str) -> None:
        self._canceled_sessions.discard(session_id)

    def is_canceled(self, session_id: str) -> bool:
        return session_id in self._canceled_sessions

    def respond(
        self,
        user_message: str,
        session_id: str,
        source_metadata: Mapping[str, Any] | None = None,
    ) -> AgentTurnResult:
        """Run one explicit agent turn."""

        acquire_request = LoopAcquireRequest(
            loop_name="runtime_turn",
            priority=LoopPriority.REACTIVE,
            max_chunk_duration=timedelta(seconds=120),
        )
        acquire_context = self.coordinator.try_acquire(acquire_request)
        try:
            acquire_context.__enter__()
        except LockBusy as busy:
            return AgentTurnResult(
                response=self._compose_busy_message(busy),
                session_id=session_id,
                debug={"busy": True, "holder": busy.holder, "since": busy.since},
            )
        turn_context = _serialized_session_turn(session_id)
        turn_context.__enter__()

        agent_turn: AgentTurnContext | None = None
        debug: dict[str, Any] = {}
        try:
            turn_id = new_id("turn")
            source_ref = Reference("session", session_id)
            counterpart = self._session_counterpart_ref(session_id, source_metadata)
            agent_turn = AgentTurnContext(
                turn_id=turn_id,
                session_id=session_id,
                started_at=Instant(utc_now_iso()),
                source=source_ref,
                counterpart=counterpart,
            )
            debug = {
                "turn_id": turn_id,
                "llm_retry_count": 0,
                "llm_round_count": 0,
                "tool_iteration_count": 0,
                "tool_call_count": 0,
                "provider_tool_call_count": 0,
                "note": "runtime-owned foreground turn",
            }
            self._check_canceled(session_id, "before_user_event")
            model_tools = self.tool_registry.to_llm_tool_definitions()
            self._session_summary_snapshots(session_id, counterpart)
            self._run_pre_user_context_maintenance(
                turn_context=agent_turn,
                session_id=session_id,
                pending_user_message=user_message,
                model_tools=model_tools or None,
                model_tool_choice="auto" if model_tools else None,
                debug=debug,
            )
            user_record = self.store.append_session_message(
                session_id=session_id,
                kind="user_message",
                llm_role="user",
                raw_content=user_message,
                source_metadata=dict(source_metadata or {}),
                metadata=_turn_metadata(agent_turn),
            )
            debug["user_message_id"] = user_record.id
            debug["user_message_ordinal"] = user_record.ordinal
            received_event = self._emit_turn_received(
                turn_context=agent_turn,
                user_message=user_message,
                user_record=user_record,
                source_metadata=source_metadata,
            )
            agent_turn = replace(
                agent_turn,
                user_message_id=user_record.id,
                turn_received_event_id=str(received_event.id),
            )
            debug["turn_received_event_id"] = str(received_event.id)

            session_context = self.session_context.load(session_id)
            debug["chat_history_message_count"] = len(session_context.chat_messages)
            prompt_frame = self._answer_prompt_frame(session_id)
            messages = self._rebuild_runtime_llm_messages(
                session_id=session_id,
                prompt_frame=prompt_frame,
            )
            self._submit_feedback_attribution(
                turn_context=agent_turn,
                user_record=user_record,
                user_message=user_message,
                prompt_messages=messages,
                debug=debug,
            )
            prompt_token_estimate = estimate_chat_tokens(messages, tools=model_tools or None)
            debug["prompt_token_estimate"] = prompt_token_estimate
            debug["renderer"] = "runtime_session_history"
            memory_state = CognitionStateStore(self.store)
            belief_projection = memory_state.beliefs
            memory_propose_context = {
                "turn_id": agent_turn.turn_id,
                "session_id": session_id,
                "user_message_id": user_record.id,
                "turn_received_event_id": agent_turn.turn_received_event_id,
                "emitter": self.emitter,
                "subject": Subject(),
                "situation": situation_ref(SituationId(f"situation:{agent_turn.turn_id}")),
                "counterpart": counterpart,
                "memory_state": memory_state,
                "belief_projection": belief_projection,
            }
            memory_recall_context = {
                "session_id": session_id,
                "counterpart": counterpart,
                "belief_projection": belief_projection,
            }
            loop_result = self._run_agent_loop(
                turn_context=agent_turn,
                session_id=session_id,
                messages=messages,
                prompt_frame=prompt_frame,
                model_tools=model_tools or None,
                model_tool_choice="auto" if model_tools else None,
                initial_prompt_token_estimate=prompt_token_estimate,
                memory_propose_context=memory_propose_context,
                memory_recall_context=memory_recall_context,
                debug=debug,
            )
            llm_response = loop_result.response
            assistant_record = self._write_assistant_message(
                session_id,
                llm_response,
                turn_context=agent_turn,
            )
            debug["assistant_message_id"] = assistant_record.id
            debug["assistant_message_ordinal"] = assistant_record.ordinal
            acted_event = self._emit_turn_acted(
                turn_context=agent_turn,
                assistant_record=assistant_record,
                loop_result=loop_result,
            )
            debug["acted_event_id"] = str(acted_event.id)
            sources_event = self._emit_turn_sources_recorded(
                turn_context=agent_turn,
                user_record=user_record,
                assistant_record=assistant_record,
                loop_result=loop_result,
                cognitive_event_ids=[str(received_event.id), str(acted_event.id)],
            )
            debug["turn_sources_event_id"] = str(sources_event.id)
            return AgentTurnResult(
                response=llm_response.content,
                session_id=session_id,
                debug=debug,
            )
        except AgentCanceledError as exc:
            if agent_turn is not None:
                self._emit_turn_failed(agent_turn, "canceled", exc.stage, exc, debug)
            self.clear_cancel(session_id)
            raise
        except LLMCallError as exc:
            debug["llm_retry_count"] = debug.get("llm_retry_count", 0) + exc.retry_count
            if agent_turn is not None:
                self._emit_turn_failed(agent_turn, "failed", "llm", exc, debug)
            self.clear_cancel(session_id)
            raise
        except Exception as exc:
            if agent_turn is not None:
                self._emit_turn_failed(
                    agent_turn,
                    "failed",
                    self._error_stage(exc),
                    exc,
                    debug,
                )
            self.clear_cancel(session_id)
            raise
        finally:
            turn_context.__exit__(None, None, None)
            acquire_context.__exit__(None, None, None)
            if session_id not in self._canceled_sessions:
                self.clear_cancel(session_id)

    def _run_pre_user_context_maintenance(
        self,
        *,
        turn_context: AgentTurnContext,
        session_id: str,
        pending_user_message: str,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        debug: dict[str, Any],
    ) -> None:
        pending_message: ChatMessage = {"role": "user", "content": pending_user_message}
        summary_snapshots = self.store.list_session_summary_snapshots(session_id)
        pending_only_estimate = self._estimate_context_budget(
            build_answer_prompt_messages(
                summary_snapshots=summary_snapshots,
                session_history=[],
                current_turn_messages=[pending_message],
                system_message=self._default_system_message(),
            ),
            tools=model_tools,
        )
        if pending_only_estimate.used_context_tokens > self.max_context_tokens:
            self._raise_pending_user_too_large(pending_only_estimate)

        planning_messages = build_answer_prompt_messages(
            summary_snapshots=summary_snapshots,
            session_history=[],
            current_turn_messages=[pending_message],
            system_message=self._default_system_message(),
        )
        projected_messages = self._source_prompt_messages(
            session_id=session_id,
            extra_source_messages=[pending_message],
        )
        estimate = self._estimate_context_budget(projected_messages, tools=model_tools)
        debug["pre_user_context_used_tokens"] = estimate.used_context_tokens
        debug["pre_user_context_remaining_tokens"] = estimate.remaining_context_tokens

        if self._needs_tool_truncation(estimate):
            truncation = self.session_context.truncate_tool_context_if_needed(
                session_id,
                context_config=self.llm_context_config,
                max_context_tokens=self.max_context_tokens,
                tools=model_tools,
                planning_messages=planning_messages,
            )
            debug["pre_user_tool_truncation_triggered"] = truncation.triggered
            debug["pre_user_tool_truncated_message_ids"] = list(
                truncation.truncated_message_ids
            )
            projected_messages = self._source_prompt_messages(
                session_id=session_id,
                extra_source_messages=[pending_message],
            )
            estimate = self._estimate_context_budget(projected_messages, tools=model_tools)

        if self._needs_handover(estimate) and self.session_context.load(
            session_id
        ).source_messages:
            compression_messages = self._source_prompt_messages(session_id=session_id)
            result = compress_session_context(
                session_id=session_id,
                assembler=self.session_context,
                llm_provider=self.llm_provider,
                llm_messages=compression_messages,
                tools=model_tools,
                tool_choice="none" if model_tools else None,
                trace_metadata=_turn_metadata(turn_context),
            )
            debug["pre_user_compressed_message_id"] = result.message.id
            debug["pre_user_compression_point_ordinal"] = result.compression_point_ordinal
            self._submit_compact_extraction(
                result,
                tools=model_tools,
                debug=debug,
                debug_prefix="pre_user",
            )
            projected_messages = self._source_prompt_messages(
                session_id=session_id,
                extra_source_messages=[pending_message],
            )
            estimate = self._estimate_context_budget(projected_messages, tools=model_tools)

        if estimate.used_context_tokens > self.max_context_tokens:
            self._raise_pending_user_too_large(estimate)
        debug["pre_user_context_after_maintenance_used_tokens"] = estimate.used_context_tokens
        debug["pre_user_context_after_maintenance_remaining_tokens"] = (
            estimate.remaining_context_tokens
        )

    def _run_tool_result_context_maintenance(
        self,
        *,
        turn_context: AgentTurnContext,
        session_id: str,
        prompt_frame: AnswerPromptFrame,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        debug: dict[str, Any],
    ) -> list[ChatMessage]:
        llm_messages = self._rebuild_runtime_llm_messages(
            session_id=session_id,
            prompt_frame=prompt_frame,
        )
        estimate = self._estimate_context_budget(llm_messages, tools=model_tools)
        debug["tool_loop_context_used_tokens"] = estimate.used_context_tokens
        debug["tool_loop_context_remaining_tokens"] = estimate.remaining_context_tokens

        if self._needs_tool_truncation(estimate):
            self.session_context.truncate_tool_context_if_needed(
                session_id,
                context_config=self.llm_context_config,
                max_context_tokens=self.max_context_tokens,
                tools=model_tools,
                planning_messages=[
                    prompt_frame.system_message,
                    *prompt_frame.summary_context_messages,
                ],
            )
            llm_messages = self._rebuild_runtime_llm_messages(
                session_id=session_id,
                prompt_frame=prompt_frame,
            )
            estimate = self._estimate_context_budget(llm_messages, tools=model_tools)

        if self._needs_handover(estimate) and self.session_context.load(
            session_id
        ).source_messages:
            result = compress_session_context(
                session_id=session_id,
                assembler=self.session_context,
                llm_provider=self.llm_provider,
                llm_messages=llm_messages,
                tools=model_tools,
                tool_choice="none" if model_tools else None,
                trace_metadata=_turn_metadata(turn_context),
            )
            debug["tool_loop_compressed_message_id"] = result.message.id
            debug["tool_loop_compression_point_ordinal"] = result.compression_point_ordinal
            self._submit_compact_extraction(
                result,
                tools=model_tools,
                debug=debug,
                debug_prefix="tool_loop",
            )
            llm_messages = self._rebuild_runtime_llm_messages(
                session_id=session_id,
                prompt_frame=prompt_frame,
            )
        return llm_messages

    def _submit_compact_extraction(
        self,
        result: HandoverCompressionResult,
        *,
        tools: Sequence[LLMToolDefinitionInput] | None,
        debug: dict[str, Any],
        debug_prefix: str,
    ) -> None:
        submitter = self.compact_extraction_submitter
        if submitter is None:
            return
        try:
            submitted = submitter(result.extraction_job, tuple(tools or ()))
            debug[f"{debug_prefix}_compact_extraction_submitted"] = (
                submitted if isinstance(submitted, bool) else True
            )
        except Exception as exc:
            debug[f"{debug_prefix}_compact_extraction_submitted"] = False
            debug[f"{debug_prefix}_compact_extraction_submit_error"] = str(exc)
            try:
                self.store.append_runtime_trace(
                    session_id=result.message.session_id,
                    event_type="direct_compact_extraction.submit_failed",
                    content="Direct compact extraction submit failed.",
                    metadata={
                        "compressed_message_id": result.message.id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            except Exception:
                return

    def _submit_feedback_attribution(
        self,
        *,
        turn_context: AgentTurnContext,
        user_record: SessionMessage,
        user_message: str,
        prompt_messages: Sequence[ChatMessage],
        debug: dict[str, Any],
    ) -> None:
        recalled_beliefs = recalled_beliefs_for_previous_turn(
            self.store,
            turn_context.session_id,
            user_record.ordinal,
        )
        debug["feedback_attribution_recalled_belief_count"] = len(recalled_beliefs)
        if not recalled_beliefs:
            debug["feedback_attribution_submitted"] = False
            return

        recall_tool_message_ids = _stable_unique_strings(
            message_id
            for handle in recalled_beliefs
            for message_id in handle.source_tool_message_ids
        )
        debug["feedback_attribution_belief_ids"] = [
            handle.belief_id for handle in recalled_beliefs
        ]
        debug["feedback_attribution_recall_tool_message_ids"] = list(
            recall_tool_message_ids
        )
        debug["feedback_attribution_prompt_message_count"] = len(prompt_messages)

        submitter = self.feedback_attribution_submitter
        if submitter is None:
            debug["feedback_attribution_submitted"] = False
            debug["feedback_attribution_submitter_configured"] = False
            return

        job = FeedbackAttributionJob(
            session_id=turn_context.session_id,
            turn_id=turn_context.turn_id,
            turn_received_event_id=turn_context.turn_received_event_id or "",
            user_message_id=user_record.id,
            user_message_text=user_message,
            prompt_messages=prompt_messages,
            recalled_beliefs=tuple(recalled_beliefs),
            recall_tool_message_ids=recall_tool_message_ids,
        )
        debug["feedback_attribution_submitter_configured"] = True
        try:
            submitted = submitter(job)
        except Exception as exc:
            debug["feedback_attribution_submitted"] = False
            debug["feedback_attribution_submit_error"] = str(exc)
            self._append_feedback_attribution_submit_trace(
                job,
                reason="submitter_exception",
                error=exc,
            )
            return

        submitted_bool = submitted if isinstance(submitted, bool) else True
        debug["feedback_attribution_submitted"] = submitted_bool
        if not submitted_bool:
            debug["feedback_attribution_submit_error"] = "submitter returned false"
            self._append_feedback_attribution_submit_trace(
                job,
                reason="submitter_returned_false",
            )

    def _append_feedback_attribution_submit_trace(
        self,
        job: FeedbackAttributionJob,
        *,
        reason: str,
        error: Exception | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "turn_id": job.turn_id,
            "turn_received_event_id": job.turn_received_event_id,
            "user_message_id": job.user_message_id,
            "belief_ids": [handle.belief_id for handle in job.recalled_beliefs],
            "recall_tool_message_ids": list(job.recall_tool_message_ids),
            "reason": reason,
        }
        if error is not None:
            metadata["error_type"] = type(error).__name__
            metadata["error"] = str(error)
        try:
            self.store.append_runtime_trace(
                session_id=job.session_id,
                event_type="feedback_attribution.submit_failed",
                content="Feedback attribution submit failed.",
                metadata=metadata,
            )
        except Exception:
            return

    def _source_prompt_messages(
        self,
        *,
        session_id: str,
        extra_source_messages: Sequence[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        return build_answer_prompt_messages(
            summary_snapshots=self.store.list_session_summary_snapshots(session_id),
            session_history=self.session_context.load(session_id).chat_messages,
            current_turn_messages=extra_source_messages or (),
            system_message=self._default_system_message(),
        )

    def _rebuild_runtime_llm_messages(
        self,
        *,
        session_id: str,
        prompt_frame: AnswerPromptFrame,
    ) -> list[ChatMessage]:
        return build_answer_prompt_messages_from_frame(
            frame=prompt_frame,
            session_history=self.session_context.load(session_id).chat_messages,
        )

    def _estimate_context_budget(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None,
    ) -> ContextBudgetEstimate:
        return estimate_context_budget(
            messages,
            tools=tools,
            context_config=self.llm_context_config,
            max_context_tokens=self.max_context_tokens,
        )

    def _needs_tool_truncation(self, estimate: ContextBudgetEstimate) -> bool:
        return (
            estimate.used_context_tokens
            > estimate.max_context_tokens
            * self.llm_context_config.tool_truncate_threshold_ratio
        )

    def _needs_handover(self, estimate: ContextBudgetEstimate) -> bool:
        return (
            estimate.used_context_tokens
            > estimate.max_context_tokens
            * self.llm_context_config.handover_compress_threshold_ratio
            or estimate.remaining_context_tokens
            < self.llm_context_config.minimum_remaining_tokens
        )

    def _default_system_message(self) -> ChatMessage:
        return default_runtime_system_message()

    def _answer_prompt_frame(self, session_id: str) -> AnswerPromptFrame:
        return build_answer_prompt_frame(
            summary_snapshots=self.store.list_session_summary_snapshots(session_id),
            system_message=self._default_system_message(),
        )

    def _session_counterpart_ref(
        self,
        session_id: str,
        source_metadata: Mapping[str, Any] | None,
    ) -> Reference | None:
        binding = self.store.get_session_counterpart(session_id)
        if binding is not None:
            return counterpart_ref(CounterpartId(binding.counterpart_id))
        if _stimulus_kind_from_metadata(source_metadata) == StimulusKind.SELF_SIGNAL:
            # Internal self-signals are not external user turns. They may carry the
            # counterpart that originated the goal, but must not fall back to the
            # default main-user counterpart when no source counterpart exists.
            source_counterpart = _self_signal_counterpart_ref(source_metadata)
            if source_counterpart is None:
                return None
            binding = self.store.create_session_counterpart(
                session_id=session_id,
                counterpart_id=source_counterpart.id,
                source_metadata=dict(source_metadata or {}),
            )
            return counterpart_ref(CounterpartId(binding.counterpart_id))
        routed = self.counterpart_router.upsert_from_source_metadata(
            source_metadata,
            emitter=self.emitter,
        )
        if routed is None:
            return None
        binding = self.store.create_session_counterpart(
            session_id=session_id,
            counterpart_id=routed.id,
            source_metadata=dict(source_metadata or {}),
        )
        return counterpart_ref(CounterpartId(binding.counterpart_id))

    def _session_summary_snapshots(
        self,
        session_id: str,
        counterpart: Reference | None,
    ) -> list[SessionSummarySnapshot]:
        self._session_self_memory_snapshot(session_id)
        self._session_counterpart_profile_snapshot(session_id, counterpart)
        return self.store.list_session_summary_snapshots(session_id)

    def _session_self_memory_snapshot(
        self,
        session_id: str,
    ) -> SessionSummarySnapshot | None:
        snapshot = self.store.get_session_summary_snapshot(
            session_id,
            SummaryKind.SELF_MEMORY_SUMMARY.value,
        )
        # Stable summary context is intentionally session-stable: once a session
        # has a snapshot, later background summary updates do not rewrite this
        # prompt prefix mid-conversation.
        if snapshot is not None:
            return snapshot
        summary = BeliefProjection(self.store).latest_summary(
            summary_kind=SummaryKind.SELF_MEMORY_SUMMARY,
            scope=BeliefScope.SELF,
        )
        if summary is None:
            return None
        content = str(summary.content).strip()
        if not content:
            return None
        target = summary.about[0] if summary.about else _SELF_MEMORY_SUMMARY_REF
        return self.store.create_session_summary_snapshot(
            session_id=session_id,
            summary_kind=SummaryKind.SELF_MEMORY_SUMMARY.value,
            target_kind=target.kind,
            target_id=target.id,
            source_belief_id=str(summary.id),
            content=content,
        )

    def _session_counterpart_profile_snapshot(
        self,
        session_id: str,
        counterpart: Reference | None,
    ) -> SessionSummarySnapshot | None:
        snapshot = self.store.get_session_summary_snapshot(
            session_id,
            SummaryKind.COUNTERPART_PROFILE.value,
        )
        if snapshot is not None or counterpart is None:
            return snapshot
        profile = BeliefProjection(self.store).latest_summary(
            summary_kind=SummaryKind.COUNTERPART_PROFILE,
            about=counterpart,
        )
        if profile is None:
            return None
        content = str(profile.content).strip()
        if not content:
            return None
        return self.store.create_session_summary_snapshot(
            session_id=session_id,
            summary_kind=SummaryKind.COUNTERPART_PROFILE.value,
            target_kind=counterpart.kind,
            target_id=counterpart.id,
            source_belief_id=str(profile.id),
            content=content,
        )

    def _raise_pending_user_too_large(self, estimate: ContextBudgetEstimate) -> None:
        raise ContextWindowExceededError(
            "pending user message exceeds configured context window "
            f"(used={estimate.used_context_tokens}, max={estimate.max_context_tokens})"
        )

    def _compose_busy_message(self, busy: LockBusy) -> str:
        return (
            f"Agent is currently {busy.holder} "
            f"(started at {busy.since}); please retry shortly."
        )

    def _run_agent_loop(
        self,
        *,
        turn_context: AgentTurnContext,
        session_id: str,
        messages: list[ChatMessage],
        prompt_frame: AnswerPromptFrame,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        initial_prompt_token_estimate: int,
        memory_propose_context: Mapping[str, Any] | None,
        memory_recall_context: Mapping[str, Any] | None,
        debug: dict[str, Any],
    ) -> AgentLoopResult:
        llm_messages = list(messages)
        provider_tool_messages: list[SessionMessage] = []
        provider_tool_calls_seen: list[ToolCall] = []
        provider_tool_trace_ids: list[str] = []
        llm_call_ids: list[str] = []
        llm_trace_ids: list[str] = []
        tool_cognitive_event_ids: list[str] = []
        tool_results_seen: list[Any] = []
        llm_round_count = llm_retry_count = tool_iteration_count = 0
        tool_call_count = provider_tool_call_count = 0
        finalizing_reason: str | None = None
        turn_tool_state = TurnToolState()

        while True:
            finalizing = finalizing_reason is not None
            round_name = (
                "finalize"
                if finalizing
                else (
                    "initial"
                    if llm_round_count == 0
                    else f"tool_result_{tool_iteration_count}"
                )
            )
            completion = self._call_model(
                turn_context=turn_context,
                session_id=session_id,
                messages=llm_messages,
                prompt_token_estimate=(
                    initial_prompt_token_estimate
                    if llm_round_count == 0
                    else estimate_chat_tokens(
                        llm_messages,
                        tools=model_tools,
                    )
                ),
                round_name=round_name,
                tools=model_tools,
                tool_choice="none" if finalizing and model_tools is not None else model_tool_choice,
            )
            llm_round_count += 1
            llm_retry_count += completion.retry_count
            llm_call_ids.append(completion.llm_call_id)
            llm_trace_ids.extend(
                [
                    item
                    for item in [completion.started_trace_id, completion.completed_trace_id]
                    if item
                ]
            )
            response = completion.response
            debug["llm_round_count"] = llm_round_count
            debug["llm_retry_count"] = llm_retry_count

            if not self._response_requests_tools(response):
                debug.update(
                    {
                        "provider": response.provider,
                        "final_provider": response.provider,
                        "llm_round_count": llm_round_count,
                        "llm_retry_count": llm_retry_count,
                        "tool_iteration_count": tool_iteration_count,
                        "tool_call_count": tool_call_count,
                        "provider_tool_call_count": provider_tool_call_count,
                        "provider_tool_call_ids": [
                            call.id for call in provider_tool_calls_seen if call.id
                        ],
                        "provider_tool_message_ids": [
                            message.id
                            for message in provider_tool_messages
                            if message.kind == "tool_message"
                        ],
                        "provider_tool_trace_ids": list(provider_tool_trace_ids),
                        "llm_call_ids": list(llm_call_ids),
                        "llm_trace_ids": list(llm_trace_ids),
                        "tool_cognitive_event_ids": list(tool_cognitive_event_ids),
                        "final_finish_reason": response.finish_reason,
                    }
                )
                return AgentLoopResult(
                    response=response,
                    provider_tool_messages=provider_tool_messages,
                    provider_tool_calls=provider_tool_calls_seen,
                    tool_results=tool_results_seen,
                    llm_round_count=llm_round_count,
                    llm_retry_count=llm_retry_count,
                    tool_iteration_count=tool_iteration_count,
                    tool_call_count=tool_call_count,
                    provider_tool_call_count=provider_tool_call_count,
                    provider_tool_trace_ids=provider_tool_trace_ids,
                    llm_call_ids=llm_call_ids,
                    llm_trace_ids=llm_trace_ids,
                    tool_cognitive_event_ids=tool_cognitive_event_ids,
                )
            if finalizing_reason is not None:
                self._emit_tool_loop_event(
                    turn_context,
                    session_id,
                    "tool_loop.finalization_failed",
                    "No-tools finalization returned tool calls.",
                    {"reason": finalizing_reason},
                )
                raise ToolLoopLimitExceeded("tool_loop_limit_exceeded")

            provider_tool_calls = self.tool_executor.normalize_calls(response.tool_calls)
            self._validate_provider_tool_calls(provider_tool_calls, response)
            if tool_iteration_count >= self.max_tool_iterations:
                finalizing_reason = "max_tool_iterations"
                llm_messages.append(self._tool_loop_finalization_message(finalizing_reason))
                continue
            if llm_round_count >= self.max_llm_rounds:
                finalizing_reason = "max_llm_rounds"
                llm_messages.append(self._tool_loop_finalization_message(finalizing_reason))
                continue

            provider_tool_call_count += len(provider_tool_calls)
            provider_tool_calls_seen.extend(provider_tool_calls)
            provider_tool_call_message = self._write_assistant_tool_call_message(
                session_id=session_id,
                turn_context=turn_context,
                calls=provider_tool_calls,
                llm_response=response,
            )
            provider_tool_messages.append(provider_tool_call_message)
            llm_messages.append(source_message_to_chat(provider_tool_call_message))
            provider_results = self._execute_tool_calls(
                turn_context=turn_context,
                session_id=session_id,
                calls=provider_tool_calls,
                turn_tool_state=turn_tool_state,
                memory_propose_context={
                    **dict(memory_propose_context or {}),
                    "llm_call_id": completion.llm_call_id,
                    "llm_trace_ids": [
                        item
                        for item in [
                            completion.started_trace_id,
                            completion.completed_trace_id,
                        ]
                        if item
                    ],
                },
                memory_recall_context=memory_recall_context,
                recover_errors=True,
            )
            provider_tool_trace_ids.extend(item.trace.id for item in provider_results)
            tool_results_seen.extend(item.result for item in provider_results)
            for item in provider_results:
                tool_cognitive_event_ids.extend(
                    _string_list(item.result.metadata.get("cognitive_event_ids"))
                )
            if tool_cognitive_event_ids:
                debug["tool_cognitive_event_ids"] = list(tool_cognitive_event_ids)
            provider_result_messages = self._write_tool_result_messages(
                session_id=session_id,
                turn_context=turn_context,
                results=provider_results,
            )
            provider_tool_messages.extend(provider_result_messages)
            tool_call_count += len(provider_results)
            tool_iteration_count += 1
            llm_messages.extend(
                source_message_to_chat(message) for message in provider_result_messages
            )
            llm_messages = self._run_tool_result_context_maintenance(
                turn_context=turn_context,
                session_id=session_id,
                prompt_frame=prompt_frame,
                model_tools=model_tools,
                model_tool_choice=model_tool_choice,
                debug=debug,
            )

    def _call_model(
        self,
        *,
        turn_context: AgentTurnContext,
        session_id: str,
        messages: list[ChatMessage],
        prompt_token_estimate: int,
        round_name: str,
        tools: Sequence[LLMToolDefinitionInput] | None,
        tool_choice: LLMToolChoice | None,
    ) -> RetriedLLMCompletion:
        self._check_canceled(session_id, "before_llm")
        llm_call_id = new_id("llm")
        trace_metadata = {
            "turn_id": turn_context.turn_id,
            "llm_call_id": llm_call_id,
            "session_id": session_id,
        }
        started_trace = self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.started",
            content="LLM call started.",
            metadata={
                "turn_id": turn_context.turn_id,
                "llm_call_id": llm_call_id,
                "provider": self.llm_provider.name,
                "round": round_name,
                "prompt_token_estimate": prompt_token_estimate,
                "max_retries": self.llm_completion.max_retries,
                "tool_count": len(tools) if tools is not None else 0,
                "tool_choice": tool_choice,
                "request_summary": _llm_request_summary(
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                ),
            },
        )
        try:
            completion = self.llm_completion.complete(
                list(messages),
                tools=tools,
                tool_choice=tool_choice,
                trace_logger=self.llm_trace_logger,
                trace_metadata=trace_metadata,
            )
        except LLMCallError as exc:
            self.store.append_runtime_trace(
                session_id=session_id,
                event_type="llm.failed",
                content=str(exc),
                metadata={
                    "turn_id": turn_context.turn_id,
                    "llm_call_id": llm_call_id,
                    "provider": self.llm_provider.name,
                    "round": round_name,
                    "retry_count": exc.retry_count,
                    "error_type": type(exc).__name__,
                },
            )
            raise
        self._check_canceled(session_id, "after_llm")
        completed_trace = self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.completed",
            content="LLM call completed.",
            metadata={
                "turn_id": turn_context.turn_id,
                "llm_call_id": llm_call_id,
                "provider": completion.response.provider,
                "model": completion.response.model,
                "round": round_name,
                "retry_count": completion.retry_count,
                "started_trace_id": started_trace.id,
                "finish_reason": completion.response.finish_reason,
                "tool_call_count": len(completion.response.tool_calls),
                "response_metadata": _llm_metadata_summary(completion.response.metadata),
            },
        )
        return RetriedLLMCompletion(
            response=completion.response,
            retry_count=completion.retry_count,
            llm_call_id=llm_call_id,
            started_trace_id=started_trace.id,
            completed_trace_id=completed_trace.id,
        )

    def _write_assistant_message(
        self,
        session_id: str,
        llm_response: LLMResponse,
        *,
        turn_context: AgentTurnContext,
    ) -> SessionMessage:
        return self.store.append_session_message(
            session_id=session_id,
            kind="assistant_message",
            llm_role="assistant",
            raw_content=llm_response.content,
            reasoning_content=llm_response.reasoning_content,
            provider_metadata={
                "provider": llm_response.provider,
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "metadata": _llm_metadata_summary(llm_response.metadata),
            },
            metadata=_turn_metadata(turn_context),
        )

    def _write_assistant_tool_call_message(
        self,
        *,
        session_id: str,
        turn_context: AgentTurnContext,
        calls: list[ToolCall],
        llm_response: LLMResponse,
    ) -> SessionMessage:
        return self.store.append_session_message(
            session_id=session_id,
            kind="assistant_message",
            llm_role="assistant",
            raw_content=llm_response.content,
            reasoning_content=llm_response.reasoning_content,
            tool_calls=[dict(self._wire_tool_call(call)) for call in calls],
            provider_metadata={
                "provider": llm_response.provider,
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "metadata": _llm_metadata_summary(llm_response.metadata),
            },
            metadata={
                "turn_id": turn_context.turn_id,
                "tool_call_ids": [self._required_tool_call_id(call) for call in calls],
            },
        )

    def _write_tool_result_messages(
        self,
        *,
        session_id: str,
        turn_context: AgentTurnContext,
        results: list[ExecutedToolResult],
    ) -> list[SessionMessage]:
        return [
            self.store.append_session_message(
                session_id=session_id,
                kind="tool_message",
                llm_role="tool",
                raw_content=item.trace.content,
                tool_call_id=self._required_tool_call_id(item.call),
                tool_result_id=item.trace.id,
                provider_metadata={"tool_name": item.result.name},
                metadata={
                    "turn_id": turn_context.turn_id,
                    "trace_id": item.trace.id,
                    "result_metadata": dict(item.result.metadata),
                    "tool_output_kind": tool_output_kind(item.result.output),
                },
            )
            for item in results
        ]

    def _emit_turn_sources_recorded(
        self,
        *,
        turn_context: AgentTurnContext,
        user_record: SessionMessage,
        assistant_record: SessionMessage,
        loop_result: AgentLoopResult,
        cognitive_event_ids: list[str],
    ) -> CognitiveEvent:
        provider_tool_message_ids = [message.id for message in loop_result.provider_tool_messages]
        provider_tool_trace_ids = list(loop_result.provider_tool_trace_ids)
        llm_call_ids = list(loop_result.llm_call_ids)
        llm_trace_ids = list(loop_result.llm_trace_ids)
        tool_cognitive_event_ids = list(loop_result.tool_cognitive_event_ids)
        source_refs = [
            Reference("session_message", user_record.id),
            Reference("session_message", assistant_record.id),
            *[Reference("session_message", item) for item in provider_tool_message_ids],
            *[Reference("runtime_trace", item) for item in provider_tool_trace_ids],
            *[Reference("runtime_trace", item) for item in llm_trace_ids],
        ]
        return self.emitter.emit(
            CognitiveEventKind.TURN_SOURCES_RECORDED,
            inputs=[Reference("agent_turn", turn_context.turn_id)],
            outputs=source_refs,
            rationale="Recorded persisted session and trace ids for the runtime turn.",
            causal_parents=[EventId(cognitive_event_ids[-1])] if cognitive_event_ids else [],
            payload={
                "turn_id": turn_context.turn_id,
                "session_id": turn_context.session_id,
                "user_message_id": user_record.id,
                "assistant_message_id": assistant_record.id,
                "provider_tool_message_ids": provider_tool_message_ids,
                "provider_tool_trace_ids": provider_tool_trace_ids,
                "llm_call_ids": llm_call_ids,
                "llm_trace_ids": llm_trace_ids,
                "cognitive_event_ids": list(cognitive_event_ids),
                "tool_cognitive_event_ids": tool_cognitive_event_ids,
            },
        )

    def _emit_turn_received(
        self,
        *,
        turn_context: AgentTurnContext,
        user_message: str,
        user_record: SessionMessage,
        source_metadata: Mapping[str, Any] | None,
    ) -> CognitiveEvent:
        source_refs = [
            Reference("session", turn_context.session_id),
            Reference("session_message", user_record.id),
        ]
        return self.emitter.emit(
            CognitiveEventKind.PERCEIVED,
            inputs=[turn_context.source] if turn_context.source is not None else [],
            outputs=[
                Reference("agent_turn", turn_context.turn_id),
                Reference("perception", f"perception:{turn_context.turn_id}"),
                Reference("session_message", user_record.id),
            ],
            rationale="Recorded accepted runtime turn input.",
            payload={
                "turn_id": turn_context.turn_id,
                "session_id": turn_context.session_id,
                "stimulus_kind": _stimulus_kind_from_metadata(source_metadata).value,
                "source": turn_context.source.to_record()
                if turn_context.source is not None
                else {"kind": "session", "id": turn_context.session_id},
                "from_counterpart": turn_context.counterpart.to_record()
                if turn_context.counterpart is not None
                else None,
                # Raw text remains in session_messages; cognition events keep refs
                # and digests so projections can recover content without duplicating it.
                "source_refs": [ref.to_record() for ref in source_refs],
                "content_digest": _content_digest(user_message),
                "content_length": len(user_message),
            },
        )

    def _emit_turn_acted(
        self,
        *,
        turn_context: AgentTurnContext,
        assistant_record: SessionMessage,
        loop_result: AgentLoopResult,
    ) -> CognitiveEvent:
        tool_call_ids = [
            self._required_tool_call_id(call) for call in loop_result.provider_tool_calls
        ]
        outputs = [
            Reference("session_message", assistant_record.id),
            *[Reference("llm_call", item) for item in loop_result.llm_call_ids],
            *[Reference("runtime_trace", item) for item in loop_result.llm_trace_ids],
            *[Reference("runtime_trace", item) for item in loop_result.provider_tool_trace_ids],
            *[Reference("tool_call", item) for item in tool_call_ids],
            *[
                Reference("cognitive_event", item)
                for item in loop_result.tool_cognitive_event_ids
            ],
        ]
        return self.emitter.emit(
            CognitiveEventKind.ACTED,
            inputs=[Reference("agent_turn", turn_context.turn_id)],
            outputs=outputs,
            rationale="Recorded runtime model and tool-loop outcome.",
            causal_parents=[EventId(turn_context.turn_received_event_id)]
            if turn_context.turn_received_event_id
            else [],
            payload={
                "turn_id": turn_context.turn_id,
                "session_id": turn_context.session_id,
                "assistant_message_id": assistant_record.id,
                "response_text_digest": _content_digest(loop_result.response.content),
                "response_text_length": len(loop_result.response.content),
                "llm_call_ids": list(loop_result.llm_call_ids),
                "llm_trace_ids": list(loop_result.llm_trace_ids),
                "tool_call_ids": tool_call_ids,
                "tool_names": [call.name for call in loop_result.provider_tool_calls],
                "tool_result_trace_ids": list(loop_result.provider_tool_trace_ids),
                "tool_cognitive_event_ids": list(loop_result.tool_cognitive_event_ids),
            },
        )

    def _emit_turn_failed(
        self,
        turn_context: AgentTurnContext,
        status: str,
        stage: str,
        error: Exception,
        debug: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=turn_context.session_id,
            event_type="turn.failed",
            content=str(error),
            metadata={
                "turn_id": turn_context.turn_id,
                "status": status,
                "stage": stage,
                "error_type": type(error).__name__,
                "error_code": self._error_code(error),
                "retry_count": debug.get("llm_retry_count", 0),
                "llm_round_count": debug.get("llm_round_count", 0),
                "tool_iteration_count": debug.get("tool_iteration_count", 0),
                "tool_call_count": debug.get("tool_call_count", 0),
                "provider_tool_call_count": debug.get("provider_tool_call_count", 0),
            },
        )

    def _tool_loop_finalization_message(self, reason: str) -> ChatMessage:
        return {
            "role": "user",
            "content": wrap_system_reminder(
                f"Tool loop stopped because {reason}. Summarize the current progress "
                "and provide the best final answer from available information. Do not call tools."
            ),
        }

    def _emit_tool_loop_event(
        self,
        turn_context: AgentTurnContext,
        session_id: str,
        event_type: str,
        content: str,
        metadata: dict[str, Any],
    ) -> RuntimeTrace:
        return self.store.append_runtime_trace(
            session_id=session_id,
            event_type=event_type,
            content=content,
            metadata={**metadata, "turn_id": turn_context.turn_id},
        )

    def _response_requests_tools(self, response: LLMResponse) -> bool:
        return response.finish_reason == "tool_calls" or bool(response.tool_calls)

    def _validate_provider_tool_calls(self, calls: list[ToolCall], response: LLMResponse) -> None:
        if not calls:
            raise ToolProtocolError(
                "Provider returned "
                f"finish_reason={response.finish_reason} "
                "but no normalized tool calls"
            )
        for call in calls:
            if not call.id:
                raise ToolProtocolError(f"Provider tool call for {call.name} is missing an id")

    def _wire_tool_call(self, call: ToolCall) -> ChatCompletionToolCall:
        raw_arguments = call.metadata.get("raw_arguments")
        return {
            "id": self._required_tool_call_id(call),
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": raw_arguments
                if isinstance(raw_arguments, str)
                else deterministic_json(call.arguments),
            },
        }

    def _required_tool_call_id(self, call: ToolCall) -> str:
        if not call.id:
            raise ToolExecutionError(call, f"Provider tool call for {call.name} is missing an id")
        return call.id

    def _execute_tool_calls(
        self,
        *,
        turn_context: AgentTurnContext,
        session_id: str,
        calls: list[ToolCall],
        turn_tool_state: TurnToolState,
        memory_propose_context: Mapping[str, Any] | None = None,
        memory_recall_context: Mapping[str, Any] | None = None,
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        return self.tool_executor.execute(
            calls=calls,
            session_id=session_id,
            output_dir=self.tool_output_dir,
            extensions={
                MEMORY_PROPOSE_CONTEXT_KEY: dict(memory_propose_context or {}),
                MEMORY_RECALL_CONTEXT_KEY: dict(memory_recall_context or {}),
            },
            turn_state=turn_tool_state,
            write_trace=lambda event_type, content, metadata: self.store.append_runtime_trace(
                session_id=session_id,
                event_type=event_type,
                content=content,
                metadata={**metadata, "turn_id": turn_context.turn_id},
            ),
            check_canceled=lambda stage: self._check_canceled(session_id, stage),
            recover_errors=recover_errors,
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


def _default_max_context_tokens(provider_name: object) -> int:
    provider_key = str(provider_name or "openai-compatible")
    return DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS.get(
        provider_key,
        DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS["openai-compatible"],
    )


def _default_alpha_config(store: StateStore) -> AlphaConfig:
    base_dir = store.db_path.parent
    return AlphaConfig(
        db_path=store.db_path,
        log_dir=base_dir / "logs",
        gateway_status_path=base_dir / "gateway-status.json",
    )


def _turn_metadata(turn_context: AgentTurnContext) -> dict[str, Any]:
    return {"turn_id": turn_context.turn_id}


def _content_digest(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]


def _stimulus_kind_from_metadata(source_metadata: Mapping[str, Any] | None) -> StimulusKind:
    raw = source_metadata.get("stimulus_kind") if source_metadata is not None else None
    if isinstance(raw, str):
        try:
            return StimulusKind(raw)
        except ValueError:
            return StimulusKind.USER_MESSAGE
    return StimulusKind.USER_MESSAGE


def _self_signal_counterpart_ref(source_metadata: Mapping[str, Any] | None) -> Reference | None:
    raw = source_metadata.get("counterpart_id") if source_metadata is not None else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    return counterpart_ref(CounterpartId(raw.strip()))


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return [str(item) for item in value if item is not None]


def _stable_unique_strings(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if not value.strip() or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)
