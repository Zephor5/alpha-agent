"""Explicit personal agent runtime backed by the reactive cognition tick."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import timedelta
from pathlib import Path
from threading import Lock, RLock
from typing import Any, cast

import httpx

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.coordinator import (
    LockBusy,
    LoopAcquireRequest,
    LoopCoordinator,
)
from alpha_agent.cognition.counterpart_profile import active_counterpart_digest
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    EventId,
    Instant,
    LoopPriority,
    Reference,
    Stimulus,
    StimulusKind,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.render import (
    COUNTERPART_PROFILE_LABEL,
    RenderResult,
    estimate_chat_tokens,
    render_counterpart_profile,
    source_message_to_chat,
    wrap_system_reminder,
)
from alpha_agent.cognition.render.view import CognitionView
from alpha_agent.cognition.stages.effector import Effector
from alpha_agent.cognition.stages.types import Outcome
from alpha_agent.cognition.threads import StimulusRouter
from alpha_agent.config import DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS, LLMContextConfig
from alpha_agent.llm.base import (
    ChatCompletionToolCall,
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.context_budget import ContextBudgetEstimate, estimate_context_budget
from alpha_agent.runtime.context_handover import compress_session_context
from alpha_agent.runtime.counterpart_router import CounterpartRouter
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.runtime.tools import ExecutedToolResult, ToolExecutionError, ToolExecutor
from alpha_agent.state.models import RuntimeTrace, SessionMessage, SessionProfileSnapshot
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import ToolCall, tool_output_kind
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_CONTEXT_KEY
from alpha_agent.tools.memory_recall import MEMORY_RECALL_CONTEXT_KEY
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


class ContextWindowExceededError(RuntimeError):
    """Raised when a pending user message cannot fit in the configured context."""


_SESSION_TURN_LOCKS: dict[str, RLock] = {}
_SESSION_TURN_LOCKS_GUARD = Lock()
_DEFAULT_RUNTIME_SYSTEM_MESSAGE: ChatMessage = {
    "role": "system",
    "content": (
        "Identity: Alpha Agent.\n"
        "Use the current reactive context and answer concisely. "
        "Call tools only when they are useful. "
        "When stable counterpart profile context is present, it is already visible near "
        "the start of the prompt. Use memory_recall for explicit long-term belief "
        "lookups, and use memory_propose only for explicit long-term memory write "
        "proposals."
    ),
}


@contextmanager
def _serialized_session_turn(session_id: str) -> Iterator[None]:
    with _SESSION_TURN_LOCKS_GUARD:
        lock = _SESSION_TURN_LOCKS.setdefault(session_id, RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _copy_chat_message(message: ChatMessage) -> ChatMessage:
    return cast(ChatMessage, dict(message))


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


@dataclass(frozen=True)
class PromptFrame:
    system_message: ChatMessage
    prefix_context_messages: list[ChatMessage]
    suffix_context_messages: list[ChatMessage]


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
        kwargs: dict[str, Any] = {}
        if tools is not None:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        for attempt in range(self.max_retries + 1):
            try:
                return RetriedLLMCompletion(
                    response=self.provider.complete(messages, **kwargs),
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
        llm_debug_logging: bool = False,
        llm_trace_log_path: str | Path | None = None,
        tool_output_dir: str | Path | None = None,
        llm_context_config: LLMContextConfig | None = None,
        max_context_tokens: int | None = None,
        event_log: EventLog | None = None,
        coordinator: LoopCoordinator | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.session_context = SessionContextAssembler(store)
        self.llm_context_config = llm_context_config or LLMContextConfig()
        self.max_context_tokens = max_context_tokens or _default_max_context_tokens(
            getattr(llm_provider, "name", None)
        )
        self.tool_registry = tool_registry or build_tool_registry()
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

        turn_id = new_id("turn")
        acquire_request = LoopAcquireRequest(
            loop_name="reactive",
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

        debug: dict[str, Any] = {
            "turn_id": turn_id,
            "llm_retry_count": 0,
            "llm_round_count": 0,
            "tool_iteration_count": 0,
            "tool_call_count": 0,
            "provider_tool_call_count": 0,
            "note": "reactive cognition tick enabled; projections are Phase 02 stubs",
        }
        try:
            self._check_canceled(session_id, "before_user_event")
            model_tools = self.tool_registry.to_llm_tool_definitions()
            counterpart_ref = self.counterpart_router.upsert_from_source_metadata(
                source_metadata,
                emitter=self.emitter,
            )
            profile_snapshot = self._session_profile_snapshot(session_id, counterpart_ref)
            self._run_pre_user_context_maintenance(
                session_id=session_id,
                counterpart=counterpart_ref,
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
            )
            debug["user_message_id"] = user_record.id
            debug["user_message_ordinal"] = user_record.ordinal
            session_context = self.session_context.load(
                session_id,
                before_ordinal=user_record.ordinal,
            )
            chat_history = list(session_context.chat_messages)
            debug["chat_history_message_count"] = len(chat_history)

            stimulus = Stimulus(
                kind=StimulusKind.USER_MESSAGE,
                source=counterpart_ref,
                payload=user_message,
                thread_id=StimulusRouter.route_kind(
                    StimulusKind.USER_MESSAGE,
                    payload={"source_metadata": dict(source_metadata or {})},
                    session_id=session_id,
                ),
                received_at=Instant(utc_now_iso()),
                source_refs=[
                    Reference("session", session_id),
                    Reference("session_message", user_record.id),
                ],
            )

            controller = CognitiveController(
                event_log=self.event_log,
                projections=default_projection_registry(self.event_log),
                llm=self.llm_provider,
                tools=self.tool_registry,
                emitter=self.emitter,
                effector=Effector(
                    llm_provider=self.llm_provider,
                    tool_registry=self.tool_registry,
                    completion_runner=lambda decision, view, rendered: (
                        self._run_reactive_completion(
                            session_id=session_id,
                            decision=decision,
                            view=view,
                            rendered=rendered,
                            model_tools=model_tools or None,
                            model_tool_choice="auto" if model_tools else None,
                            debug=debug,
                        )
                    ),
                ),
            )
            loop_result = controller.reactive_tick(
                stimulus=stimulus,
                thread_id=stimulus.thread_id,
                chat_history=chat_history,
                counterpart_profile=profile_snapshot.content if profile_snapshot else None,
            )
            debug.update(loop_result.debug)
            llm_response = loop_result.outcome.raw_llm_response
            if not isinstance(llm_response, LLMResponse):
                llm_response = LLMResponse(
                    content=loop_result.response_text,
                    model="unknown",
                    provider="unknown",
                )
            assistant_record = self._write_assistant_message(session_id, llm_response)
            debug["assistant_message_id"] = assistant_record.id
            debug["assistant_message_ordinal"] = assistant_record.ordinal
            self._emit_turn_sources_recorded(
                session_id=session_id,
                user_record=user_record,
                assistant_record=assistant_record,
                loop_result=loop_result,
            )
            return AgentTurnResult(
                response=loop_result.response_text,
                session_id=session_id,
                debug=debug,
            )
        except AgentCanceledError as exc:
            self._emit_turn_failed(session_id, turn_id, "canceled", exc.stage, exc, debug)
            self.clear_cancel(session_id)
            raise
        except LLMCallError as exc:
            debug["llm_retry_count"] = debug.get("llm_retry_count", 0) + exc.retry_count
            self._emit_turn_failed(session_id, turn_id, "failed", "llm", exc, debug)
            self.clear_cancel(session_id)
            raise
        except Exception as exc:
            self._emit_turn_failed(
                session_id,
                turn_id,
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

    def _run_reactive_completion(
        self,
        *,
        session_id: str,
        decision: Any,
        view: CognitionView,
        rendered: RenderResult,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        debug: dict[str, Any],
    ) -> Outcome:
        del decision
        self._check_canceled(session_id, "before_prompt")
        prompt_frame = self._prompt_frame_from_rendered(
            view=view,
            rendered=rendered,
        )
        messages = self._rebuild_runtime_llm_messages(
            session_id=session_id,
            prompt_frame=prompt_frame,
        )
        prompt_token_estimate = estimate_chat_tokens(messages, tools=model_tools)
        debug["context_window_foreground_count"] = len(view.window.foreground)
        debug["prompt_token_estimate"] = prompt_token_estimate
        debug["renderer"] = "text_chat"
        debug["render_used_tokens"] = rendered.used_tokens
        debug["render_dropped_sections"] = list(rendered.dropped_sections)
        reactive_metadata = _mapping(view.metadata.get("_reactive_completion_context"))
        memory_propose_context = {
            "tick_id": str(reactive_metadata.get("tick_id") or ""),
            "session_id": session_id,
            "user_message_id": str(debug.get("user_message_id") or ""),
            "decision_event_id": str(reactive_metadata.get("decision_event_id") or ""),
            "emitter": self.emitter,
            "apply_cognitive_event": self._apply_tool_cognitive_event,
            "subject": view.subject,
            "situation": view.window.situation_at,
            "counterpart": view.window.counterpart,
        }
        memory_recall_context = {
            "session_id": session_id,
            "counterpart": view.window.counterpart,
            "belief_projection": BeliefProjection(
                self.store,
                event_log=self.event_log,
                auto_rebuild=True,
            ),
        }
        loop_result = self._run_agent_loop(
            session_id=session_id,
            messages=messages,
            prompt_frame=prompt_frame,
            model_tools=model_tools,
            model_tool_choice=model_tool_choice,
            initial_prompt_token_estimate=prompt_token_estimate,
            memory_propose_context=memory_propose_context,
            memory_recall_context=memory_recall_context,
            debug=debug,
        )
        llm_response = loop_result.response
        return Outcome(
            text=llm_response.content,
            tool_calls=list(loop_result.provider_tool_calls),
            tool_results=list(loop_result.tool_results),
            raw_llm_response=llm_response,
            debug={
                "provider": llm_response.provider,
                "final_provider": llm_response.provider,
                "llm_round_count": loop_result.llm_round_count,
                "llm_retry_count": loop_result.llm_retry_count,
                "tool_iteration_count": loop_result.tool_iteration_count,
                "tool_call_count": loop_result.tool_call_count,
                "provider_tool_call_count": loop_result.provider_tool_call_count,
                "provider_tool_call_ids": [
                    call.id for call in loop_result.provider_tool_calls if call.id
                ],
                "provider_tool_message_ids": [
                    message.id for message in loop_result.provider_tool_messages
                ],
                "provider_tool_trace_ids": list(loop_result.provider_tool_trace_ids),
                "llm_call_ids": list(loop_result.llm_call_ids),
                "llm_trace_ids": list(loop_result.llm_trace_ids),
                "tool_cognitive_event_ids": list(loop_result.tool_cognitive_event_ids),
                "final_finish_reason": llm_response.finish_reason,
            },
        )

    def _run_pre_user_context_maintenance(
        self,
        *,
        session_id: str,
        counterpart: Reference | None,
        pending_user_message: str,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        debug: dict[str, Any],
    ) -> None:
        pending_message: ChatMessage = {"role": "user", "content": pending_user_message}
        profile_messages = self._session_profile_context_messages(session_id, counterpart)
        pending_only_estimate = self._estimate_context_budget(
            [self._default_system_message(), *profile_messages, pending_message],
            tools=model_tools,
        )
        if pending_only_estimate.used_context_tokens > self.max_context_tokens:
            self._raise_pending_user_too_large(pending_only_estimate)

        planning_messages = [self._default_system_message(), *profile_messages, pending_message]
        projected_messages = self._source_prompt_messages(
            session_id=session_id,
            counterpart=counterpart,
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
                counterpart=counterpart,
                extra_source_messages=[pending_message],
            )
            estimate = self._estimate_context_budget(projected_messages, tools=model_tools)

        if self._needs_handover(estimate) and self.session_context.load(
            session_id
        ).source_messages:
            compression_messages = self._source_prompt_messages(
                session_id=session_id,
                counterpart=counterpart,
            )
            result = compress_session_context(
                session_id=session_id,
                assembler=self.session_context,
                llm_provider=self.llm_provider,
                llm_messages=compression_messages,
                tools=model_tools,
                tool_choice="none" if model_tools else None,
            )
            debug["pre_user_compressed_message_id"] = result.message.id
            debug["pre_user_compression_point_ordinal"] = result.compression_point_ordinal
            projected_messages = self._source_prompt_messages(
                session_id=session_id,
                counterpart=counterpart,
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
        session_id: str,
        prompt_frame: PromptFrame,
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
                    *prompt_frame.prefix_context_messages,
                    *prompt_frame.suffix_context_messages,
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
            )
            debug["tool_loop_compressed_message_id"] = result.message.id
            debug["tool_loop_compression_point_ordinal"] = result.compression_point_ordinal
            llm_messages = self._rebuild_runtime_llm_messages(
                session_id=session_id,
                prompt_frame=prompt_frame,
            )
        return llm_messages

    def _source_prompt_messages(
        self,
        *,
        session_id: str,
        counterpart: Reference | None = None,
        extra_source_messages: Sequence[ChatMessage] | None = None,
    ) -> list[ChatMessage]:
        return [
            self._default_system_message(),
            *self._session_profile_context_messages(session_id, counterpart),
            *self.session_context.load(session_id).chat_messages,
            *(extra_source_messages or ()),
        ]

    def _rebuild_runtime_llm_messages(
        self,
        *,
        session_id: str,
        prompt_frame: PromptFrame,
    ) -> list[ChatMessage]:
        return [
            _copy_chat_message(prompt_frame.system_message),
            *[
                _copy_chat_message(message)
                for message in prompt_frame.prefix_context_messages
            ],
            *self.session_context.load(session_id).chat_messages,
            *[
                _copy_chat_message(message)
                for message in prompt_frame.suffix_context_messages
            ],
        ]

    def _prompt_frame_from_rendered(
        self,
        *,
        view: CognitionView,
        rendered: RenderResult,
    ) -> PromptFrame:
        rendered_messages = cast(list[ChatMessage], rendered.payload)
        if not rendered_messages:
            return PromptFrame(self._default_system_message(), [], [])
        system_message = _copy_chat_message(rendered_messages[0])
        remaining = list(rendered_messages[1:])
        prefix_messages: list[ChatMessage] = []
        if remaining and self._is_counterpart_profile_message(
            remaining[0],
            view.counterpart_profile,
        ):
            prefix_messages.append(_copy_chat_message(remaining.pop(0)))
        source_count = len(view.chat_history)
        suffix_messages = list(remaining[source_count:])
        if suffix_messages and self._is_current_query_message(
            suffix_messages[-1],
            view.current_query,
        ):
            suffix_messages = suffix_messages[:-1]
        return PromptFrame(
            system_message=system_message,
            prefix_context_messages=prefix_messages,
            suffix_context_messages=[_copy_chat_message(message) for message in suffix_messages],
        )

    def _is_counterpart_profile_message(
        self,
        message: ChatMessage,
        counterpart_profile: str | None,
    ) -> bool:
        content = str(message.get("content") or "").strip()
        return (
            bool(counterpart_profile)
            and message.get("role") == "user"
            and content.startswith("<system-reminder>")
            and COUNTERPART_PROFILE_LABEL in content
        )

    def _is_current_query_message(
        self,
        message: ChatMessage,
        current_query: str | None,
    ) -> bool:
        return (
            current_query is not None
            and message.get("role") == "user"
            and message.get("content") == current_query
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
        return _copy_chat_message(_DEFAULT_RUNTIME_SYSTEM_MESSAGE)

    def _session_profile_snapshot(
        self,
        session_id: str,
        counterpart: Reference | None,
    ) -> SessionProfileSnapshot | None:
        if counterpart is None:
            return None
        snapshot = self.store.get_session_profile_snapshot(session_id, counterpart.id)
        if snapshot is not None:
            return snapshot
        projection = BeliefProjection(
            self.store,
            event_log=self.event_log,
            auto_rebuild=True,
        )
        digest = active_counterpart_digest(projection, counterpart)
        if digest is None:
            return None
        content = str(digest.content).strip()
        if not content:
            return None
        return self.store.create_session_profile_snapshot(
            session_id=session_id,
            counterpart_id=counterpart.id,
            source_belief_id=str(digest.id),
            content=content,
        )

    def _session_profile_context_messages(
        self,
        session_id: str,
        counterpart: Reference | None,
    ) -> list[ChatMessage]:
        if counterpart is None:
            return []
        snapshot = self.store.get_session_profile_snapshot(session_id, counterpart.id)
        if snapshot is None:
            return []
        return [
            {
                "role": "user",
                "content": wrap_system_reminder(render_counterpart_profile(snapshot.content)),
            }
        ]

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
        session_id: str,
        messages: list[ChatMessage],
        prompt_frame: PromptFrame,
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
                calls=provider_tool_calls,
                llm_response=response,
            )
            provider_tool_messages.append(provider_tool_call_message)
            llm_messages.append(source_message_to_chat(provider_tool_call_message))
            provider_results = self._execute_tool_calls(
                session_id=session_id,
                calls=provider_tool_calls,
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
                results=provider_results,
            )
            provider_tool_messages.extend(provider_result_messages)
            tool_call_count += len(provider_results)
            tool_iteration_count += 1
            llm_messages.extend(
                source_message_to_chat(message) for message in provider_result_messages
            )
            llm_messages = self._run_tool_result_context_maintenance(
                session_id=session_id,
                prompt_frame=prompt_frame,
                model_tools=model_tools,
                model_tool_choice=model_tool_choice,
                debug=debug,
            )

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
        started_trace = self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.started",
            content="LLM call started.",
            metadata={
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
        if request_log is not None:
            self._append_llm_trace(
                event="llm.request",
                metadata={
                    "llm_call_id": llm_call_id,
                    "session_id": session_id,
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
                },
            )
            raise
        self._check_canceled(session_id, "after_llm")
        if request_log is not None:
            self._append_llm_trace(
                event="llm.response",
                metadata={
                    "llm_call_id": llm_call_id,
                    "session_id": session_id,
                    "response": _llm_response_log(completion.response),
                },
            )
        completed_trace = self.store.append_runtime_trace(
            session_id=session_id,
            event_type="llm.completed",
            content="LLM call completed.",
            metadata={
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
        )

    def _write_assistant_tool_call_message(
        self,
        *,
        session_id: str,
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
                "tool_call_ids": [self._required_tool_call_id(call) for call in calls],
            },
        )

    def _write_tool_result_messages(
        self,
        *,
        session_id: str,
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
        session_id: str,
        user_record: SessionMessage,
        assistant_record: SessionMessage,
        loop_result: Any,
    ) -> None:
        debug = loop_result.debug if isinstance(loop_result.debug, dict) else {}
        provider_tool_message_ids = _string_list(debug.get("provider_tool_message_ids"))
        provider_tool_trace_ids = _string_list(debug.get("provider_tool_trace_ids"))
        llm_call_ids = _string_list(debug.get("llm_call_ids"))
        llm_trace_ids = _string_list(debug.get("llm_trace_ids"))
        reactive_event_ids = _string_list(debug.get("event_ids"))
        tool_cognitive_event_ids = _string_list(debug.get("tool_cognitive_event_ids"))
        tick_id = str(debug.get("tick_id") or "")
        source_refs = [
            Reference("session_message", user_record.id),
            Reference("session_message", assistant_record.id),
            *[Reference("session_message", item) for item in provider_tool_message_ids],
            *[Reference("runtime_trace", item) for item in provider_tool_trace_ids],
            *[Reference("runtime_trace", item) for item in llm_trace_ids],
        ]
        self.emitter.emit(
            CognitiveEventKind.TURN_SOURCES_RECORDED,
            inputs=[Reference("cognitive_tick", tick_id)] if tick_id else [],
            outputs=source_refs,
            rationale="Recorded persisted session and trace ids for the reactive turn.",
            causal_parents=[EventId(reactive_event_ids[-1])] if reactive_event_ids else [],
            payload={
                "tick_id": tick_id,
                "session_id": session_id,
                "user_message_id": user_record.id,
                "assistant_message_id": assistant_record.id,
                "assistant_message_ordinal": assistant_record.ordinal,
                "provider_tool_message_ids": provider_tool_message_ids,
                "provider_tool_trace_ids": provider_tool_trace_ids,
                "llm_call_ids": llm_call_ids,
                "llm_trace_ids": llm_trace_ids,
                "reactive_event_ids": reactive_event_ids,
                "tool_cognitive_event_ids": tool_cognitive_event_ids,
            },
        )

    def _emit_turn_failed(
        self,
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
        session_id: str,
        calls: list[ToolCall],
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
            write_trace=lambda event_type, content, metadata: self.store.append_runtime_trace(
                session_id=session_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            ),
            check_canceled=lambda stage: self._check_canceled(session_id, stage),
            recover_errors=recover_errors,
        )

    def _apply_tool_cognitive_event(self, event: Any) -> None:
        projection = BeliefProjection(self.store)
        if event.kind in projection.handles:
            projection.apply(event)

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


def _llm_request_summary(
    *,
    messages: list[ChatMessage],
    tools: Sequence[LLMToolDefinitionInput] | None,
    tool_choice: LLMToolChoice | None,
) -> dict[str, Any]:
    return {
        "message_count": len(messages),
        "roles": [str(message.get("role", "")) for message in messages],
        "tool_count": len(tools) if tools is not None else 0,
        "tool_names": [_llm_tool_name(tool) for tool in tools or []],
        "tool_choice": _json_safe(tool_choice),
    }


def _llm_response_log(response: LLMResponse) -> dict[str, Any]:
    response_payload = response.metadata.get("response_payload")
    if isinstance(response_payload, dict):
        return _json_safe(response_payload)

    return {
        "content": response.content,
        "finish_reason": response.finish_reason,
        "model": response.model,
        "provider": response.provider,
        "tool_calls": [tool_call.to_dict() for tool_call in response.tool_calls],
    }


def _llm_metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    return _json_safe(
        {
            key: value
            for key, value in metadata.items()
            if key
            not in {
                "request_payload",
                "response_payload",
                "raw_tool_calls",
                "normalized_tool_calls",
                "tool_calls",
            }
        }
    )


def _llm_tool_name(tool: LLMToolDefinitionInput) -> str:
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name") is not None:
            return str(function["name"])
        return str(tool.get("name", ""))
    return str(getattr(tool, "name", ""))


def _default_max_context_tokens(provider_name: object) -> int:
    provider_key = str(provider_name or "openai-compatible")
    return DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS.get(
        provider_key,
        DEFAULT_PROVIDER_MAX_CONTEXT_TOKENS["openai-compatible"],
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return []
    return [str(item) for item in value if item is not None]


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


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
