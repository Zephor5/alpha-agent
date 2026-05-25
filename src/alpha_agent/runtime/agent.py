"""Explicit personal agent runtime backed by the reactive cognition tick."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx

from alpha_agent.cognition.controller import CognitiveController, default_projection_registry
from alpha_agent.cognition.coordinator import (
    LockBusy,
    LoopAcquireRequest,
    LoopCoordinator,
)
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.base import EventLog
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import Instant, LoopPriority, Stimulus, StimulusKind
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.projections.counterpart import CounterpartProjection
from alpha_agent.cognition.stages.effector import Effector, build_reactive_messages
from alpha_agent.cognition.stages.types import Outcome
from alpha_agent.cognition.threads import StimulusRouter
from alpha_agent.llm.base import (
    ChatCompletionToolCall,
    ChatMessage,
    LLMProvider,
    LLMResponse,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.counterpart_router import CounterpartRouter
from alpha_agent.runtime.events import deterministic_json
from alpha_agent.runtime.prompt_builder import PromptBuilder, wrap_system_reminder
from alpha_agent.runtime.session_context import SessionContextManager
from alpha_agent.runtime.tools import ExecutedToolResult, ToolExecutionError, ToolExecutor
from alpha_agent.state.models import ConversationMessage, RuntimeTrace
from alpha_agent.state.store import StateStore
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
    response: LLMResponse
    retry_count: int


@dataclass(frozen=True)
class AgentLoopResult:
    response: LLMResponse
    provider_tool_messages: list[ConversationMessage]
    provider_tool_calls: list[ToolCall]
    tool_results: list[Any]
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
        prompt_builder: PromptBuilder | None = None,
        tool_registry: ToolRegistry | None = None,
        max_llm_retries: int = 2,
        llm_retry_sleep_seconds: float = 0.0,
        max_tool_iterations: int = 8,
        max_llm_rounds: int | None = None,
        llm_debug_logging: bool = False,
        llm_trace_log_path: str | Path | None = None,
        context_recent_tail_messages: int = 8,
        event_log: EventLog | None = None,
        coordinator: LoopCoordinator | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.session_context = SessionContextManager(
            store,
            recent_tail_messages=context_recent_tail_messages,
        )
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
            counterpart_ref = self.counterpart_router.upsert_from_source_metadata(
                source_metadata,
                emitter=self.emitter,
            )
            user_record = self.store.append_conversation_message(
                session_id=session_id,
                role="user",
                raw_content=user_message,
                source_metadata=dict(source_metadata or {}),
            )
            debug["user_message_id"] = user_record.id
            debug["user_message_ordinal"] = user_record.ordinal

            model_tools = self.tool_registry.to_llm_tool_definitions()

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
                    completion_runner=lambda decision, window: self._run_reactive_completion(
                        session_id=session_id,
                        decision=decision,
                        window=window,
                        model_tools=model_tools or None,
                        model_tool_choice="auto" if model_tools else None,
                        debug=debug,
                    ),
                ),
            )
            loop_result = controller.reactive_tick(
                stimulus=stimulus,
                thread_id=stimulus.thread_id,
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
            acquire_context.__exit__(None, None, None)
            if session_id not in self._canceled_sessions:
                self.clear_cancel(session_id)

    def _run_reactive_completion(
        self,
        *,
        session_id: str,
        decision: Any,
        window: Any,
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        debug: dict[str, Any],
    ) -> Outcome:
        self._check_canceled(session_id, "before_prompt")
        messages = build_reactive_messages(decision, window)
        prompt_token_estimate = self.prompt_builder.estimate_prompt_tokens(
            messages,
            tools=model_tools,
        )
        debug["context_window_foreground_count"] = len(window.foreground)
        debug["prompt_token_estimate"] = prompt_token_estimate
        loop_result = self._run_agent_loop(
            session_id=session_id,
            messages=messages,
            model_tools=model_tools,
            model_tool_choice=model_tool_choice,
            initial_prompt_token_estimate=prompt_token_estimate,
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
                "final_finish_reason": llm_response.finish_reason,
            },
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
        model_tools: Sequence[LLMToolDefinitionInput] | None,
        model_tool_choice: LLMToolChoice | None,
        initial_prompt_token_estimate: int,
        debug: dict[str, Any],
    ) -> AgentLoopResult:
        conversation_messages = list(messages)
        provider_tool_messages: list[ConversationMessage] = []
        provider_tool_calls_seen: list[ToolCall] = []
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
                messages=conversation_messages,
                prompt_token_estimate=(
                    initial_prompt_token_estimate
                    if llm_round_count == 0
                    else self.prompt_builder.estimate_prompt_tokens(
                        conversation_messages,
                        tools=model_tools,
                    )
                ),
                round_name=round_name,
                tools=model_tools,
                tool_choice="none" if finalizing and model_tools is not None else model_tool_choice,
            )
            llm_round_count += 1
            llm_retry_count += completion.retry_count
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
                conversation_messages.append(self._tool_loop_finalization_message(finalizing_reason))
                continue
            if llm_round_count >= self.max_llm_rounds:
                finalizing_reason = "max_llm_rounds"
                conversation_messages.append(self._tool_loop_finalization_message(finalizing_reason))
                continue

            provider_tool_call_count += len(provider_tool_calls)
            provider_tool_calls_seen.extend(provider_tool_calls)
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
            tool_results_seen.extend(item.result for item in provider_results)
            provider_result_messages = self._write_tool_result_messages(
                session_id=session_id,
                results=provider_results,
            )
            provider_tool_messages.extend(provider_result_messages)
            tool_call_count += len(provider_results)
            tool_iteration_count += 1
            conversation_messages.extend(
                self.prompt_builder.conversation_message_to_chat(message)
                for message in provider_result_messages
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
                "request": request_log,
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
        self.store.append_runtime_trace(
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
                "response_metadata": _llm_metadata_for_event(
                    completion.response.metadata,
                    include_raw_payloads=self.llm_debug_logging,
                ),
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
        llm_response: LLMResponse,
    ) -> ConversationMessage:
        return self.store.append_conversation_message(
            session_id=session_id,
            role="assistant",
            raw_content=llm_response.content,
            tool_calls=[dict(self._wire_tool_call(call)) for call in calls],
            provider_metadata={
                "provider": llm_response.provider,
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "metadata": _llm_metadata_for_event(
                    llm_response.metadata,
                    include_raw_payloads=self.llm_debug_logging,
                ),
            },
            metadata={"tool_call_ids": [self._required_tool_call_id(call) for call in calls]},
        )

    def _write_tool_result_messages(
        self,
        *,
        session_id: str,
        results: list[ExecutedToolResult],
    ) -> list[ConversationMessage]:
        return [
            self.store.append_conversation_message(
                session_id=session_id,
                role="tool",
                raw_content=item.trace.content,
                tool_call_id=self._required_tool_call_id(item.call),
                tool_result_id=item.trace.id,
                provider_metadata={"tool_name": item.result.name},
                metadata={
                    "trace_id": item.trace.id,
                    "result_metadata": dict(item.result.metadata),
                },
            )
            for item in results
        ]

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
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        return self.tool_executor.execute(
            calls=calls,
            write_trace=lambda event_type, content, metadata: self.store.append_runtime_trace(
                session_id=session_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
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
