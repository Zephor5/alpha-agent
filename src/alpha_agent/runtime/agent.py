"""Explicit personal agent runtime."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
from alpha_agent.memory.models import Event, ExtractedMemoryCandidate, RetrievedContext
from alpha_agent.memory.persistence import persist_candidates
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.semantic import SemanticMemoryManager
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.runtime.events import create_event, create_runtime_event, deterministic_json
from alpha_agent.runtime.prompt_builder import PromptBuilder
from alpha_agent.runtime.tools import ExecutedToolResult, ToolExecutionError, ToolExecutor
from alpha_agent.tools.base import ToolCall
from alpha_agent.tools.registry import ToolRegistry
from alpha_agent.utils.ids import new_id


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
        working_memory: WorkingMemoryManager,
        retriever: MemoryRetriever,
        retrieval_limit: int = 8,
        prompt_builder: PromptBuilder | None = None,
        extractor: MemoryExtractor | None = None,
        tool_registry: ToolRegistry | None = None,
        max_llm_retries: int = 2,
        llm_retry_sleep_seconds: float = 0.0,
        max_tool_iterations: int = 8,
        max_llm_rounds: int | None = None,
    ):
        self.store = store
        self.llm_provider = llm_provider
        self.working_memory = working_memory
        self.retriever = retriever
        self.retrieval_limit = retrieval_limit
        self.prompt_builder = prompt_builder or PromptBuilder()
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
        *,
        tool_calls: Sequence[ToolCall | Mapping[str, Any]] | None = None,
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
            turn_started_event = self._emit_turn_started(
                session_id=session_id,
                turn_id=turn_id,
                user_message=user_message,
            )
            debug["turn_started_event_id"] = turn_started_event.id

            self._check_canceled(session_id, "before_user_event")
            user_event = self._write_user_event(session_id, user_message)
            self._update_working_memory(
                session_id=session_id,
                content=f"User: {user_message}",
                source_event_id=user_event.id,
                priority=0.6,
            )

            requested_tool_calls = self.tool_executor.normalize_calls(tool_calls)
            requested_tool_results = self._execute_tool_calls(
                session_id=session_id,
                calls=requested_tool_calls,
                source="caller",
            )
            self._add_tool_results_to_working_memory(session_id, requested_tool_results)
            debug["tool_call_count"] = len(requested_tool_results)

            self._check_canceled(session_id, "before_retrieval")
            context = self._retrieve_memory(user_message, session_id)
            retrieved_ids = self._retrieved_ids(context)
            debug["retrieved_memory_ids"] = retrieved_ids

            messages = self._build_prompt(user_message, context)
            prompt_token_estimate = self.prompt_builder.rough_token_estimate(messages)
            debug["prompt_token_estimate"] = prompt_token_estimate
            model_tools = self.tool_registry.to_llm_tool_definitions()
            model_tool_choice: LLMToolChoice | None = "auto" if model_tools else None

            llm_completion = self._call_model(
                session_id=session_id,
                messages=messages,
                prompt_token_estimate=prompt_token_estimate,
                round_name="initial",
                tools=model_tools or None,
                tool_choice=model_tool_choice,
            )
            llm_response = llm_completion.response
            llm_round_count = 1
            debug["llm_retry_count"] = llm_completion.retry_count
            debug["llm_round_count"] = llm_round_count
            debug["initial_provider"] = llm_response.provider

            provider_tool_results: list[ExecutedToolResult] = []
            tool_iteration_count = 0
            conversation_messages = list(messages)
            while self._response_requests_tools(llm_response):
                provider_tool_calls = self.tool_executor.normalize_calls(llm_response.tool_calls)
                self._validate_provider_tool_calls(provider_tool_calls, llm_response)
                if tool_iteration_count >= self.max_tool_iterations:
                    llm_response, llm_round_count = self._finalize_tool_loop(
                        session_id=session_id,
                        messages=conversation_messages,
                        reason="max_tool_iterations",
                        llm_round_count=llm_round_count,
                        tool_iteration_count=tool_iteration_count,
                        debug=debug,
                    )
                    break
                if llm_round_count >= self.max_llm_rounds:
                    llm_response, llm_round_count = self._finalize_tool_loop(
                        session_id=session_id,
                        messages=conversation_messages,
                        reason="max_llm_rounds",
                        llm_round_count=llm_round_count,
                        tool_iteration_count=tool_iteration_count,
                        debug=debug,
                    )
                    break

                debug["provider_tool_call_count"] += len(provider_tool_calls)
                provider_results = self._execute_tool_calls(
                    session_id=session_id,
                    calls=provider_tool_calls,
                    source="provider",
                    recover_errors=True,
                )
                provider_tool_results.extend(provider_results)
                self._add_tool_results_to_working_memory(session_id, provider_results)
                debug["tool_call_count"] += len(provider_results)
                tool_iteration_count += 1
                debug["tool_iteration_count"] = tool_iteration_count

                conversation_messages.extend(
                    self._tool_result_messages(
                        calls=provider_tool_calls,
                        results=provider_results,
                    )
                )
                tool_result_completion = self._call_model(
                    session_id=session_id,
                    messages=conversation_messages,
                    prompt_token_estimate=self.prompt_builder.rough_token_estimate(
                        conversation_messages
                    ),
                    round_name=f"tool_result_{tool_iteration_count}",
                    tools=model_tools or None,
                    tool_choice=model_tool_choice,
                )
                llm_round_count += 1
                debug["llm_retry_count"] += tool_result_completion.retry_count
                debug["llm_round_count"] = llm_round_count
                llm_response = tool_result_completion.response

            debug["provider"] = llm_response.provider
            debug["final_provider"] = llm_response.provider
            debug["llm_round_count"] = llm_round_count
            debug["final_finish_reason"] = llm_response.finish_reason

            assistant_event = self._write_assistant_event(session_id, llm_response)
            self._update_working_memory(
                session_id=session_id,
                content=f"Assistant: {llm_response.content}",
                source_event_id=assistant_event.id,
                priority=0.45,
            )

            extraction_source_event_ids = [
                user_event.id,
                assistant_event.id,
                *[item.event.id for item in requested_tool_results],
                *[item.event.id for item in provider_tool_results],
            ]
            candidates = self._extract_memory(
                session_id=session_id,
                user_message=user_message,
                assistant_response=llm_response.content,
                source_event_ids=extraction_source_event_ids,
            )
            self._persist_extracted_memories(candidates)
            debug["extracted_memory_count"] = len(candidates)
            debug["consolidation"] = self._decide_consolidation_trigger(candidates)

            delivery_event = self._emit_delivery_event(
                session_id=session_id,
                turn_id=turn_id,
                assistant_event=assistant_event,
                debug=debug,
            )
            debug["delivery_event_id"] = delivery_event.id

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

    def _emit_turn_started(self, session_id: str, turn_id: str, user_message: str) -> Event:
        return self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="turn.started",
                content="Agent turn started.",
                metadata={
                    "turn_id": turn_id,
                    "user_message_length": len(user_message),
                },
            )
        )

    def _write_user_event(self, session_id: str, user_message: str) -> Event:
        return self.store.insert_event(create_event(session_id, "user", user_message))

    def _update_working_memory(
        self,
        *,
        session_id: str,
        content: str,
        source_event_id: str | None,
        priority: float,
    ) -> None:
        self.working_memory.add_active_context(
            session_id=session_id,
            content=content,
            source_event_id=source_event_id,
            priority=priority,
        )

    def _retrieve_memory(self, user_message: str, session_id: str) -> RetrievedContext:
        context = self.retriever.retrieve_context(
            user_message,
            session_id,
            limit=self.retrieval_limit,
        )
        retrieved_ids = self._retrieved_ids(context)
        self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="memory.retrieved",
                content=deterministic_json(retrieved_ids),
                metadata={
                    "retrieval_limit": self.retrieval_limit,
                    "retrieved_memory_ids": retrieved_ids,
                    "counts": {key: len(value) for key, value in retrieved_ids.items()},
                },
            )
        )
        return context

    def _build_prompt(self, user_message: str, context: RetrievedContext) -> list[ChatMessage]:
        return self.prompt_builder.build(user_message, context)

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
        started_event = self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="llm.started",
                content="LLM call started.",
                metadata={
                    "provider": self.llm_provider.name,
                    "round": round_name,
                    "prompt_token_estimate": prompt_token_estimate,
                    "max_retries": self.llm_completion.max_retries,
                    "tool_count": len(tools) if tools is not None else 0,
                    "tool_choice": tool_choice,
                },
            )
        )
        completion = self.llm_completion.complete(
            list(messages),
            tools=tools,
            tool_choice=tool_choice,
        )
        self._check_canceled(session_id, "after_llm")
        self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="llm.completed",
                content="LLM call completed.",
                metadata={
                    "provider": completion.response.provider,
                    "model": completion.response.model,
                    "round": round_name,
                    "retry_count": completion.retry_count,
                    "started_event_id": started_event.id,
                    "finish_reason": completion.response.finish_reason,
                    "tool_call_count": len(completion.response.tool_calls),
                    "response_metadata": completion.response.metadata,
                },
            )
        )
        return completion

    def _write_assistant_event(self, session_id: str, llm_response: LLMResponse) -> Event:
        return self.store.insert_event(
            create_event(
                session_id,
                "assistant",
                llm_response.content,
                metadata={
                    "provider": llm_response.provider,
                    "model": llm_response.model,
                    "provider_metadata": llm_response.metadata,
                },
            )
        )

    def _extract_memory(
        self,
        *,
        session_id: str,
        user_message: str,
        assistant_response: str,
        source_event_ids: list[str],
    ) -> list[ExtractedMemoryCandidate]:
        candidates = self.extractor.extract(
            user_message=user_message,
            assistant_response=assistant_response,
            source_event_ids=source_event_ids,
        )
        type_counts: dict[str, int] = {}
        for candidate in candidates:
            type_counts[candidate.type] = type_counts.get(candidate.type, 0) + 1
        self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="memory.extracted",
                content=deterministic_json(type_counts),
                metadata={
                    "extracted_memory_count": len(candidates),
                    "candidate_type_counts": type_counts,
                    "source_event_ids": source_event_ids,
                },
            )
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

    def _emit_delivery_event(
        self,
        *,
        session_id: str,
        turn_id: str,
        assistant_event: Event,
        debug: dict[str, Any],
    ) -> Event:
        return self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type="turn.completed",
                content="Agent turn completed.",
                metadata={
                    "turn_id": turn_id,
                    "assistant_response_event_id": assistant_event.id,
                    "retry_count": debug.get("llm_retry_count", 0),
                    "llm_round_count": debug.get("llm_round_count", 0),
                    "tool_iteration_count": debug.get("tool_iteration_count", 0),
                    "final_finish_reason": debug.get("final_finish_reason"),
                    "tool_call_count": debug.get("tool_call_count", 0),
                    "extracted_memory_count": debug.get("extracted_memory_count", 0),
                    "consolidation": debug.get("consolidation", {}),
                },
            )
        )

    def _emit_turn_failed(
        self,
        *,
        session_id: str,
        turn_id: str,
        status: str,
        stage: str,
        error: Exception,
        debug: dict[str, Any],
    ) -> Event:
        return self.store.insert_event(
            create_runtime_event(
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
        )

    def _finalize_tool_loop(
        self,
        *,
        session_id: str,
        messages: list[ChatMessage],
        reason: str,
        llm_round_count: int,
        tool_iteration_count: int,
        debug: dict[str, Any],
    ) -> tuple[LLMResponse, int]:
        self._emit_tool_loop_event(
            session_id=session_id,
            event_type="tool_loop.finalizing",
            content="Tool loop limit reached; requesting no-tools final answer.",
            metadata={
                "reason": reason,
                "llm_round_count": llm_round_count,
                "tool_iteration_count": tool_iteration_count,
                "max_llm_rounds": self.max_llm_rounds,
                "max_tool_iterations": self.max_tool_iterations,
            },
        )
        final_messages = [
            *messages,
            self._tool_loop_finalization_message(reason=reason),
        ]
        final_completion = self._call_model(
            session_id=session_id,
            messages=final_messages,
            prompt_token_estimate=self.prompt_builder.rough_token_estimate(final_messages),
            round_name="finalize",
            tools=None,
            tool_choice=None,
        )
        llm_round_count += 1
        response = final_completion.response
        debug["llm_retry_count"] += final_completion.retry_count
        debug["llm_round_count"] = llm_round_count
        debug["final_provider"] = response.provider
        debug["final_finish_reason"] = response.finish_reason
        if self._response_requests_tools(response):
            self._emit_tool_loop_event(
                session_id=session_id,
                event_type="tool_loop.finalization_failed",
                content="No-tools finalization returned tool calls.",
                metadata={
                    "reason": reason,
                    "finish_reason": response.finish_reason,
                    "llm_round_count": llm_round_count,
                    "tool_iteration_count": tool_iteration_count,
                    "tool_call_count": len(response.tool_calls),
                },
            )
            raise ToolLoopLimitExceeded(
                "tool_loop_limit_exceeded: finalization returned tool calls"
            )
        return response, llm_round_count

    def _tool_loop_finalization_message(self, *, reason: str) -> ChatMessage:
        return {
            "role": "user",
            "content": (
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
    ) -> Event:
        return self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type=event_type,
                content=content,
                metadata=metadata,
            )
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

    def _tool_result_messages(
        self,
        *,
        calls: list[ToolCall],
        results: list[ExecutedToolResult],
    ) -> list[ChatMessage]:
        messages = [self._assistant_tool_call_message(calls)]
        for item in results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": self._required_tool_call_id(item.call),
                    "content": item.event.content,
                }
            )
        return messages

    def _assistant_tool_call_message(self, calls: list[ToolCall]) -> ChatMessage:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [self._wire_tool_call(call) for call in calls],
        }

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
        source: str,
        recover_errors: bool = False,
    ) -> list[ExecutedToolResult]:
        return self.tool_executor.execute(
            calls=calls,
            write_event=lambda event_type, content, metadata: self._write_tool_event(
                session_id=session_id,
                event_type=event_type,
                content=content,
                source=source,
                metadata=metadata,
            ),
            check_canceled=lambda stage: self._check_canceled(session_id, stage),
            recover_errors=recover_errors,
        )

    def _write_tool_event(
        self,
        *,
        session_id: str,
        event_type: str,
        content: str,
        source: str,
        metadata: dict[str, Any],
    ) -> Event:
        event_metadata = {"source": source}
        event_metadata.update(metadata)
        return self.store.insert_event(
            create_runtime_event(
                session_id=session_id,
                event_type=event_type,
                content=content,
                role="tool",
                metadata=event_metadata,
            )
        )

    def _add_tool_results_to_working_memory(
        self,
        session_id: str,
        results: list[ExecutedToolResult],
    ) -> None:
        for item in results:
            self._update_working_memory(
                session_id=session_id,
                content=f"Tool {item.result.name}: {item.result.content}",
                source_event_id=item.event.id,
                priority=0.5,
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
            "working": [item.id for item in context.working_memory],
            "episodic": [item.id for item in context.episodic_memories],
            "semantic": [item.id for item in context.semantic_memories],
            "procedural": [item.id for item in context.procedural_memories],
        }
