from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from threading import Event
from typing import TypedDict, cast

from alpha_agent.cognition.authority import CognitionSourceKind
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops.feedback_attribution import (
    FeedbackAttributionJob,
    RealtimeFeedbackAttributionService,
)
from alpha_agent.cognition.loops.workers.memory_consolidation import (
    MemoryConflictReviewWorker,
)
from alpha_agent.cognition.models import (
    SUBJECT_SELF,
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    CognitiveEventKind,
    CounterpartId,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    Role,
    SituationId,
    ValidityWindow,
    counterpart_ref,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.processing_ledger import (
    BackgroundProgressStatus,
    BackgroundSourceRef,
    BackgroundStage,
)
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMResponseFormat,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.runtime.counterpart_router import DEFAULT_COUNTERPART_ID
from alpha_agent.state.models import SessionMessage
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import JSONValue, ToolExecutionContext
from alpha_agent.tools.memory_recall import (
    MEMORY_RECALL_CONTEXT_KEY,
    MEMORY_RECALL_TOOL_NAME,
    MemoryRecallTool,
)


def test_recalled_wrong_preference_is_corrected_through_feedback_loop(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    state = CognitionStateStore(store)
    event_log = SQLiteEventLog(store)
    wrong_belief_id = "belief:example-language"
    corrected_quote = "I prefer TypeScript examples"
    state.write_atomic_belief(
        _preference_belief(
            wrong_belief_id,
            "User prefers Python examples.",
            object_="example language preference",
        ),
        source_kind=CognitionSourceKind.DIRECT_USER_STATEMENT,
    )
    submitted_jobs: list[FeedbackAttributionJob] = []
    runtime_provider = _RecallThenAnswerProvider()
    agent = AlphaAgent(
        store=store,
        llm_provider=runtime_provider,
        event_log=event_log,
        feedback_attribution_submitter=submitted_jobs.append,
    )

    first = agent.respond("What examples do I prefer?", session_id="s1")

    assert first.response == "You prefer Python examples."
    assert submitted_jobs == []
    recall_message = _only_tool_message(store, "s1")
    assert recall_message.provider_metadata == {"tool_name": MEMORY_RECALL_TOOL_NAME}
    assert json.loads(recall_message.raw_content)["results"] == [
        {
            "content": "User prefers Python examples.",
            "held_since": "2026-01-01T00:00:00+00:00",
            "id": wrong_belief_id,
            "lifecycle": "active",
            "memory_kind": "preference",
            "scope": "counterpart",
        }
    ]

    second = agent.respond(
        "Actually, I prefer TypeScript examples.",
        session_id="s1",
    )

    assert second.response == "Got it, TypeScript examples."
    assert len(submitted_jobs) == 1
    job = submitted_jobs[0]
    assert job.user_message_text == "Actually, I prefer TypeScript examples."
    assert job.recall_tool_message_ids == (recall_message.id,)
    assert [handle.belief_id for handle in job.recalled_beliefs] == [wrong_belief_id]
    assert state.ledger.list_source_progress(
        stage=BackgroundStage.FEEDBACK_ATTRIBUTION
    ) == []

    feedback_provider = _BlockingProvider(
        _feedback_json(
            {
                "belief_id": wrong_belief_id,
                "verdict": "corrected",
                "evidence_quote": corrected_quote,
            }
        )
    )
    feedback_service = RealtimeFeedbackAttributionService(
        store=store,
        llm_provider=feedback_provider,
        max_workers=1,
    )

    assert feedback_service.submit(job)
    assert feedback_provider.started.wait(timeout=2.0)
    # Feedback attribution rows are created only after a worker slot is acquired,
    # so absence before submit is the deterministic pending equivalent here.
    rows_while_blocked = state.ledger.list_source_progress(
        stage=BackgroundStage.FEEDBACK_ATTRIBUTION
    )
    assert [(row.source_ref.source_id, row.status) for row in rows_while_blocked] == [
        (recall_message.id, BackgroundProgressStatus.CLAIMED)
    ]
    feedback_provider.release()
    feedback_service.shutdown(wait=True, timeout=2.0)

    feedback_rows = state.ledger.list_source_progress(
        stage=BackgroundStage.FEEDBACK_ATTRIBUTION
    )
    assert [
        (row.source_ref.source_id, row.status, row.last_error, row.checkpoint_id)
        for row in feedback_rows
    ] == [
        (
            recall_message.id,
            BackgroundProgressStatus.PROCESSED,
            None,
            f"feedback_attribution:{job.turn_id}",
        )
    ]
    assert state.ledger.list_source_progress(
        stage=BackgroundStage.FEEDBACK_ATTRIBUTION,
        status=BackgroundProgressStatus.FAILED,
    ) == []

    feedback_events = list(event_log.iter(kinds=[CognitiveEventKind.RECEIVED_FEEDBACK]))
    assert len(feedback_events) == 1
    feedback_event = feedback_events[0]
    assert feedback_event.payload["feedback_kind"] == "belief_corrected"
    assert feedback_event.payload["verdict"] == "corrected"
    assert feedback_event.payload["matched_expected"] is False
    assert feedback_event.payload["evidence_quote"] == corrected_quote
    assert feedback_event.payload["belief_id"] == wrong_belief_id
    assert feedback_event.payload["user_message_id"] == job.user_message_id
    assert feedback_event.payload["recall_tool_message_ids"] == [recall_message.id]
    assert feedback_event.inputs == [
        Reference("belief", wrong_belief_id),
        Reference("session_message", job.user_message_id),
    ]
    assert [str(parent) for parent in feedback_event.causal_parents] == [
        job.turn_received_event_id
    ]

    attribution_audits = state.audit_records(kind="feedback_attribution_completed")
    assert len(attribution_audits) == 1
    attribution_audit = attribution_audits[0]
    assert attribution_audit.payload == {
        "belief_ids": [wrong_belief_id],
        "event_ids": [str(feedback_event.id)],
        "recall_tool_message_ids": [recall_message.id],
        "session_id": "s1",
        "turn_id": job.turn_id,
        "turn_received_event_id": job.turn_received_event_id,
        "user_message_id": job.user_message_id,
        "verdict_count": 1,
    }

    belief_feedback_audits = state.audit_records(kind="belief_feedback_recorded")
    assert len(belief_feedback_audits) == 1
    belief_feedback_audit = belief_feedback_audits[0]
    assert belief_feedback_audit.payload == {
        "at": str(feedback_event.timestamp),
        "belief_id": wrong_belief_id,
        "event_id": str(feedback_event.id),
        "kind": "corrected",
    }
    assert attribution_audit.payload["event_ids"] == [
        belief_feedback_audit.payload["event_id"]
    ]
    assert attribution_audit.payload["belief_ids"] == [
        belief_feedback_audit.payload["belief_id"]
    ]

    challenged = state.beliefs.get_by_id(wrong_belief_id)
    assert isinstance(challenged, AtomicBelief)
    assert [json.loads(entry) for entry in challenged.feedback_history] == [
        {
            "at": str(feedback_event.timestamp),
            "event_id": str(feedback_event.id),
            "kind": "corrected",
        }
    ]

    windows = state.ledger.list_source_windows(stage=BackgroundStage.CONFLICT_REVIEW)
    assert len(windows) == 1
    conflict_window = windows[0]
    assert conflict_window.status == BackgroundProgressStatus.PENDING
    assert conflict_window.source_refs == (
        BackgroundSourceRef(
            "conflict",
            f"belief_feedback:{wrong_belief_id}:{job.user_message_id}",
        ),
    )
    assert conflict_window.metadata == {
        "active_belief_ids": [wrong_belief_id],
        "belief_content": "User prefers Python examples.",
        "belief_id": wrong_belief_id,
        "evidence_quote": corrected_quote,
        "feedback_event_id": str(feedback_event.id),
        "session_id": "s1",
        "user_message_id": job.user_message_id,
        "verdict": "corrected",
    }

    conflict_provider = _RecordingProvider(
        _background_llm_json(
            operation="supersede",
            payload={
                "belief_update": {
                    "target_belief_id": wrong_belief_id,
                    "rationale": "The user corrected the recalled preference.",
                },
                "atomic_belief_draft": {
                    "memory_kind": MemoryKind.PREFERENCE.value,
                    "scope": BeliefScope.COUNTERPART.value,
                    "about": [_counterpart_record()],
                    "object": "example language preference",
                    "content": "User prefers TypeScript examples.",
                },
            },
        )
    )

    report = MemoryConflictReviewWorker(state, conflict_provider).run_once()

    assert report.emitted == 1
    superseded = state.beliefs.get_by_id(wrong_belief_id)
    assert isinstance(superseded, AtomicBelief)
    assert superseded.lifecycle == BeliefLifecycle.SUPERSEDED
    replacement = _only_corrected_active_belief(state, wrong_belief_id)
    assert str(replacement.content) == "User prefers TypeScript examples."
    assert replacement.supersedes == Reference("belief", wrong_belief_id)
    assert superseded.superseded_by == Reference("belief", str(replacement.id))

    conflict_source = BackgroundSourceRef(
        "conflict",
        f"belief_feedback:{wrong_belief_id}:{job.user_message_id}",
    )
    assert state.ledger.get_source_progress(
        conflict_source,
        stage=BackgroundStage.CONFLICT_REVIEW,
        target_unit="scope:global",
    ).status == BackgroundProgressStatus.PROCESSED
    assert state.ledger.get_source_window(conflict_window.window_id).status == (
        BackgroundProgressStatus.PROCESSED
    )

    fresh_recall = _memory_recall(
        state,
        tmp_path,
        query="TypeScript examples preference",
    )
    assert [item["content"] for item in fresh_recall] == [
        "User prefers TypeScript examples."
    ]
    assert [item["id"] for item in fresh_recall] == [str(replacement.id)]


class _ProviderCall(TypedDict):
    messages: list[ChatMessage]
    tools: Sequence[LLMToolDefinitionInput] | None
    tool_choice: LLMToolChoice | None
    response_format: LLMResponseFormat | None


class _RecallThenAnswerProvider:
    name = "feedback-loop-runtime"

    def __init__(self) -> None:
        self.calls: list[list[ChatMessage]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        del tools, tool_choice, response_format
        self.calls.append([cast(ChatMessage, dict(message)) for message in messages])
        if len(self.calls) == 1:
            return LLMResponse(
                content="",
                model="test-runtime-model",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_recall",
                        name=MEMORY_RECALL_TOOL_NAME,
                        arguments={
                            "query": "example language preference",
                            "scope": "counterpart",
                            "types": ["preference"],
                            "max_results": 1,
                        },
                        raw_arguments=json.dumps(
                            {
                                "max_results": 1,
                                "query": "example language preference",
                                "scope": "counterpart",
                                "types": ["preference"],
                            },
                            sort_keys=True,
                        ),
                    )
                ],
            )
        if len(self.calls) == 2:
            return LLMResponse(
                content="You prefer Python examples.",
                model="test-runtime-model",
                provider=self.name,
            )
        return LLMResponse(
            content="Got it, TypeScript examples.",
            model="test-runtime-model",
            provider=self.name,
        )


class _BlockingProvider:
    name = "blocking-feedback-attribution"

    def __init__(self, response: str) -> None:
        self.response = response
        self.started = Event()
        self._release = Event()
        self.calls: list[_ProviderCall] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        self.started.set()
        self._release.wait(timeout=2.0)
        return LLMResponse(
            content=self.response,
            model="test-feedback-model",
            provider=self.name,
        )

    def release(self) -> None:
        self._release.set()


class _RecordingProvider:
    name = "recording-conflict-review"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[_ProviderCall] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: LLMResponseFormat | None = None,
    ) -> LLMResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "tool_choice": tool_choice,
                "response_format": response_format,
            }
        )
        return LLMResponse(
            content=self.response,
            model="test-conflict-model",
            provider=self.name,
        )


def _store(tmp_path: Path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _preference_belief(
    belief_id: str,
    content: str,
    *,
    object_: str,
) -> AtomicBelief:
    counterpart = counterpart_ref(CounterpartId(str(DEFAULT_COUNTERPART_ID)))
    return AtomicBelief(
        id=BeliefId(belief_id),
        subject=subject_ref(SUBJECT_SELF),
        about=[counterpart],
        object=object_,
        content=NLStatement(content),
        memory_kind=MemoryKind.PREFERENCE,
        derivation_stage=DerivationStage.TOOL_WRITTEN,
        scope=BeliefScope.COUNTERPART,
        authority=Authority.USER_ASSERTED,
        sources=[Reference("session_message", "msg_seed")],
        validity=ValidityWindow(observed_at=Instant("2026-01-01T00:00:00+00:00")),
        formed_in=situation_ref(SituationId("situation:test")),
        holder_role=Role("agent"),
        lifecycle=BeliefLifecycle.ACTIVE,
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


def _feedback_json(*verdicts: dict[str, str]) -> str:
    return json.dumps({"payload": {"verdicts": list(verdicts)}}, sort_keys=True)


def _background_llm_json(
    *,
    operation: str,
    payload: dict[str, object],
) -> str:
    return json.dumps(
        {
            "authority": Authority.BACKGROUND_SYNTHESIZED.value,
            "operation": operation,
            "payload": payload,
            "rationale": "Fixture conflict review decision.",
            "requires_confirmation": False,
            "source_span_note": "from user correction feedback",
        },
        sort_keys=True,
    )


def _counterpart_record() -> dict[str, str]:
    counterpart = counterpart_ref(CounterpartId(str(DEFAULT_COUNTERPART_ID)))
    return {"kind": counterpart.kind, "id": counterpart.id}


def _only_tool_message(store: StateStore, session_id: str) -> SessionMessage:
    messages = [
        message
        for message in store.list_session_messages(session_id)
        if message.kind == "tool_message"
    ]
    assert len(messages) == 1
    return messages[0]


def _only_corrected_active_belief(
    state: CognitionStateStore,
    wrong_belief_id: str,
) -> AtomicBelief:
    active = [
        belief
        for belief in state.beliefs.list_active()
        if str(belief.id) != wrong_belief_id
    ]
    assert len(active) == 1
    return active[0]


def _memory_recall(
    state: CognitionStateStore,
    tmp_path: Path,
    *,
    query: str,
) -> list[Mapping[str, JSONValue]]:
    result = MemoryRecallTool().run(
        {
            "query": query,
            "scope": "counterpart",
            "types": ["preference"],
            "max_results": 4,
        },
        ToolExecutionContext(
            session_id="s1",
            tool_call_id="call_fresh_recall",
            output_dir=tmp_path,
            check_canceled=lambda _stage: None,
            extensions={
                MEMORY_RECALL_CONTEXT_KEY: {
                    "session_id": "s1",
                    "counterpart": counterpart_ref(
                        CounterpartId(str(DEFAULT_COUNTERPART_ID))
                    ),
                    "belief_projection": state.beliefs,
                }
            },
        ),
    )
    assert isinstance(result.output, dict)
    raw_results = result.output["results"]
    assert isinstance(raw_results, list)
    return [
        item
        for item in raw_results
        if isinstance(item, Mapping)
    ]
