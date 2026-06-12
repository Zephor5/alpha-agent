from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    AtomicBelief,
    Authority,
    BeliefId,
    BeliefLifecycle,
    BeliefScope,
    CognitiveEventKind,
    DerivationStage,
    Instant,
    MemoryKind,
    NLStatement,
    Reference,
    SummaryBelief,
    SummaryKind,
    ValidityWindow,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.llm.base import (
    ChatMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinition,
    LLMToolDefinitionInput,
)
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.runtime.tools import ToolExecutor
from alpha_agent.state.store import StateStore
from alpha_agent.tools.base import ToolCall
from alpha_agent.tools.default import build_tool_registry
from alpha_agent.tools.memory_propose import MEMORY_PROPOSE_TOOL_NAME
from tests.cognition.test_belief_projection_apply import counterpart_a, summary_belief


def test_memory_propose_tool_schema_exposes_new_memory_kinds() -> None:
    definition = build_tool_registry().to_llm_tool_definitions()[0]

    assert definition.name == MEMORY_PROPOSE_TOOL_NAME
    assert "long-term memories" in definition.description
    memory_schema = definition.parameters["properties"]["updates"]["items"]["properties"][
        "memory"
    ]
    assert memory_schema["properties"]["type"]["enum"] == [
        "fact",
        "preference",
        "constraint",
        "procedure",
        "value",
        "relationship",
    ]


def test_memory_propose_append_writes_atomic_belief_directly_in_runtime_turn(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _MemoryProposeProvider(
        updates=[
            _update(
                "append_distinct",
                memory={
                    "type": "preference",
                    "content": "User prefers future answers in Chinese.",
                    "evidence": "User said: 以后都用中文回答我.",
                    "scope": "counterpart",
                },
                reason="User explicitly stated a stable answer-language preference.",
            )
        ]
    )
    agent = AlphaAgent(store=store, llm_provider=provider)

    result = agent.respond("以后都用中文回答我", session_id="s1")

    assert result.response == "好的，以后我会用中文回答。"
    assert MEMORY_PROPOSE_TOOL_NAME in provider.tool_names_seen[0]
    all_messages = store.list_session_messages("s1")
    assert [message.kind for message in all_messages] == [
        "system_reminder",
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    messages = [message for message in all_messages if message.kind != "system_reminder"]
    assert [message.kind for message in messages] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    tool_output = json.loads(messages[2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["next_action"] == "none"
    assert tool_output["results"][0]["decision"] == "accepted"
    assert tool_output["results"][0]["new_belief_id"]

    events = list(SQLiteEventLog(store).iter())
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    assert len(proposed) == 1
    assert messages[2].metadata["result_metadata"]["cognitive_event_ids"] == [
        str(proposed[0].id)
    ]
    assert proposed[0].payload["gate"] == {
        "decision": "accepted",
        "reason": "accepted_append_distinct",
    }

    belief = BeliefProjection(store).list_active()[0]
    assert str(belief.id) == tool_output["results"][0]["new_belief_id"]
    assert belief.content == "User prefers future answers in Chinese."
    assert belief.memory_kind == MemoryKind.PREFERENCE
    assert belief.derivation_stage == DerivationStage.TOOL_WRITTEN
    assert belief.authority == Authority.USER_ASSERTED
    assert belief.scope == BeliefScope.COUNTERPART
    assert belief.sources == [Reference("session_message", messages[0].id)]
    assert belief.derivation is not None
    assert "memory_propose" in str(belief.derivation)
    audit_records = CognitionStateStore(store).audit_records(kind="memory_propose_write")
    assert len(audit_records) == 1
    assert audit_records[0].entity_refs == (Reference("belief", str(belief.id)),)
    assert audit_records[0].payload["source"] == MEMORY_PROPOSE_TOOL_NAME
    assert audit_records[0].payload["operation"] == "append_distinct"

    source_recorded = [
        event for event in events if event.kind == CognitiveEventKind.TURN_SOURCES_RECORDED
    ][0]
    assert source_recorded.payload["tool_cognitive_event_ids"] == [str(proposed[0].id)]
    assert result.debug["tool_cognitive_event_ids"] == [str(proposed[0].id)]


def test_memory_propose_constraint_is_first_class_memory_kind(tmp_path) -> None:
    store = _store(tmp_path)

    _run_updates(
        store,
        session_id="s1",
        message="Never write local machine-specific absolute paths into the repository.",
        updates=[
            _update(
                "append_distinct",
                memory={
                    "type": "constraint",
                    "content": "Do not write local machine-specific absolute paths into the repo.",
                    "evidence": "User stated this project rule explicitly.",
                    "scope": "global",
                },
                reason="The user stated a stable repository constraint.",
            )
        ],
    )

    belief = BeliefProjection(store).list_active()[0]
    assert belief.memory_kind == MemoryKind.CONSTRAINT
    assert belief.scope == BeliefScope.GLOBAL
    assert not belief.object.startswith("constraint:")


def test_memory_propose_duplicate_append_reaffirms_without_new_belief(tmp_path) -> None:
    store = _store(tmp_path)
    original = _append_memory(
        store,
        session_id="s1",
        content="User prefers future answers in Chinese.",
        evidence="User said: 以后都用中文回答我.",
    )
    original_sources = list(original.sources)

    _run_updates(
        store,
        session_id="s1",
        message="提醒一下，还是用中文",
        updates=[
            _update(
                "append_distinct",
                memory={
                    "type": "preference",
                    "content": "User prefers future answers in Chinese.",
                    "evidence": "User repeated the same preference.",
                    "scope": "counterpart",
                },
                reason="The user repeated an already active preference.",
            )
        ],
    )

    active = BeliefProjection(store).list_active()
    assert len(active) == 1
    assert active[0].id == original.id
    assert len(active[0].sources) == len(original_sources) + 1
    assert active[0].validity.observed_at is not None
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["operation"] == "reinforce"
    assert tool_output["results"][0]["reason"] == "accepted_duplicate_reinforced"


def test_memory_propose_replace_supersedes_directly_without_belief_event(tmp_path) -> None:
    store = _store(tmp_path)
    original = _append_memory(
        store,
        session_id="s1",
        content="User prefers Python examples.",
        evidence="User said they prefer Python examples.",
    )

    _run_updates(
        store,
        session_id="s1",
        message="Actually use Rust examples.",
        updates=[
            _update(
                "replace",
                target_belief_ids=[str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": (
                        "User said: actually replace my Python example preference with Rust."
                    ),
                    "scope": "counterpart",
                },
                reason="User explicitly changed and replaced the previous example preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    active = projection.list_active()
    assert len(active) == 1
    assert active[0].id != original.id
    assert active[0].content == "User prefers Rust examples."
    superseded_original = projection.get_by_id(original.id)
    assert superseded_original is not None
    assert superseded_original.lifecycle == BeliefLifecycle.SUPERSEDED
    assert superseded_original.superseded_by is not None
    assert superseded_original.superseded_by.id == str(active[0].id)

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["reason"] == "accepted_replace"


def test_memory_propose_correct_pending_writes_pending_atomic_belief(tmp_path) -> None:
    store = _store(tmp_path)
    target = _append_memory(
        store,
        session_id="s1",
        content="User prefers Python examples.",
        evidence="User said they prefer Python examples.",
    )

    _run_updates(
        store,
        session_id="s1",
        message="Correct that memory.",
        updates=[
            _update(
                "correct",
                target_belief_ids=[str(target.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said the previous Python-example memory is wrong.",
                    "scope": "counterpart",
                },
                reason="The user is correcting an existing remembered preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    assert [item.id for item in projection.list_active()] == [target.id]
    events = list(SQLiteEventLog(store).iter())
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    assert proposed[-1].payload["gate"] == {
        "decision": "pending_confirmation",
        "reason": "correct_requires_confirmation",
    }
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "pending_confirmation"
    assert tool_output["next_action"] == "ask_user_confirmation"
    pending_id = tool_output["results"][0]["new_belief_id"]
    pending = projection.get_by_id(pending_id)
    assert isinstance(pending, AtomicBelief)
    assert pending.lifecycle == BeliefLifecycle.PENDING_CONFIRMATION
    assert pending.content == "User prefers Rust examples."
    assert pending.memory_kind == MemoryKind.PREFERENCE
    assert pending.scope == BeliefScope.COUNTERPART
    assert [item.id for item in projection.list_active()] == [target.id]


def test_memory_propose_rejects_summary_belief_targets_cleanly(tmp_path) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    profile = summary_belief(
        "belief:profile",
        "User likes concise Python examples.",
        about=[counterpart_a()],
    )
    projection.upsert_summary(profile)

    _run_updates(
        store,
        session_id="s1",
        message="Replace that profile.",
        updates=[
            _update(
                "replace",
                target_belief_ids=[str(profile.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User asked to replace the profile.",
                    "scope": "counterpart",
                },
                reason="The model incorrectly targeted a summary belief.",
            )
        ],
    )

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "rejected"
    assert tool_output["results"][0]["reason"] == "target_not_atomic"
    assert BeliefProjection(store).get_by_id(profile.id) == profile


@pytest.mark.parametrize(
    ("target_domain", "valid_until", "expected_status", "expected_reason"),
    [
        ("memory_propose", None, "pending_confirmation", "domain_guidance_requires_confirmation"),
        ("memory_propose", "2020-01-01T00:00:00+00:00", "accepted", "accepted_append_distinct"),
        ("memory_recall", None, "accepted", "accepted_append_distinct"),
    ],
)
def test_memory_propose_applies_only_active_matching_domain_summary_guidance(
    tmp_path,
    target_domain: str,
    valid_until: str | None,
    expected_status: str,
    expected_reason: str,
) -> None:
    store = _store(tmp_path)
    projection = BeliefProjection(store)
    projection.upsert_summary(
        _domain_summary(
            "belief:memory-propose-guidance",
            target_domain=target_domain,
            valid_until=valid_until,
        )
    )

    _run_updates(
        store,
        session_id="s1",
        message="Remember that I prefer Rust examples.",
        updates=[
            _update(
                "append_distinct",
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User explicitly asked to remember Rust examples.",
                    "scope": "counterpart",
                },
                reason="User explicitly stated a stable example-language preference.",
            )
        ],
    )

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == expected_status
    assert tool_output["results"][0]["reason"] == expected_reason
    written = projection.get_by_id(tool_output["results"][0]["new_belief_id"])
    assert isinstance(written, AtomicBelief)
    expected_lifecycle = (
        BeliefLifecycle.PENDING_CONFIRMATION
        if expected_status == "pending_confirmation"
        else BeliefLifecycle.ACTIVE
    )
    assert written.lifecycle == expected_lifecycle


def test_memory_propose_noops_without_reactive_write_context(tmp_path) -> None:
    store = _store(tmp_path)
    registry = build_tool_registry()
    executor = ToolExecutor(registry)

    executed = executor.execute(
        calls=[
            ToolCall(
                id="call_memory",
                name=MEMORY_PROPOSE_TOOL_NAME,
                arguments={
                    "updates": [
                        _update(
                            "append_distinct",
                            memory={
                                "type": "preference",
                                "content": "User prefers Chinese.",
                                "evidence": "User said so.",
                                "scope": "counterpart",
                            },
                            reason="User explicitly stated a stable answer-language preference.",
                        )
                    ]
                },
            )
        ],
        session_id="s1",
        write_trace=lambda event_type, content, metadata: store.append_runtime_trace(
            session_id="s1",
            event_type=event_type,
            content=content,
            metadata=metadata,
        ),
        check_canceled=lambda _stage: None,
        recover_errors=False,
    )

    assert executed[0].result.output == {
        "status": "rejected",
        "next_action": "explain_rejection",
        "results": [],
    }
    assert list(SQLiteEventLog(store).iter(kinds=[CognitiveEventKind.MEMORY_PROPOSED])) == []
    assert BeliefProjection(store).list_active() == []


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _append_memory(
    store: StateStore,
    *,
    session_id: str,
    content: str,
    evidence: str,
) -> Any:
    _run_updates(
        store,
        session_id=session_id,
        message=evidence,
        updates=[
            _update(
                "append_distinct",
                memory={
                    "type": "preference",
                    "content": content,
                    "evidence": evidence,
                    "scope": "counterpart",
                },
                reason="User explicitly stated a stable preference.",
            )
        ],
    )
    return BeliefProjection(store).list_active()[-1]


def _domain_summary(
    belief_id: str,
    *,
    target_domain: str,
    valid_until: str | None,
) -> SummaryBelief:
    return SummaryBelief(
        id=BeliefId(belief_id),
        subject=Reference("subject", "subject:self"),
        about=[],
        object="memory proposal domain guidance",
        content=NLStatement("Memory proposal guidance requires confirmation."),
        summary_kind=SummaryKind.DOMAIN_SUMMARY,
        derivation_stage=DerivationStage.BACKGROUND_SUMMARIZED,
        scope=BeliefScope.GLOBAL,
        authority=Authority.BACKGROUND_SYNTHESIZED,
        structure={
            "target_domain": target_domain,
            "memory_propose": {"requires_confirmation": True},
        },
        validity=ValidityWindow(
            observed_at=Instant("2026-01-01T00:00:00+00:00"),
            valid_until=Instant(valid_until) if valid_until is not None else None,
        ),
        lifecycle=BeliefLifecycle.ACTIVE,
        held_since=Instant("2026-01-01T00:00:00+00:00"),
    )


def _run_updates(
    store: StateStore,
    *,
    session_id: str,
    message: str,
    updates: list[dict[str, Any]],
) -> None:
    provider = _MemoryProposeProvider(updates=updates)
    AlphaAgent(store=store, llm_provider=provider).respond(message, session_id=session_id)


def _update(
    operation: str,
    *,
    memory: dict[str, str],
    reason: str,
    target_belief_ids: list[str] | None = None,
    reviewed_candidate_ids: list[str] | None = None,
    target_hint: str = "",
) -> dict[str, Any]:
    return {
        "operation": operation,
        "target_belief_ids": list(target_belief_ids or []),
        "reviewed_candidate_ids": list(reviewed_candidate_ids or []),
        "target_hint": target_hint,
        "memory": memory,
        "reason": reason,
    }


class _MemoryProposeProvider:
    name = "memory-propose-provider"

    def __init__(self, *, updates: list[dict[str, Any]]):
        self.updates = updates
        self.calls = 0
        self.tool_names_seen: list[list[str]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
        response_format: object | None = None,
    ) -> LLMResponse:
        del messages, tool_choice, response_format
        self.calls += 1
        self.tool_names_seen.append([_tool_name(tool) for tool in tools or []])
        if self.calls == 1:
            return LLMResponse(
                content="",
                model="test",
                provider=self.name,
                finish_reason="tool_calls",
                tool_calls=[
                    LLMToolCall(
                        id="call_memory",
                        name=MEMORY_PROPOSE_TOOL_NAME,
                        arguments={"updates": self.updates},
                        raw_arguments=json.dumps(
                            {"updates": self.updates},
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                ],
            )
        return LLMResponse(
            content="好的，以后我会用中文回答。",
            model="test",
            provider=self.name,
        )


def _tool_name(tool: LLMToolDefinitionInput) -> str:
    if isinstance(tool, LLMToolDefinition):
        return tool.name
    if isinstance(tool, dict):
        function = tool.get("function")
        if isinstance(function, dict):
            return str(function.get("name") or "")
        return str(tool.get("name") or "")
    function = tool.get("function")
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool.get("name") or "")
