from __future__ import annotations

import json
from collections.abc import Sequence

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEventKind, CognitiveType, Reference
from alpha_agent.cognition.projections.belief import BeliefProjection
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


def test_memory_propose_tool_description_explains_usage_contract() -> None:
    definition = build_tool_registry().to_llm_tool_definitions()[0]

    assert definition.name == MEMORY_PROPOSE_TOOL_NAME
    assert "explicit long-term" in definition.description
    assert "preferences" in definition.description
    assert "stable constraints" in definition.description
    assert "reusable procedures" in definition.description
    assert "direct corrections" in definition.description
    assert "ordinary facts" in definition.description
    assert "structured memory proposal result" in definition.description
    assert "pending_confirmation" in definition.description


def test_memory_propose_promotes_explicit_preference_in_runtime_turn(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "preference",
                "content": "User prefers future answers in Chinese.",
                "evidence": "User said: 以后都用中文回答我.",
                "scope": "counterpart",
            }
        ]
    )
    agent = AlphaAgent(store=store, llm_provider=provider)

    result = agent.respond("以后都用中文回答我", session_id="s1")

    assert result.response == "好的，以后我会用中文回答。"
    assert MEMORY_PROPOSE_TOOL_NAME in provider.tool_names_seen[0]
    messages = store.list_session_messages("s1")
    assert [message.kind for message in messages] == [
        "user_message",
        "assistant_message",
        "tool_message",
        "assistant_message",
    ]
    turn_id = result.debug["turn_id"]
    assert messages[1].metadata == {
        "turn_id": turn_id,
        "tool_call_ids": ["call_memory"],
    }
    assert messages[2].provider_metadata["tool_name"] == MEMORY_PROPOSE_TOOL_NAME
    tool_output = json.loads(messages[2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["user_action"] == "none"
    assert tool_output["message_hint"] == ""
    assert len(tool_output["proposal_results"]) == 1
    assert tool_output["proposal_results"][0]["decision"] == "accepted"
    assert tool_output["proposal_results"][0]["reason"] == "accepted_foreground_preference"
    assert messages[2].metadata["result_metadata"]["cognitive_event_ids"]
    assert messages[2].metadata["tool_output_kind"] == "json"

    events = list(SQLiteEventLog(store).iter())
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    assert len(proposed) == 1
    assert len(formed) == 1
    assert tool_output["proposal_results"][0]["proposal_id"] == proposed[0].payload["proposal_id"]
    assert proposed[0].causal_parents == [result.debug["turn_received_event_id"]]
    assert proposed[0].payload["gate"] == {
        "decision": "accepted",
        "reason": "accepted_foreground_preference",
    }
    assert proposed[0].payload["source_refs"] == [
        {"kind": "session", "id": "s1"},
        {"kind": "session_message", "id": messages[0].id},
    ]
    assert {"kind": "tool_call", "id": "call_memory"} in proposed[0].payload["audit_refs"]
    assert formed[0].causal_parents == [proposed[0].id]

    belief = BeliefProjection(store).list_active()[0]
    assert belief.content == "User prefers future answers in Chinese."
    assert belief.cognitive_type == CognitiveType.PREFERENCE
    assert belief.sources == [Reference("session_message", messages[0].id)]
    assert belief.derivation is not None
    assert str(proposed[0].id) in belief.derivation

    source_recorded = [
        event for event in events if event.kind == CognitiveEventKind.TURN_SOURCES_RECORDED
    ][0]
    assert source_recorded.payload["tool_cognitive_event_ids"] == [
        str(proposed[0].id),
        str(formed[0].id),
    ]
    assert result.debug["tool_cognitive_event_ids"] == [
        str(proposed[0].id),
        str(formed[0].id),
    ]


def test_memory_propose_records_each_proposal_and_leaves_correction_pending(tmp_path) -> None:
    store = _store(tmp_path)
    provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "constraint",
                "content": "Do not write local machine-specific absolute paths into the repo.",
                "evidence": "User stated this project rule explicitly.",
                "scope": "global",
            },
            {
                "kind": "correction",
                "content": "User corrected an older belief.",
                "evidence": "User said the previous belief is wrong.",
                "scope": "counterpart",
            },
        ]
    )
    agent = AlphaAgent(store=store, llm_provider=provider)

    agent.respond("记住规则，并纠正之前的记忆", session_id="s1")

    events = list(SQLiteEventLog(store).iter())
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    pending = [
        event
        for event in events
        if event.kind == CognitiveEventKind.BELIEF_FORM_PENDING_CONFIRMATION
    ]
    assert [event.payload["proposal"]["kind"] for event in proposed] == [
        "constraint",
        "correction",
    ]
    assert [event.payload["gate"]["decision"] for event in proposed] == [
        "accepted",
        "pending_confirmation",
    ]
    assert len(formed) == 1
    assert formed[0].payload["belief"]["content"] == (
        "Do not write local machine-specific absolute paths into the repo."
    )
    assert len(pending) == 1
    assert pending[0].payload == {
        "turn_id": proposed[1].payload["turn_id"],
        "session_id": "s1",
        "proposal_id": proposed[1].payload["proposal_id"],
        "reason": "correction_requires_review",
        "required_user_action": "confirm_memory_change",
        "candidate_change": {
            "kind": "correct",
            "content": "User corrected an older belief.",
        },
        "conflict_belief_ids": [],
    }
    assert pending[0].causal_parents == [proposed[1].id]

    tool_message = store.list_session_messages("s1")[2]
    tool_output = json.loads(tool_message.raw_content)
    assert tool_output["status"] == "mixed"
    assert tool_output["user_action"] == "ask_confirmation"
    assert "confirm" in tool_output["message_hint"]
    assert [item["decision"] for item in tool_output["proposal_results"]] == [
        "accepted",
        "pending_confirmation",
    ]
    assert tool_output["proposal_results"][1]["proposal_id"] == proposed[1].payload["proposal_id"]


def test_memory_propose_duplicate_active_belief_is_accepted_without_new_belief(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    first_provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "preference",
                "content": "User prefers future answers in Chinese.",
                "evidence": "User said: 以后都用中文回答我.",
                "scope": "counterpart",
            }
        ]
    )
    AlphaAgent(store=store, llm_provider=first_provider).respond(
        "以后都用中文回答我",
        session_id="s1",
    )

    duplicate_provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "preference",
                "content": "User prefers future answers in Chinese.",
                "evidence": "User repeated the same preference.",
                "scope": "counterpart",
            }
        ]
    )
    AlphaAgent(store=store, llm_provider=duplicate_provider).respond(
        "提醒一下，还是用中文",
        session_id="s1",
    )

    events = list(SQLiteEventLog(store).iter())
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    assert len(formed) == 1
    assert len(BeliefProjection(store).list_active()) == 1
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["proposal_results"][0]["reason"] == "accepted_duplicate_active_belief"


def test_memory_propose_single_structured_conflict_supersedes_existing_belief(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    first_provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "preference",
                "content": "User prefers Python examples.",
                "evidence": "User said they prefer Python examples.",
                "scope": "counterpart",
            }
        ]
    )
    AlphaAgent(store=store, llm_provider=first_provider).respond(
        "Remember I prefer Python examples.",
        session_id="s1",
    )
    original_belief = BeliefProjection(store).list_active()[0]

    conflict_provider = _MemoryProposeProvider(
        proposals=[
            {
                "kind": "preference",
                "content": "User prefers Rust examples.",
                "evidence": "User said they now prefer Rust examples.",
                "scope": "counterpart",
            }
        ]
    )
    AlphaAgent(store=store, llm_provider=conflict_provider).respond(
        "Actually use Rust examples.",
        session_id="s1",
    )

    projection = BeliefProjection(store)
    active = projection.list_active()
    assert len(active) == 1
    assert active[0].id != original_belief.id
    assert active[0].content == "User prefers Rust examples."
    assert active[0].supersedes is not None
    assert active[0].supersedes.id == str(original_belief.id)

    superseded_original = projection.get_by_id(original_belief.id)
    assert superseded_original is not None
    assert str(superseded_original.status) == "superseded"
    assert superseded_original.superseded_by is not None
    assert superseded_original.superseded_by.id == str(active[0].id)

    events = list(SQLiteEventLog(store).iter())
    superseded = [event for event in events if event.kind == CognitiveEventKind.BELIEF_SUPERSEDED]
    assert len(superseded) == 1
    assert superseded[0].payload["old_belief_id"] == str(original_belief.id)
    assert superseded[0].payload["new_belief_id"] == str(active[0].id)
    assert superseded[0].payload["reason"] == "accepted_single_structured_replacement"
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["user_action"] == "none"
    assert tool_output["proposal_results"][0]["reason"] == (
        "accepted_single_structured_replacement"
    )


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
                    "proposals": [
                        {
                            "kind": "preference",
                            "content": "User prefers Chinese.",
                            "evidence": "User said so.",
                            "scope": "counterpart",
                        }
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

    # The tool cannot infer user asked-to-remember intent without runtime context.
    # This direct tool invocation is still explicit enough to explain rejection.
    assert executed[0].result.output == {
        "status": "rejected",
        "user_action": "explain_rejection",
        "message_hint": "Memory proposal rejected: missing_runtime_turn_context.",
        "proposal_results": [],
    }
    assert list(SQLiteEventLog(store).iter(kinds=[CognitiveEventKind.MEMORY_PROPOSED])) == []
    assert list(SQLiteEventLog(store).iter(kinds=[CognitiveEventKind.BELIEF_FORMED])) == []


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


class _MemoryProposeProvider:
    name = "memory-propose-provider"

    def __init__(self, *, proposals: list[dict[str, str]]):
        self.proposals = proposals
        self.calls = 0
        self.tool_names_seen: list[list[str]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        tools: Sequence[LLMToolDefinitionInput] | None = None,
        tool_choice: LLMToolChoice | None = None,
    ) -> LLMResponse:
        del messages, tool_choice
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
                        arguments={"proposals": self.proposals},
                        raw_arguments=json.dumps(
                            {"proposals": self.proposals},
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
