from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

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


def test_memory_propose_tool_description_explains_update_contract() -> None:
    definition = build_tool_registry().to_llm_tool_definitions()[0]

    assert definition.name == MEMORY_PROPOSE_TOOL_NAME
    assert "long-term memories" in definition.description
    assert "append_distinct" in definition.description
    assert "reinforce" in definition.description
    assert "replace" in definition.description
    assert "merge" in definition.description
    assert "correct" in definition.description
    assert "retract" in definition.description
    assert "transient facts" in definition.description
    assert "next_action" in definition.description


def test_memory_propose_append_promotes_explicit_preference_in_runtime_turn(tmp_path) -> None:
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
    assert tool_output["next_action"] == "none"
    assert len(tool_output["results"]) == 1
    assert tool_output["results"][0]["update_index"] == 0
    assert tool_output["results"][0]["operation"] == "append_distinct"
    assert tool_output["results"][0]["decision"] == "accepted"
    assert tool_output["results"][0]["reason"] == "accepted_append_distinct"
    assert messages[2].metadata["result_metadata"]["cognitive_event_ids"]
    assert messages[2].metadata["tool_output_kind"] == "json"

    events = list(SQLiteEventLog(store).iter())
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    assert len(proposed) == 1
    assert len(formed) == 1
    assert tool_output["results"][0]["proposal_id"] == proposed[0].payload["proposal_id"]
    assert proposed[0].causal_parents == [result.debug["turn_received_event_id"]]
    assert proposed[0].payload["operation"] == "append_distinct"
    assert proposed[0].payload["target_belief_ids"] == []
    assert proposed[0].payload["reason"] == (
        "User explicitly stated a stable answer-language preference."
    )
    assert proposed[0].payload["evidence"] == "User said: 以后都用中文回答我."
    assert proposed[0].payload["gate"] == {
        "decision": "accepted",
        "reason": "accepted_append_distinct",
    }
    assert proposed[0].payload["source_refs"] == [
        {"kind": "session", "id": "s1"},
        {"kind": "session_message", "id": messages[0].id},
    ]
    assert {"kind": "tool_call", "id": "call_memory"} in proposed[0].payload["audit_refs"]
    assert formed[0].causal_parents == [proposed[0].id]
    assert formed[0].payload["operation"] == "append_distinct"
    assert formed[0].payload["target_belief_ids"] == []
    assert formed[0].payload["new_belief_id"] == tool_output["results"][0]["new_belief_id"]

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


def test_memory_propose_records_each_update_and_leaves_correct_pending(tmp_path) -> None:
    store = _store(tmp_path)
    target = _append_memory(
        store,
        session_id="s1",
        content="User prefers Python examples.",
        evidence="User said they prefer Python examples.",
    )
    provider = _MemoryProposeProvider(
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
            ),
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
            ),
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
    assert [event.payload["operation"] for event in proposed[-2:]] == ["append_distinct", "correct"]
    assert [event.payload["gate"]["decision"] for event in proposed[-2:]] == [
        "accepted",
        "pending_confirmation",
    ]
    assert len(formed) == 2
    assert formed[-1].payload["belief"]["content"] == (
        "Do not write local machine-specific absolute paths into the repo."
    )
    assert len(pending) == 1
    assert pending[0].payload["operation"] == "correct"
    assert pending[0].payload["target_belief_ids"] == [str(target.id)]
    assert pending[0].payload["candidate_change"] == {
        "operation": "correct",
        "memory": {
            "type": "preference",
            "content": "User prefers Rust examples.",
            "evidence": "User said the previous Python-example memory is wrong.",
            "scope": "counterpart",
        },
    }
    assert pending[0].causal_parents == [proposed[-1].id]
    assert [item.content for item in BeliefProjection(store).list_active()] == [
        "User prefers Python examples.",
        "Do not write local machine-specific absolute paths into the repo.",
    ]

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "mixed"
    assert tool_output["next_action"] == "ask_user_confirmation"
    assert [(item["operation"], item["decision"]) for item in tool_output["results"]] == [
        ("append_distinct", "accepted"),
        ("correct", "pending_confirmation"),
    ]


def test_memory_propose_duplicate_append_reinforces_without_new_belief(tmp_path) -> None:
    store = _store(tmp_path)
    original = _append_memory(
        store,
        session_id="s1",
        content="User prefers future answers in Chinese.",
        evidence="User said: 以后都用中文回答我.",
    )
    before_confidence = original.confidence

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

    events = list(SQLiteEventLog(store).iter())
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    strengthened = [
        event for event in events if event.kind == CognitiveEventKind.BELIEF_STRENGTHENED
    ]
    assert len(formed) == 1
    assert len(strengthened) == 1
    assert strengthened[0].payload["operation"] == "reinforce"
    assert strengthened[0].payload["target_belief_ids"] == [str(original.id)]
    active = BeliefProjection(store).list_active()
    assert len(active) == 1
    assert active[0].confidence > before_confidence
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["operation"] == "reinforce"
    assert tool_output["results"][0]["reason"] == "accepted_duplicate_reinforced"


def test_memory_propose_append_related_candidate_needs_target_selection(tmp_path) -> None:
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
                "append_distinct",
                target_hint="code example language preference",
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said they now prefer Rust examples.",
                    "scope": "counterpart",
                },
                reason="The user expressed a related but non-identical preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    active = projection.list_active()
    assert len(active) == 1
    assert active[0].id == original.id
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "needs_target_selection"
    assert tool_output["next_action"] == "review_candidates"
    assert tool_output["results"][0]["decision"] == "needs_target_selection"
    assert tool_output["results"][0]["resolution_options"] == [
        "append_distinct",
        "reinforce",
        "replace",
        "merge",
        "correct",
        "retract",
    ]
    assert tool_output["results"][0]["candidates"] == [
        {
            "id": str(original.id),
            "content": "User prefers Python examples.",
            "type": "preference",
            "scope": "counterpart",
            "status": "active",
            "relation_hint": "possibly_related",
        }
    ]


def test_memory_propose_append_with_reviewed_candidates_adds_without_superseding(
    tmp_path,
) -> None:
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
        message="Keep that, but also remember I prefer Rust examples for systems topics.",
        updates=[
            _update(
                "append_distinct",
                reviewed_candidate_ids=[str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples for systems topics.",
                    "evidence": (
                        "User said to keep the Python example memory and also remember "
                        "the Rust systems-topic preference."
                    ),
                    "scope": "counterpart",
                },
                reason=(
                    "The model reviewed the related candidate and chose to add "
                    "a distinct preference."
                ),
            )
        ],
    )

    projection = BeliefProjection(store)
    active = projection.list_active()
    assert [item.content for item in active] == [
        "User prefers Python examples.",
        "User prefers Rust examples for systems topics.",
    ]
    original_belief = projection.get_by_id(original.id)
    assert original_belief is not None
    assert original_belief.status == "active"

    events = list(SQLiteEventLog(store).iter())
    superseded = [event for event in events if event.kind == CognitiveEventKind.BELIEF_SUPERSEDED]
    assert superseded == []
    proposed = [event for event in events if event.kind == CognitiveEventKind.MEMORY_PROPOSED]
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    assert proposed[-1].payload["operation"] == "append_distinct"
    assert proposed[-1].payload["target_belief_ids"] == []
    assert proposed[-1].payload["reviewed_candidate_ids"] == [str(original.id)]
    assert formed[-1].payload["operation"] == "append_distinct"
    assert formed[-1].payload["target_belief_ids"] == []
    assert formed[-1].payload["reviewed_candidate_ids"] == [str(original.id)]

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["operation"] == "append_distinct"
    assert tool_output["results"][0]["decision"] == "accepted"
    assert tool_output["results"][0]["target_belief_ids"] == []
    assert tool_output["results"][0]["reviewed_candidate_ids"] == [str(original.id)]


def test_memory_propose_replace_requires_one_active_target_and_supersedes(tmp_path) -> None:
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
    assert active[0].supersedes is not None
    assert active[0].supersedes.id == str(original.id)
    superseded_original = projection.get_by_id(original.id)
    assert superseded_original is not None
    assert superseded_original.status == "superseded"
    assert superseded_original.superseded_by is not None
    assert superseded_original.superseded_by.id == str(active[0].id)

    events = list(SQLiteEventLog(store).iter())
    superseded = [event for event in events if event.kind == CognitiveEventKind.BELIEF_SUPERSEDED]
    assert len(superseded) == 1
    assert superseded[0].payload["operation"] == "replace"
    assert superseded[0].payload["target_belief_ids"] == [str(original.id)]
    assert superseded[0].payload["old_belief_id"] == str(original.id)
    assert superseded[0].payload["new_belief_id"] == str(active[0].id)
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["reason"] == "accepted_replace"


def test_memory_propose_replace_trusts_model_target_and_evidence(tmp_path) -> None:
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
        message="Rust seems nice.",
        updates=[
            _update(
                "replace",
                target_belief_ids=[str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said Rust seems nice.",
                    "scope": "counterpart",
                },
                reason="Maybe the user changed their example preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    assert [item.content for item in projection.list_active()] == ["User prefers Rust examples."]
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["next_action"] == "none"
    assert tool_output["results"][0]["reason"] == "accepted_replace"


def test_memory_propose_replace_does_not_gate_on_reason_words(tmp_path) -> None:
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
        message="Rust seems nice.",
        updates=[
            _update(
                "replace",
                target_belief_ids=[str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said Rust seems nice.",
                    "scope": "counterpart",
                },
                reason="The model thinks this should replace the previous preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    assert [item.content for item in projection.list_active()] == ["User prefers Rust examples."]
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["reason"] == "accepted_replace"


def test_memory_propose_merge_supersedes_multiple_targets_to_same_belief(tmp_path) -> None:
    store = _store(tmp_path)
    first = _append_memory(
        store,
        session_id="s1",
        content="User prefers concise answers.",
        evidence="User said they prefer concise answers.",
    )
    _run_updates(
        store,
        session_id="s1",
        message="Also remember that I prefer direct answers.",
        updates=[
            _update(
                "append_distinct",
                reviewed_candidate_ids=[str(first.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers direct answers.",
                    "evidence": "User said they also prefer direct answers.",
                    "scope": "counterpart",
                },
                reason=(
                    "The model reviewed the related concise-answer preference and chose "
                    "to add a distinct direct-answer preference."
                ),
            )
        ],
    )
    second = BeliefProjection(store).list_active()[-1]

    _run_updates(
        store,
        session_id="s1",
        message="Merge those as concise direct answers.",
        updates=[
            _update(
                "merge",
                target_belief_ids=[str(first.id), str(second.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers concise, direct answers.",
                    "evidence": "User asked to merge the concise and direct answer preferences.",
                    "scope": "counterpart",
                },
                reason="The two active beliefs describe the same answer-style preference.",
            )
        ],
    )

    projection = BeliefProjection(store)
    active = projection.list_active()
    assert len(active) == 1
    assert active[0].content == "User prefers concise, direct answers."
    assert {source.id for source in active[0].sources} >= {
        str(first.id),
        str(second.id),
        store.list_session_messages("s1")[-4].id,
    }
    old_first = projection.get_by_id(first.id)
    old_second = projection.get_by_id(second.id)
    assert old_first is not None
    assert old_second is not None
    assert old_first.status == "superseded"
    assert old_second.status == "superseded"
    assert old_first.superseded_by is not None
    assert old_second.superseded_by is not None
    assert old_first.superseded_by.id == str(active[0].id)
    assert old_second.superseded_by.id == str(active[0].id)

    events = list(SQLiteEventLog(store).iter())
    formed = [event for event in events if event.kind == CognitiveEventKind.BELIEF_FORMED]
    superseded = [event for event in events if event.kind == CognitiveEventKind.BELIEF_SUPERSEDED]
    assert formed[-1].payload["operation"] == "merge"
    assert len([event for event in superseded if event.payload.get("operation") == "merge"]) == 2
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["operation"] == "merge"
    assert tool_output["results"][0]["target_belief_ids"] == [str(first.id), str(second.id)]


def test_memory_propose_duplicate_target_belief_ids_are_rejected(tmp_path) -> None:
    store = _store(tmp_path)
    original = _append_memory(
        store,
        session_id="s1",
        content="User prefers concise answers.",
        evidence="User said they prefer concise answers.",
    )

    _run_updates(
        store,
        session_id="s1",
        message="Merge the duplicated target.",
        updates=[
            _update(
                "merge",
                target_belief_ids=[str(original.id), str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers concise answers.",
                    "evidence": "User asked to merge a duplicated target.",
                    "scope": "counterpart",
                },
                reason="The model accidentally supplied the same target twice.",
            )
        ],
    )

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "rejected"
    assert tool_output["results"][0]["reason"] == "duplicate_target_belief_ids"
    assert [item.id for item in BeliefProjection(store).list_active()] == [original.id]


def test_memory_propose_retract_requires_target_and_clear_evidence(tmp_path) -> None:
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
        message="Forget that Python examples preference.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [str(original.id)],
                "reviewed_candidate_ids": [],
                "target_hint": "",
                "reason": "User explicitly asked to forget the stored preference.",
                "memory": {
                    "type": "preference",
                    "content": "User prefers Python examples.",
                    "evidence": "User said: forget that Python examples preference.",
                    "scope": "counterpart",
                },
            }
        ],
    )

    projection = BeliefProjection(store)
    assert projection.list_active() == []
    retracted = projection.get_by_id(original.id)
    assert retracted is not None
    assert retracted.status == "retracted"
    events = list(SQLiteEventLog(store).iter())
    retract_events = [
        event for event in events if event.kind == CognitiveEventKind.BELIEF_RETRACTED
    ]
    assert len(retract_events) == 1
    assert retract_events[0].payload["operation"] == "retract"
    assert retract_events[0].payload["target_belief_ids"] == [str(original.id)]
    assert "new_belief_id" not in retract_events[0].payload


def test_memory_propose_retract_trusts_model_target_and_evidence(tmp_path) -> None:
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
        message="Python examples came up again.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [str(original.id)],
                "reviewed_candidate_ids": [],
                "target_hint": "",
                "reason": "The model thinks the user wants to forget this preference.",
                "memory": {
                    "type": "preference",
                    "content": "User prefers Python examples.",
                    "evidence": "User mentioned Python examples.",
                    "scope": "counterpart",
                },
            }
        ],
    )

    projection = BeliefProjection(store)
    assert projection.list_active() == []
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "accepted"
    assert tool_output["results"][0]["reason"] == "accepted_retract"


def test_memory_propose_targeted_retract_without_memory_evidence_waits(tmp_path) -> None:
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
        message="Python examples came up again.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [str(original.id)],
                "reviewed_candidate_ids": [],
                "target_hint": "",
                "reason": "The model thinks the user wants to forget this preference.",
            }
        ],
    )

    projection = BeliefProjection(store)
    assert [item.id for item in projection.list_active()] == [original.id]
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "pending_confirmation"
    assert tool_output["results"][0]["reason"] == "retract_requires_evidence"


def test_memory_propose_target_must_be_active_and_scope_matched(tmp_path) -> None:
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
        message="Forget that Python examples preference.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [str(original.id)],
                "reviewed_candidate_ids": [],
                "target_hint": "",
                "reason": "User explicitly asked to forget the stored preference.",
                "memory": {
                    "type": "preference",
                    "content": "User prefers Python examples.",
                    "evidence": "User said: forget that Python examples preference.",
                    "scope": "counterpart",
                },
            }
        ],
    )
    active_target = _append_memory(
        store,
        session_id="s1",
        content="User prefers concise answers.",
        evidence="User said they prefer concise answers.",
    )

    _run_updates(
        store,
        session_id="s1",
        message="Replace the old preference.",
        updates=[
            _update(
                "replace",
                target_belief_ids=[str(original.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said to replace the old preference with Rust.",
                    "scope": "counterpart",
                },
                reason="The target is no longer active.",
            ),
            _update(
                "replace",
                target_belief_ids=[str(active_target.id)],
                memory={
                    "type": "preference",
                    "content": "User prefers Rust examples.",
                    "evidence": "User said to replace the old preference with Rust.",
                    "scope": "global",
                },
                reason="The target scope does not match the requested memory scope.",
            ),
        ],
    )

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "rejected"
    assert tool_output["next_action"] == "explain_rejection"
    assert [item["reason"] for item in tool_output["results"]] == [
        "target_not_active",
        "target_scope_mismatch",
    ]


def test_memory_propose_no_target_retract_returns_candidates(tmp_path) -> None:
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
        message="Forget the Python examples preference.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [],
                "reviewed_candidate_ids": [],
                "target_hint": "Python examples preference",
                "reason": "User asked to forget a memory but did not provide a target id.",
                "memory": {
                    "type": "preference",
                    "content": "User prefers Python examples.",
                    "evidence": "User said: forget the Python examples preference.",
                    "scope": "counterpart",
                },
            }
        ],
    )

    projection = BeliefProjection(store)
    assert [item.id for item in projection.list_active()] == [original.id]
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "needs_target_selection"
    assert tool_output["results"][0]["candidates"][0]["id"] == str(original.id)


def test_memory_propose_no_memory_retract_returns_candidates_from_hint(tmp_path) -> None:
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
        message="Forget the Python examples preference.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [],
                "reviewed_candidate_ids": [],
                "target_hint": "Python examples preference",
                "reason": "User asked to forget a memory but did not provide a target id.",
            }
        ],
    )

    projection = BeliefProjection(store)
    assert [item.id for item in projection.list_active()] == [original.id]
    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "needs_target_selection"
    assert tool_output["results"][0]["operation"] == "retract"
    assert tool_output["results"][0]["candidates"][0] == {
        "id": str(original.id),
        "content": "User prefers Python examples.",
        "type": "preference",
        "scope": "counterpart",
        "status": "active",
        "relation_hint": "possibly_related",
    }


def test_memory_propose_no_target_retract_without_candidates_still_needs_selection(
    tmp_path,
) -> None:
    store = _store(tmp_path)

    _run_updates(
        store,
        session_id="s1",
        message="Forget the Zig examples preference.",
        updates=[
            {
                "operation": "retract",
                "target_belief_ids": [],
                "reviewed_candidate_ids": [],
                "target_hint": "Zig examples preference",
                "reason": "User asked to forget a memory but did not provide a target id.",
                "memory": {
                    "type": "preference",
                    "content": "User prefers Zig examples.",
                    "evidence": "User said: forget the Zig examples preference.",
                    "scope": "counterpart",
                },
            }
        ],
    )

    tool_output = json.loads(store.list_session_messages("s1")[-2].raw_content)
    assert tool_output["status"] == "needs_target_selection"
    assert tool_output["next_action"] == "review_candidates"
    assert tool_output["results"][0]["operation"] == "retract"
    assert tool_output["results"][0]["decision"] == "needs_target_selection"
    assert "candidates" not in tool_output["results"][0]


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
    assert list(SQLiteEventLog(store).iter(kinds=[CognitiveEventKind.BELIEF_FORMED])) == []


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
