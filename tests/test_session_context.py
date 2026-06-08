from __future__ import annotations

import json

from alpha_agent.cognition.processing_ledger import (
    BackgroundSourceRef,
    BackgroundStage,
    BackgroundStageRunStatus,
)
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.config import LLMContextConfig
from alpha_agent.runtime.chat_messages import wrap_system_reminder
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.state.store import StateStore


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def test_source_stream_append_read_and_latest_compressed_message(tmp_path) -> None:
    store = _store(tmp_path)

    user = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    compressed = store.append_compressed_message(
        session_id="s1",
        raw_content="handover",
        compression_point_ordinal=user.ordinal,
        compression_version="test-v1",
        metadata={"reason": "test"},
    )
    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="after",
    )

    messages = store.list_session_messages("s1")

    assert [message.id for message in messages] == [user.id, compressed.id, assistant.id]
    assert [message.kind for message in messages] == [
        "user_message",
        "compressed_message",
        "assistant_message",
    ]
    assert store.latest_session_ordinal("s1") == assistant.ordinal
    assert store.find_latest_compressed_message("s1") == compressed
    assert compressed.llm_role == "user"
    assert compressed.raw_content == wrap_system_reminder("handover")
    assert compressed.compression_point_ordinal == user.ordinal
    assert compressed.compression_version == "test-v1"
    assert compressed.metadata == {"reason": "test"}


def test_tool_replay_fields_survive_source_schema_refactor(tmp_path) -> None:
    store = _store(tmp_path)
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"alpha"}'},
        }
    ]

    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        model_content=None,
        reasoning_content="I need the lookup result.",
        tool_calls=tool_calls,
        provider_metadata={"provider": "test", "model": "m1"},
        source_metadata={"channel": "cli"},
        metadata={"tool_call_ids": ["call_1"]},
    )
    tool = store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content='{"ok": true}',
        model_content='{"visible": true}',
        tool_call_id="call_1",
        tool_result_id="trace_1",
        provider_metadata={"tool_name": "lookup"},
        source_metadata={"source": "runtime"},
        metadata={"trace_id": "trace_1"},
    )

    reloaded = store.list_session_messages("s1")

    assert reloaded[0] == assistant
    assert reloaded[1] == tool
    assert reloaded[0].reasoning_content == "I need the lookup result."
    assert reloaded[0].tool_calls == tool_calls
    assert reloaded[0].provider_metadata == {"provider": "test", "model": "m1"}
    assert reloaded[0].source_metadata == {"channel": "cli"}
    assert reloaded[0].metadata == {"tool_call_ids": ["call_1"]}
    assert reloaded[1].raw_content == '{"ok": true}'
    assert reloaded[1].model_content == '{"visible": true}'
    assert reloaded[1].tool_call_id == "call_1"
    assert reloaded[1].tool_result_id == "trace_1"


def test_reasoning_content_persists_and_replays_for_assistant_messages(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="hello",
    )
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="I will check.",
        reasoning_content="The user is asking for current context.",
    )
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        reasoning_content="A tool is needed.",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    )

    projection = SessionContextAssembler(store).load("s1")

    assert [message.reasoning_content for message in projection.source_messages] == [
        None,
        "The user is asking for current context.",
        "A tool is needed.",
    ]
    assert projection.chat_messages == [
        {"role": "user", "content": "hello"},
        {
            "role": "assistant",
            "content": "I will check.",
            "reasoning_content": "The user is asking for current context.",
        },
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "A tool is needed.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
    ]


def test_assembler_uses_all_source_messages_when_no_compression(tmp_path) -> None:
    store = _store(tmp_path)
    for index in range(1, 12):
        store.append_session_message(
            session_id="s1",
            kind="user_message" if index % 2 else "assistant_message",
            llm_role="user" if index % 2 else "assistant",
            raw_content=f"message {index}",
        )

    projection = SessionContextAssembler(store).load("s1")

    assert [message["content"] for message in projection.chat_messages] == [
        f"message {index}" for index in range(1, 12)
    ]
    assert projection.compressed_message is None


def test_assembler_after_compressed_message_uses_compressed_ordinal_boundary(tmp_path) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="tool request before boundary",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{}"},
            }
        ],
    )
    store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content='{"old": true}',
        tool_call_id="call_1",
    )
    compressed = store.append_compressed_message(
        session_id="s1",
        raw_content="handover through tool result",
        compression_point_ordinal=2,
        compression_version="test-v1",
    )
    after = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="continue",
    )

    projection = SessionContextAssembler(store).load("s1")

    assert projection.compressed_message == compressed
    assert [message.id for message in projection.source_messages] == [compressed.id, after.id]
    assert projection.chat_messages == [
        {"role": "user", "content": wrap_system_reminder("handover through tool result")},
        {"role": "user", "content": "continue"},
    ]


def test_session_context_keeps_runtime_handover_visible_and_background_artifacts_hidden(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    covered = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="covered source before handover",
    )
    compressed = store.append_compressed_message(
        session_id="s1",
        raw_content="RUNTIME_HANDOVER_CONTINUITY_SENTINEL",
        compression_point_ordinal=covered.ordinal,
        compression_version="test-v1",
    )
    fresh = store.append_session_message(
        session_id="s1",
        kind="user_message",
        llm_role="user",
        raw_content="fresh visible source",
    )
    trace = store.append_runtime_trace(
        session_id="s1",
        event_type="background.debug",
        content="BACKGROUND_RUNTIME_TRACE_SENTINEL",
    )
    service = CognitionStateStore(store)
    source_ref = BackgroundSourceRef("session_message", covered.id)
    trace_ref = BackgroundSourceRef("runtime_trace", trace.id)
    window = service.ledger.create_source_window(
        stage=BackgroundStage.EXTRACTION,
        target_unit="session:s1",
        source_refs=(source_ref, trace_ref),
        idempotency_key="phase9:session-context:window",
        metadata={"source_window_text": "BACKGROUND_SOURCE_WINDOW_SENTINEL"},
    )
    run = service.ledger.start_stage_run(
        worker_id="phase9-worker",
        stage=BackgroundStage.EXTRACTION,
        target_unit="session:s1",
        window_id=window.window_id,
        input_refs=(source_ref, trace_ref),
    )
    service.ledger.finish_stage_run(
        run.run_id,
        status=BackgroundStageRunStatus.FAILED,
        error="BACKGROUND_STAGE_RUN_SENTINEL",
    )
    service.write_audit_record(
        "phase9_guard",
        payload={"audit_payload": "COGNITION_STATE_AUDIT_SENTINEL"},
    )

    projection = SessionContextAssembler(store).load("s1")

    assert projection.compressed_message == compressed
    assert [message.id for message in projection.source_messages] == [compressed.id, fresh.id]
    rendered_context = json.dumps(projection.chat_messages, sort_keys=True)
    assert "RUNTIME_HANDOVER_CONTINUITY_SENTINEL" in rendered_context
    assert "fresh visible source" in rendered_context
    assert "covered source before handover" not in rendered_context
    for hidden_text in [
        "BACKGROUND_RUNTIME_TRACE_SENTINEL",
        "BACKGROUND_SOURCE_WINDOW_SENTINEL",
        "BACKGROUND_STAGE_RUN_SENTINEL",
        "COGNITION_STATE_AUDIT_SENTINEL",
    ]:
        assert hidden_text not in rendered_context


def test_truncate_tool_context_if_needed_truncates_unchecked_tool_replay_payloads(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    assembler = SessionContextAssembler(store)
    long_input = "A" * 12
    long_raw_output = "B" * 14
    long_model_output = "C" * 16
    marker = "<system-reminder>truncated</system-reminder>"
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": json.dumps(
                    {"query": {"deep": ["ok", {"body": long_input}]}},
                    ensure_ascii=False,
                ),
            },
            "metadata": {"provider_trace": "keep"},
        }
    ]
    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=tool_calls,
        provider_metadata={"provider": "test", "model": "m1"},
        source_metadata={"channel": "cli"},
        metadata={"diagnostic": {"trace_id": "a1"}},
    )
    tool = store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content=json.dumps({"result": {"body": long_raw_output}}, ensure_ascii=False),
        model_content=json.dumps({"visible": [{"text": long_model_output}]}, ensure_ascii=False),
        tool_call_id="call_1",
        tool_result_id="trace_1",
        provider_metadata={"tool_name": "lookup"},
        source_metadata={"source": "runtime"},
        metadata={"diagnostic": {"trace_id": "t1"}},
    )

    result = assembler.truncate_tool_context_if_needed(
        "s1",
        context_config=LLMContextConfig(
            tool_string_truncate_chars=5,
            expected_output_reserve_tokens=0,
            safety_margin_tokens=0,
        ),
        max_context_tokens=1,
    )

    reloaded = store.list_session_messages("s1")
    assistant_after = reloaded[0]
    tool_after = reloaded[1]
    arguments = json.loads(assistant_after.tool_calls[0]["function"]["arguments"])
    raw_output = json.loads(tool_after.raw_content)
    model_output = json.loads(tool_after.model_content or "{}")

    assert result.triggered is True
    assert result.checked_message_ids == [assistant.id, tool.id]
    assert result.truncated_message_ids == [assistant.id, tool.id]
    assert arguments["query"]["deep"][1]["body"] == "AAAAA" + marker
    assert raw_output["result"]["body"] == "BBBBB" + marker
    assert model_output["visible"][0]["text"] == "CCCCC" + marker
    assert assistant_after.metadata["truncate_checked"] is True
    assert tool_after.metadata["truncate_checked"] is True
    assert assistant_after.metadata["original_lengths"] == {
        "tool_calls[0].function.arguments.query.deep[1].body": 12
    }
    assert tool_after.metadata["original_lengths"] == {
        "model_content.visible[0].text": 16,
        "raw_content.result.body": 14,
    }
    assert assistant_after.provider_metadata == {"provider": "test", "model": "m1"}
    assert assistant_after.source_metadata == {"channel": "cli"}
    assert assistant_after.metadata["diagnostic"] == {"trace_id": "a1"}
    assert assistant_after.tool_calls[0]["metadata"] == {"provider_trace": "keep"}
    assert tool_after.provider_metadata == {"tool_name": "lookup"}
    assert tool_after.source_metadata == {"source": "runtime"}
    assert tool_after.metadata["diagnostic"] == {"trace_id": "t1"}


def test_truncate_tool_context_if_needed_skips_messages_before_latest_compressed_message(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    old_arguments = json.dumps({"body": "A" * 12})
    old_assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "old_call",
                "type": "function",
                "function": {"name": "lookup", "arguments": old_arguments},
            }
        ],
    )
    old_tool = store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content=json.dumps({"body": "B" * 12}),
        tool_call_id="old_call",
    )
    store.append_compressed_message(
        session_id="s1",
        raw_content="handover",
        compression_point_ordinal=old_tool.ordinal,
        compression_version="test-v1",
    )
    new_assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "new_call",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"body": "C" * 12}),
                },
            }
        ],
    )

    SessionContextAssembler(store).truncate_tool_context_if_needed(
        "s1",
        context_config=LLMContextConfig(
            tool_string_truncate_chars=5,
            expected_output_reserve_tokens=0,
            safety_margin_tokens=0,
        ),
        max_context_tokens=1,
    )

    messages = store.list_session_messages("s1")
    old_assistant_after = messages[0]
    old_tool_after = messages[1]
    new_assistant_after = messages[3]

    assert old_assistant_after.id == old_assistant.id
    assert old_assistant_after.tool_calls[0]["function"]["arguments"] == old_arguments
    assert old_assistant_after.metadata == {}
    assert old_tool_after.raw_content == json.dumps({"body": "B" * 12})
    assert old_tool_after.metadata == {}
    assert new_assistant_after.id == new_assistant.id
    assert json.loads(new_assistant_after.tool_calls[0]["function"]["arguments"]) == {
        "body": "CCCCC<system-reminder>truncated</system-reminder>"
    }
    assert new_assistant_after.metadata["truncate_checked"] is True


def test_truncate_tool_context_if_needed_truncates_plain_text_tool_outputs(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    valid_arguments = json.dumps({"body": "A" * 12})
    assistant = store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": valid_arguments},
            }
        ],
    )
    plain_tool = store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content="plain text tool output",
        tool_call_id="call_1",
        metadata={"tool_output_kind": "text"},
    )

    result = SessionContextAssembler(store).truncate_tool_context_if_needed(
        "s1",
        context_config=LLMContextConfig(
            tool_string_truncate_chars=5,
            expected_output_reserve_tokens=0,
            safety_margin_tokens=0,
        ),
        max_context_tokens=1,
    )

    messages = store.list_session_messages("s1")
    assistant_after = messages[0]
    plain_tool_after = messages[1]

    assert result.checked_message_ids == [assistant.id, plain_tool.id]
    assert result.truncated_message_ids == [assistant.id, plain_tool.id]
    assert assistant_after.id == assistant.id
    assert json.loads(assistant_after.tool_calls[0]["function"]["arguments"]) == {
        "body": "AAAAA<system-reminder>truncated</system-reminder>"
    }
    assert plain_tool_after.id == plain_tool.id
    assert plain_tool_after.raw_content == "plain<system-reminder>truncated</system-reminder>"
    assert plain_tool_after.metadata["tool_output_kind"] == "text"
    assert plain_tool_after.metadata["truncate_checked"] is True
    assert plain_tool_after.metadata["original_lengths"] == {"raw_content": 22}


def test_truncate_tool_context_if_needed_marks_checked_without_truncating_short_strings(
    tmp_path,
) -> None:
    store = _store(tmp_path)
    store.append_session_message(
        session_id="s1",
        kind="assistant_message",
        llm_role="assistant",
        raw_content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"body": "ABCDE"}),
                },
            }
        ],
    )
    store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content=json.dumps({"body": "12345"}),
        tool_call_id="call_1",
    )

    result = SessionContextAssembler(store).truncate_tool_context_if_needed(
        "s1",
        context_config=LLMContextConfig(
            tool_string_truncate_chars=5,
            expected_output_reserve_tokens=0,
            safety_margin_tokens=0,
        ),
        max_context_tokens=1,
    )

    assistant_after, tool_after = store.list_session_messages("s1")

    assert result.checked_message_ids == [assistant_after.id, tool_after.id]
    assert result.truncated_message_ids == []
    assert json.loads(assistant_after.tool_calls[0]["function"]["arguments"]) == {
        "body": "ABCDE"
    }
    assert json.loads(tool_after.raw_content) == {"body": "12345"}
    assert assistant_after.metadata["truncate_checked"] is True
    assert assistant_after.metadata["original_lengths"] == {}
    assert tool_after.metadata["truncate_checked"] is True
    assert tool_after.metadata["original_lengths"] == {}


def test_truncate_tool_context_if_needed_skips_already_checked_messages(tmp_path) -> None:
    store = _store(tmp_path)
    checked = store.append_session_message(
        session_id="s1",
        kind="tool_message",
        llm_role="tool",
        raw_content="{not valid json",
        tool_call_id="call_1",
        metadata={"truncate_checked": True, "original_lengths": {}},
    )

    result = SessionContextAssembler(store).truncate_tool_context_if_needed(
        "s1",
        context_config=LLMContextConfig(
            tool_string_truncate_chars=5,
            expected_output_reserve_tokens=0,
            safety_margin_tokens=0,
        ),
        max_context_tokens=1,
    )

    checked_after = store.list_session_messages("s1")[0]

    assert result.checked_message_ids == []
    assert result.truncated_message_ids == []
    assert checked_after.id == checked.id
    assert checked_after.raw_content == "{not valid json"
    assert checked_after.metadata == {"truncate_checked": True, "original_lengths": {}}
