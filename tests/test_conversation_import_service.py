from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from alpha_agent.daemon.conversation_import import (
    ConversationImportService,
    ConversationImportValidationFailed,
)
from alpha_agent.runtime.counterpart_router import DEFAULT_COUNTERPART_ID
from alpha_agent.state.store import StateStore


def _store(tmp_path) -> StateStore:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    return store


def _payload(
    *,
    timezone: str | None = "Asia/Shanghai",
    messages: list[dict[str, object]] | None = None,
    external_conversation_id: str = "conv_1",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "source_provider": "chatgpt",
        "conversations": [
            {
                "external_conversation_id": external_conversation_id,
                "title": "Alpha Agent design",
                "created_at": "2026-01-01T10:00:00+08:00",
                "updated_at": "2026-01-01T10:04:00+08:00",
                "messages": messages
                if messages is not None
                else [
                    {
                        "external_message_id": "msg_1",
                        "role": "system",
                        "content": "You are a helpful assistant.",
                        "created_at": "2026-01-01T10:01:00+08:00",
                    },
                    {
                        "external_message_id": "msg_2",
                        "role": "user",
                        "content": "I prefer direct feedback.",
                        "created_at": "2026-01-01T10:02:00+08:00",
                    },
                    {
                        "external_message_id": "msg_3",
                        "role": "assistant",
                        "content": "Understood.",
                        "created_at": "2026-01-01T10:03:00+08:00",
                    },
                ],
                "metadata": {"topic": "design"},
            }
        ],
        "metadata": {"export": "normalized"},
    }
    if timezone is not None:
        payload["timezone"] = timezone
    return payload


def _run(
    service: ConversationImportService,
    payload: dict[str, object],
    *,
    dry_run: bool = False,
):
    return service.import_payload(
        json.dumps(payload),
        input_name="external.json",
        dry_run=dry_run,
    )


def test_valid_import_persists_hidden_session_messages_mapping_and_counterpart(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    summary = _run(service, _payload())

    assert summary.batch_id is not None
    assert summary.source_provider == "chatgpt"
    assert summary.conversations_seen == 1
    assert summary.messages_seen == 3
    assert summary.conversations_created == 1
    assert summary.conversations_reused == 0
    assert summary.messages_inserted == 3
    assert summary.messages_deduped == 0

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    assert imported.title == "Alpha Agent design"
    assert store.is_import_session(imported.session_id) is True

    record = store.get_session_record(imported.session_id)
    assert record is not None
    assert record.timezone == "Asia/Shanghai"
    assert record.created_at == "2026-01-01T02:01:00+00:00"
    assert record.updated_at == "2026-01-01T02:03:00+00:00"

    messages = store.list_session_messages(imported.session_id)
    assert [message.kind for message in messages] == [
        "system_message",
        "user_message",
        "assistant_message",
    ]
    assert [message.llm_role for message in messages] == ["system", "user", "assistant"]
    assert [message.created_at for message in messages] == [
        "2026-01-01T02:01:00+00:00",
        "2026-01-01T02:02:00+00:00",
        "2026-01-01T02:03:00+00:00",
    ]
    assert messages[0].raw_content == "You are a helpful assistant."

    imported_messages = store.list_imported_messages(
        source_provider="chatgpt",
        external_conversation_id="conv_1",
    )
    assert [message.external_message_id for message in imported_messages] == [
        "msg_1",
        "msg_2",
        "msg_3",
    ]
    assert [message.session_message_id for message in imported_messages] == [
        message.id for message in messages
    ]

    binding = store.get_session_counterpart(imported.session_id)
    assert binding is not None
    assert binding.counterpart_id == str(DEFAULT_COUNTERPART_ID)

    assert store.list_runtime_traces(imported.session_id) == []
    assert store.find_latest_session_time_reminder(imported.session_id) is None
    assert store.find_latest_compressed_message(imported.session_id) is None

    status = store.get_import_status_summary(summary.batch_id)
    assert status is not None
    assert status.batch_id == summary.batch_id
    assert status.messages_inserted == 3
    assert status.extraction_pending == 3


def test_dry_run_plans_without_writes(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    summary = _run(service, _payload(), dry_run=True)

    assert summary.batch_id is None
    assert summary.dry_run is True
    assert summary.conversations_created == 1
    assert summary.messages_inserted == 3
    assert store.get_imported_conversation("chatgpt", "conv_1") is None
    assert store.list_session_ids() == []
    assert store.list_import_batches() == []


def test_duplicate_reimport_creates_new_batch_and_no_duplicate_session_messages(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    first = _run(service, _payload())
    second = _run(service, _payload())

    assert first.batch_id != second.batch_id
    assert second.conversations_created == 0
    assert second.conversations_reused == 1
    assert second.messages_inserted == 0
    assert second.messages_deduped == 3

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    assert len(store.list_session_messages(imported.session_id)) == 3
    assert len(store.list_imported_messages(source_provider="chatgpt")) == 3

    batches = store.list_import_batches()
    assert [batch.id for batch in batches] == [first.batch_id, second.batch_id]
    assert batches[1].messages_deduped == 3


def test_existing_conversation_append_writes_only_new_messages(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    _run(service, _payload())

    appended = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "system",
                "content": "You are a helpful assistant.",
                "created_at": "2026-01-01T10:01:00+08:00",
            },
            {
                "external_message_id": "msg_2",
                "role": "user",
                "content": "I prefer direct feedback.",
                "created_at": "2026-01-01T10:02:00+08:00",
            },
            {
                "external_message_id": "msg_3",
                "role": "assistant",
                "content": "Understood.",
                "created_at": "2026-01-01T10:03:00+08:00",
            },
            {
                "external_message_id": "msg_4",
                "role": "user",
                "content": "Also keep answers concise.",
                "created_at": "2026-01-01T10:04:00+08:00",
            },
        ]
    )

    summary = _run(service, appended)

    assert summary.conversations_created == 0
    assert summary.conversations_reused == 1
    assert summary.messages_inserted == 1
    assert summary.messages_deduped == 3

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    messages = store.list_session_messages(imported.session_id)
    assert [message.external_message_id for message in store.list_imported_messages()] == [
        "msg_1",
        "msg_2",
        "msg_3",
        "msg_4",
    ]
    assert [message.raw_content for message in messages] == [
        "You are a helpful assistant.",
        "I prefer direct feedback.",
        "Understood.",
        "Also keep answers concise.",
    ]
    record = store.get_session_record(imported.session_id)
    assert record is not None
    assert record.updated_at == "2026-01-01T02:04:00+00:00"


@pytest.mark.parametrize(
    ("messages", "expected_path", "expected_fragment"),
    [
        (
            [
                {
                    "external_message_id": "msg_1",
                    "role": "critic",
                    "content": "bad role",
                    "created_at": "2026-01-01T10:01:00+08:00",
                }
            ],
            "conversations[0].messages[0].role",
            "role must be one of",
        ),
        (
            [
                {
                    "external_message_id": "msg_1",
                    "role": "user",
                    "content": "missing timezone",
                    "created_at": "2026-01-01T10:01:00",
                }
            ],
            "conversations[0].messages[0].created_at",
            "timezone",
        ),
        (
            [
                {
                    "external_message_id": "msg_1",
                    "role": "user",
                    "content": "",
                    "created_at": "2026-01-01T10:01:00+08:00",
                }
            ],
            "conversations[0].messages[0].content",
            "non-empty",
        ),
        (
            [
                {
                    "external_message_id": "msg_1",
                    "role": "tool",
                    "tool_call_id": "missing_call",
                    "content": "result",
                    "created_at": "2026-01-01T10:01:00+08:00",
                }
            ],
            "conversations[0].messages[0].tool_call_id",
            "matching assistant",
        ),
        (
            [
                {
                    "external_message_id": "msg_1",
                    "role": "assistant",
                    "content": "hidden reasoning",
                    "reasoning_content": "no",
                    "created_at": "2026-01-01T10:01:00+08:00",
                }
            ],
            "conversations[0].messages[0].reasoning_content",
            "not supported",
        ),
    ],
)
def test_validation_errors_include_payload_paths(
    tmp_path,
    messages: list[dict[str, object]],
    expected_path: str,
    expected_fragment: str,
) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, _payload(messages=messages))

    errors = exc_info.value.errors
    assert any(
        error.path == expected_path and expected_fragment in error.message for error in errors
    )
    assert store.list_import_batches() == []


@pytest.mark.parametrize(
    ("payload_update", "expected_path"),
    [
        ({"unexpected": True}, "unexpected"),
        ({"conversations": [{"timezone": "UTC"}]}, "conversations[0].timezone"),
        (
            {"conversations": [{"source_provider": "claude"}]},
            "conversations[0].source_provider",
        ),
        (
            {"conversations": [{"unknown": "value"}]},
            "conversations[0].unknown",
        ),
    ],
)
def test_rejects_unknown_top_level_and_conversation_fields(
    tmp_path,
    payload_update: dict[str, object],
    expected_path: str,
) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    payload = _payload()
    if "conversations" in payload_update:
        conversation_update = payload_update["conversations"]
        assert isinstance(conversation_update, list)
        update = conversation_update[0]
        assert isinstance(update, dict)
        conversations = payload["conversations"]
        assert isinstance(conversations, list)
        conversation = conversations[0]
        assert isinstance(conversation, dict)
        conversation.update(update)
    else:
        payload.update(payload_update)

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, payload)

    assert any(
        error.path == expected_path and "not supported" in error.message
        for error in exc_info.value.errors
    )
    assert store.list_import_batches() == []


@pytest.mark.parametrize(
    ("tool_call", "expected_path", "expected_fragment"),
    [
        (
            {"id": "", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            "conversations[0].messages[0].tool_calls[0].id",
            "non-empty",
        ),
        (
            {"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}},
            "conversations[0].messages[0].tool_calls[0].type",
            "function",
        ),
        (
            {
                "id": "call_1",
                "type": "web_search",
                "function": {"name": "lookup", "arguments": "{}"},
            },
            "conversations[0].messages[0].tool_calls[0].type",
            "function",
        ),
        (
            {"id": "call_1", "type": "function", "function": "lookup"},
            "conversations[0].messages[0].tool_calls[0].function",
            "object",
        ),
        (
            {"id": "call_1", "type": "function", "function": {}},
            "conversations[0].messages[0].tool_calls[0].function.name",
            "non-empty",
        ),
        (
            {"id": "call_1", "type": "function", "function": {"name": "lookup"}},
            "conversations[0].messages[0].tool_calls[0].function.arguments",
            "JSON string",
        ),
        (
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup", "arguments": "{not json"},
            },
            "conversations[0].messages[0].tool_calls[0].function.arguments",
            "valid JSON",
        ),
    ],
)
def test_rejects_malformed_assistant_tool_calls(
    tmp_path,
    tool_call: dict[str, object],
    expected_path: str,
    expected_fragment: str,
) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    payload = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "assistant",
                "tool_calls": [tool_call],
                "created_at": "2026-01-01T10:01:00+08:00",
            }
        ]
    )

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, payload)

    assert any(
        error.path == expected_path and expected_fragment in error.message
        for error in exc_info.value.errors
    )
    assert store.list_import_batches() == []


def test_rejects_middle_insertion_for_existing_conversation(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    _run(service, _payload())

    middle_insert = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "system",
                "content": "You are a helpful assistant.",
                "created_at": "2026-01-01T10:01:00+08:00",
            },
            {
                "external_message_id": "msg_2",
                "role": "user",
                "content": "I prefer direct feedback.",
                "created_at": "2026-01-01T10:02:00+08:00",
            },
            {
                "external_message_id": "msg_2b",
                "role": "assistant",
                "content": "Inserted historical answer.",
                "created_at": "2026-01-01T10:02:30+08:00",
            },
            {
                "external_message_id": "msg_3",
                "role": "assistant",
                "content": "Understood.",
                "created_at": "2026-01-01T10:03:00+08:00",
            },
        ]
    )

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, middle_insert)

    assert any(
        error.path == "conversations[0].messages[2].created_at"
        and "strictly later than the latest imported message" in error.message
        for error in exc_info.value.errors
    )

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    assert len(store.list_session_messages(imported.session_id)) == 3


def test_existing_conversation_append_ignores_deduped_message_order(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    _run(service, _payload())

    payload = _payload(
        messages=[
            {
                "external_message_id": "msg_2",
                "role": "user",
                "content": "I prefer direct feedback.",
                "created_at": "2026-01-01T10:02:00+08:00",
            },
            {
                "external_message_id": "msg_1",
                "role": "system",
                "content": "You are a helpful assistant.",
                "created_at": "2026-01-01T10:01:00+08:00",
            },
            {
                "external_message_id": "msg_4",
                "role": "user",
                "content": "Append after dedup.",
                "created_at": "2026-01-01T10:04:00+08:00",
            },
        ]
    )

    summary = _run(service, payload)

    assert summary.messages_deduped == 2
    assert summary.messages_inserted == 1
    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    messages = store.list_session_messages(imported.session_id)
    assert [message.raw_content for message in messages] == [
        "You are a helpful assistant.",
        "I prefer direct feedback.",
        "Understood.",
        "Append after dedup.",
    ]


def test_existing_conversation_rejects_non_increasing_new_append_messages(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    _run(service, _payload())

    payload = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "system",
                "content": "You are a helpful assistant.",
                "created_at": "2026-01-01T10:01:00+08:00",
            },
            {
                "external_message_id": "msg_5",
                "role": "assistant",
                "content": "New later response.",
                "created_at": "2026-01-01T10:05:00+08:00",
            },
            {
                "external_message_id": "msg_4",
                "role": "user",
                "content": "New earlier prompt.",
                "created_at": "2026-01-01T10:04:00+08:00",
            },
        ]
    )

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, payload)

    assert any(
        error.path == "conversations[0].messages[2].created_at"
        and "strictly later than the previous new message" in error.message
        for error in exc_info.value.errors
    )


def test_rejects_non_increasing_append_messages(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    payload = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "user",
                "content": "first",
                "created_at": "2026-01-01T10:02:00+08:00",
            },
            {
                "external_message_id": "msg_2",
                "role": "assistant",
                "content": "second",
                "created_at": "2026-01-01T10:02:00+08:00",
            },
        ]
    )

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        _run(service, payload)

    assert any(
        error.path == "conversations[0].messages[1].created_at"
        and "strictly later" in error.message
        for error in exc_info.value.errors
    )


def test_absent_top_level_timezone_derives_hidden_session_fixed_offset(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    summary = _run(service, _payload(timezone=None))

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    record = store.get_session_record(imported.session_id)
    assert record is not None
    assert record.timezone == "+08:00"
    assert summary.messages_inserted == 3


def test_assistant_tool_calls_and_tool_results_are_persisted_as_history(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    tool_calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
        }
    ]
    payload = _payload(
        messages=[
            {
                "external_message_id": "msg_1",
                "role": "assistant",
                "tool_calls": tool_calls,
                "created_at": "2026-01-01T10:01:00+08:00",
            },
            {
                "external_message_id": "msg_2",
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"ok": true}',
                "created_at": "2026-01-01T10:02:00+08:00",
            },
        ]
    )

    _run(service, payload)

    imported = store.get_imported_conversation("chatgpt", "conv_1")
    assert imported is not None
    messages = store.list_session_messages(imported.session_id)
    assert messages[0].kind == "assistant_message"
    assert messages[0].raw_content == ""
    assert messages[0].tool_calls == tool_calls
    assert messages[1].kind == "tool_message"
    assert messages[1].tool_call_id == "call_1"


def test_payload_size_limit_is_enforced_before_json_parsing(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)

    with pytest.raises(ConversationImportValidationFailed) as exc_info:
        service.import_payload(b"x" * (service.MAX_PAYLOAD_BYTES + 1), input_name="large.json")

    assert exc_info.value.errors[0].path == "$"
    assert "50 MB" in exc_info.value.errors[0].message


def test_store_latest_imported_message_timestamp_uses_utc_instants(tmp_path) -> None:
    store = _store(tmp_path)
    service = ConversationImportService(store)
    _run(service, _payload())

    latest = store.latest_imported_message_timestamp("chatgpt", "conv_1")

    assert latest == "2026-01-01T02:03:00+00:00"
    assert datetime.fromisoformat(latest).tzinfo == UTC
