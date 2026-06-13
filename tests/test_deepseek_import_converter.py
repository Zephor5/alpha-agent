from __future__ import annotations

import json
from copy import deepcopy

import pytest

from alpha_agent.conversation_import.deepseek import (
    DeepSeekExportConversionError,
    convert_deepseek_export,
)
from alpha_agent.daemon.conversation_import import ConversationImportService
from alpha_agent.state.store import StateStore


def _deepseek_export() -> list[dict[str, object]]:
    return [
        {
            "id": "conv_1",
            "title": "Market discussion",
            "inserted_at": "2026-01-01T10:00:00.000000+08:00",
            "updated_at": "2026-01-01T10:05:00.000000+08:00",
            "mapping": {
                "root": {
                    "id": "root",
                    "parent": None,
                    "children": ["1"],
                    "message": None,
                },
                "1": {
                    "id": "1",
                    "parent": "root",
                    "children": ["2"],
                    "message": {
                        "files": [],
                        "model": "deepseek-chat",
                        "inserted_at": "2026-01-01T10:01:00.000000+08:00",
                        "fragments": [{"type": "REQUEST", "content": "Why did stocks fall?"}],
                    },
                },
                "2": {
                    "id": "2",
                    "parent": "1",
                    "children": [],
                    "message": {
                        "files": [],
                        "model": "deepseek-reasoner",
                        "inserted_at": "2026-01-01T10:01:00.000000+08:00",
                        "fragments": [
                            {
                                "type": "SEARCH",
                                "results": [
                                    {
                                        "url": "https://example.test/a",
                                        "title": "A",
                                        "snippet": "large text",
                                    },
                                    {"url": "https://example.test/b", "title": "B"},
                                ],
                            },
                            {"type": "TOOL_OPEN", "content": "tool open details"},
                            {"type": "TOOL_SEARCH", "content": "tool search details"},
                            {"type": "THINK", "content": "hidden reasoning"},
                            {"type": "RESPONSE", "content": "Because risk appetite weakened."},
                        ],
                    },
                },
            },
        }
    ]


def _convert(source: object) -> dict[str, object]:
    return convert_deepseek_export(json.dumps(source))


def _assert_import_validator_accepts(
    payload: dict[str, object],
    tmp_path,
    *,
    expected_messages: int = 2,
) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    summary = ConversationImportService(store).import_payload(json.dumps(payload), dry_run=True)
    assert summary.source_provider == "deepseek"
    assert summary.messages_inserted == expected_messages


def test_converts_linear_deepseek_export_to_normalized_import_payload(tmp_path) -> None:
    payload = _convert(_deepseek_export())

    assert payload["source_provider"] == "deepseek"
    assert payload["timezone"] == "+08:00"
    assert payload["metadata"] == {
        "source_format": "deepseek_export",
        "converter": "alpha-agent",
    }

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    assert len(conversations) == 1
    conversation = conversations[0]
    assert conversation == {
        "external_conversation_id": "conv_1",
        "title": "Market discussion",
        "created_at": "2026-01-01T10:00:00.000000+08:00",
        "updated_at": "2026-01-01T10:05:00.000000+08:00",
        "messages": [
            {
                "external_message_id": "1",
                "role": "user",
                "content": "Why did stocks fall?",
                "created_at": "2026-01-01T10:01:00.000000+08:00",
                "metadata": {"deepseek": {"model": "deepseek-chat"}},
            },
            {
                "external_message_id": "2",
                "role": "assistant",
                "content": "Because risk appetite weakened.",
                "created_at": "2026-01-01T10:01:00.000001+08:00",
                "metadata": {
                    "deepseek": {
                        "model": "deepseek-reasoner",
                        "omitted_fragment_types": [
                            "SEARCH",
                            "TOOL_OPEN",
                            "TOOL_SEARCH",
                            "THINK",
                        ],
                        "think_fragment_count": 1,
                        "search_result_count": 2,
                    }
                },
            },
        ],
    }
    assert "hidden reasoning" not in json.dumps(payload)
    assert "large text" not in json.dumps(payload)
    assert "tool open details" not in json.dumps(payload)
    assert "tool search details" not in json.dumps(payload)
    assert "original_inserted_at" not in json.dumps(payload)
    _assert_import_validator_accepts(payload, tmp_path)


def test_adjusts_timestamps_by_tree_order_when_deepseek_times_move_backwards(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    node = mapping["2"]
    assert isinstance(node, dict)
    message = node["message"]
    assert isinstance(message, dict)
    message["inserted_at"] = "2026-01-01T10:00:59.999000+08:00"

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert messages[1]["created_at"] == "2026-01-01T10:01:00.000001+08:00"
    _assert_import_validator_accepts(payload, tmp_path)


def test_filters_server_busy_response_child_from_deepseek_branch(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    first_node = mapping["1"]
    assert isinstance(first_node, dict)
    first_node["children"] = ["busy", "2"]
    mapping["busy"] = {
        "id": "busy",
        "parent": "1",
        "children": [],
        "message": {
            "files": [],
            "model": "deepseek-chat",
            "inserted_at": "2026-01-01T10:01:00.000000+08:00",
            "fragments": [
                {
                    "type": "RESPONSE",
                    "content": "服务器繁忙，请稍后再试。",
                }
            ],
        },
    }

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1", "2"]
    assert all("服务器繁忙" not in message["content"] for message in messages)
    _assert_import_validator_accepts(payload, tmp_path)


def test_chooses_last_child_after_filtering_busy_deepseek_branch(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    first_node = mapping["1"]
    assert isinstance(first_node, dict)
    first_node["children"] = ["busy", "2", "alt"]
    mapping["busy"] = {
        "id": "busy",
        "parent": "1",
        "children": [],
        "message": {
            "files": [],
            "model": "deepseek-chat",
            "inserted_at": "2026-01-01T10:01:00.000000+08:00",
            "fragments": [{"type": "RESPONSE", "content": "服务器繁忙，请稍后再试。"}],
        },
    }
    mapping["alt"] = {
        "id": "alt",
        "parent": "1",
        "children": [],
        "message": {
            "files": [],
            "model": "deepseek-chat",
            "inserted_at": "2026-01-01T10:01:01.000000+08:00",
            "fragments": [{"type": "RESPONSE", "content": "alternate final answer"}],
        },
    }

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1", "alt"]
    assert messages[1]["content"] == "alternate final answer"
    assert all("服务器繁忙" not in message["content"] for message in messages)
    _assert_import_validator_accepts(payload, tmp_path)


def test_filters_terminal_server_busy_response_children_from_deepseek_branch() -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    first_node = mapping["1"]
    assert isinstance(first_node, dict)
    first_node["children"] = ["busy_1", "busy_2"]
    del mapping["2"]
    for index, busy_id in enumerate(("busy_1", "busy_2"), start=1):
        mapping[busy_id] = {
            "id": busy_id,
            "parent": "1",
            "children": [],
            "message": {
                "files": [],
                "model": "deepseek-chat",
                "inserted_at": f"2026-01-01T10:01:0{index}.000000+08:00",
                "fragments": [
                    {
                        "type": "RESPONSE",
                        "content": "服务器繁忙，请稍后再试。",
                    }
                ],
            },
        }

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1"]


def test_skips_terminal_think_only_interrupted_assistant_message(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    assistant_node = mapping["2"]
    assert isinstance(assistant_node, dict)
    assistant_message = assistant_node["message"]
    assert isinstance(assistant_message, dict)
    assistant_message["fragments"] = [
        {"type": "THINK", "content": "interrupted reasoning only"}
    ]

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1"]
    assert "interrupted reasoning only" not in json.dumps(payload)
    _assert_import_validator_accepts(payload, tmp_path, expected_messages=1)


def test_skips_empty_fragments_interrupted_message(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    assistant_node = mapping["2"]
    assert isinstance(assistant_node, dict)
    assistant_message = assistant_node["message"]
    assert isinstance(assistant_message, dict)
    assistant_message["fragments"] = []

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1"]
    _assert_import_validator_accepts(payload, tmp_path, expected_messages=1)


def test_skips_internal_think_only_interrupted_assistant_message_and_continues(
    tmp_path,
) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    assistant_node = mapping["2"]
    assert isinstance(assistant_node, dict)
    assistant_node["children"] = ["3"]
    assistant_message = assistant_node["message"]
    assert isinstance(assistant_message, dict)
    assistant_message["fragments"] = [
        {"type": "THINK", "content": "interrupted reasoning only"}
    ]
    mapping["3"] = {
        "id": "3",
        "parent": "2",
        "children": [],
        "message": {
            "files": [],
            "model": "deepseek-chat",
            "inserted_at": "2026-01-01T10:02:00.000000+08:00",
            "fragments": [{"type": "REQUEST", "content": "continue with a new prompt"}],
        },
    }

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert [message["external_message_id"] for message in messages] == ["1", "3"]
    assert [message["role"] for message in messages] == ["user", "user"]
    assert "interrupted reasoning only" not in json.dumps(payload)
    _assert_import_validator_accepts(payload, tmp_path)


def test_converts_deepseek_files_to_user_messages_before_request(tmp_path) -> None:
    source = _deepseek_export()
    mapping = source[0]["mapping"]
    assert isinstance(mapping, dict)
    user_node = mapping["1"]
    assert isinstance(user_node, dict)
    user_message = user_node["message"]
    assert isinstance(user_message, dict)
    user_message["files"] = [
        {
            "id": "file-301f4afb-6972-4dff-8787-febf7b10a3de",
            "file_name": "README.md",
        }
    ]

    payload = _convert(source)

    conversations = payload["conversations"]
    assert isinstance(conversations, list)
    conversation = conversations[0]
    assert isinstance(conversation, dict)
    messages = conversation["messages"]
    assert isinstance(messages, list)
    assert messages[:2] == [
        {
            "external_message_id": "1:file:0",
            "role": "user",
            "content": "file: README.md",
            "created_at": "2026-01-01T10:01:00.000000+08:00",
            "metadata": {
                "deepseek": {
                    "model": "deepseek-chat",
                    "file_id": "file-301f4afb-6972-4dff-8787-febf7b10a3de",
                }
            },
        },
        {
            "external_message_id": "1",
            "role": "user",
            "content": "Why did stocks fall?",
            "created_at": "2026-01-01T10:01:00.000001+08:00",
            "metadata": {"deepseek": {"model": "deepseek-chat"}},
        },
    ]
    assert messages[2]["external_message_id"] == "2"
    assert messages[2]["created_at"] == "2026-01-01T10:01:00.000002+08:00"
    _assert_import_validator_accepts(payload, tmp_path, expected_messages=3)


@pytest.mark.parametrize(
    ("mutate", "expected_fragment"),
    [
        (
            lambda source: source[0]["mapping"]["1"]["message"].update(
                {"files": [{"id": "file_1", "file_name": ""}]}
            ),
            "file_name must be a non-empty string",
        ),
        (
            lambda source: source[0]["mapping"]["root"].update(
                {
                    "message": {
                        "files": [],
                        "model": "deepseek-chat",
                        "inserted_at": "2026-01-01T10:00:00.000000+08:00",
                        "fragments": [{"type": "REQUEST", "content": "root content"}],
                    }
                }
            ),
            "root message must be null",
        ),
        (
            lambda source: source[0]["mapping"]["1"]["message"]["fragments"].append(
                {"type": "AUDIO", "content": "voice"}
            ),
            "unsupported fragment type",
        ),
        (
            lambda source: source[0]["mapping"]["1"]["message"]["fragments"].append(
                {"type": "RESPONSE", "content": "mixed role"}
            ),
            "must not contain both REQUEST and RESPONSE",
        ),
        (
            lambda source: source[0]["mapping"]["1"]["message"].update(
                {"inserted_at": "2026-01-01T10:01:00.000000+09:00"}
            ),
            "message timestamp offsets must be consistent",
        ),
    ],
)
def test_rejects_unsupported_deepseek_shapes(mutate, expected_fragment: str) -> None:
    source = deepcopy(_deepseek_export())
    mutate(source)

    with pytest.raises(DeepSeekExportConversionError) as exc_info:
        _convert(source)

    assert any(expected_fragment in error.message for error in exc_info.value.errors)


def test_rejects_non_array_top_level() -> None:
    with pytest.raises(DeepSeekExportConversionError) as exc_info:
        _convert({"conversations": _deepseek_export()})

    assert exc_info.value.errors[0].path == "$"
    assert "top-level DeepSeek export must be an array" in exc_info.value.errors[0].message
