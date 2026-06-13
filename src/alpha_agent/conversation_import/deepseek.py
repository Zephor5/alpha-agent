"""Convert DeepSeek raw conversation exports into Alpha's normalized import JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

NormalizedImportPayload = dict[str, Any]
_DeepSeekRole = Literal["user", "assistant"]

_REQUEST_FRAGMENT = "REQUEST"
_RESPONSE_FRAGMENT = "RESPONSE"
_THINK_FRAGMENT = "THINK"
_SEARCH_FRAGMENT = "SEARCH"
_TOOL_OPEN_FRAGMENT = "TOOL_OPEN"
_TOOL_SEARCH_FRAGMENT = "TOOL_SEARCH"
_KNOWN_FRAGMENT_TYPES = frozenset(
    {
        _REQUEST_FRAGMENT,
        _RESPONSE_FRAGMENT,
        _THINK_FRAGMENT,
        _SEARCH_FRAGMENT,
        _TOOL_OPEN_FRAGMENT,
        _TOOL_SEARCH_FRAGMENT,
    }
)


@dataclass(frozen=True)
class DeepSeekConversionErrorDetail:
    """One path-aware error found while converting a DeepSeek export."""

    path: str
    message: str
    code: str = "invalid"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "code": self.code}


class DeepSeekExportConversionError(ValueError):
    """Raised when a DeepSeek export cannot be converted without losing semantics."""

    def __init__(self, errors: Sequence[DeepSeekConversionErrorDetail]):
        self.errors = tuple(errors)
        super().__init__("Invalid DeepSeek conversation export.")


@dataclass
class _ConversionContext:
    message_offset: str | None = None


@dataclass(frozen=True)
class _ConvertedMessage:
    external_message_id: str
    role: _DeepSeekRole
    content: str
    created_at: datetime
    metadata: dict[str, Any]


def convert_deepseek_export(
    payload_json: str | bytes,
    *,
    limit: int | None = None,
) -> NormalizedImportPayload:
    """Return Alpha's normalized import payload for one raw DeepSeek export."""

    if limit is not None and limit < 1:
        raise ValueError("limit must be greater than or equal to 1")

    errors: list[DeepSeekConversionErrorDetail] = []
    decoded = _parse_payload(payload_json, errors)
    if errors:
        _raise_errors(errors)

    if not isinstance(decoded, list):
        _raise_errors(
            [
                DeepSeekConversionErrorDetail(
                    "$",
                    "top-level DeepSeek export must be an array of conversations",
                    code="invalid_top_level",
                )
            ]
        )
    if not decoded:
        _raise_errors(
            [
                DeepSeekConversionErrorDetail(
                    "$",
                    "top-level DeepSeek export must include at least one conversation",
                    code="empty_export",
                )
            ]
        )

    context = _ConversionContext()
    conversations: list[dict[str, Any]] = []
    seen_conversation_ids: set[str] = set()
    raw_conversations = decoded
    if limit is not None:
        raw_conversations = decoded[:limit]
    for index, raw_conversation in enumerate(raw_conversations):
        conversation = _convert_conversation(
            raw_conversation,
            f"$[{index}]",
            context=context,
            seen_conversation_ids=seen_conversation_ids,
            errors=errors,
        )
        if conversation is not None:
            conversations.append(conversation)

    if errors:
        _raise_errors(errors)
    if context.message_offset is None:
        _raise_errors(
            [
                DeepSeekConversionErrorDetail(
                    "$",
                    "DeepSeek export did not contain any importable messages",
                    code="empty_export",
                )
            ]
        )

    return {
        "source_provider": "deepseek",
        "timezone": context.message_offset,
        "metadata": {
            "source_format": "deepseek_export",
            "converter": "alpha-agent",
        },
        "conversations": conversations,
    }


def _parse_payload(
    payload_json: str | bytes,
    errors: list[DeepSeekConversionErrorDetail],
) -> Any:
    try:
        if isinstance(payload_json, bytes):
            return json.loads(payload_json.decode("utf-8"))
        return json.loads(payload_json)
    except UnicodeDecodeError:
        errors.append(
            DeepSeekConversionErrorDetail(
                "$",
                "DeepSeek export must be UTF-8 encoded JSON",
                code="malformed_json",
            )
        )
        return None
    except json.JSONDecodeError as exc:
        errors.append(
            DeepSeekConversionErrorDetail(
                "$",
                f"DeepSeek export must be valid JSON: {exc.msg}",
                code="malformed_json",
            )
        )
        return None


def _convert_conversation(
    raw_conversation: Any,
    path: str,
    *,
    context: _ConversionContext,
    seen_conversation_ids: set[str],
    errors: list[DeepSeekConversionErrorDetail],
) -> dict[str, Any] | None:
    if not isinstance(raw_conversation, dict):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek conversation must be an object",
                code="invalid_conversation",
            )
        )
        return None

    external_id = _required_non_empty_string(
        raw_conversation,
        "id",
        f"{path}.id",
        errors,
    )
    if external_id:
        if external_id in seen_conversation_ids:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.id",
                    "DeepSeek conversation id must be unique within the export",
                    code="duplicate_conversation_id",
                )
            )
        seen_conversation_ids.add(external_id)

    title = _optional_string(raw_conversation, "title", f"{path}.title", errors)
    created_at = _required_timestamp(raw_conversation, "inserted_at", f"{path}.inserted_at", errors)
    updated_at = _required_timestamp(raw_conversation, "updated_at", f"{path}.updated_at", errors)

    raw_mapping = raw_conversation.get("mapping")
    if not isinstance(raw_mapping, dict):
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.mapping",
                "DeepSeek conversation mapping must be an object",
                code="invalid_mapping",
            )
        )
        return None

    message_node_ids = _linear_message_node_ids(raw_mapping, f"{path}.mapping", errors)
    converted_messages: list[_ConvertedMessage] = []
    for node_id in message_node_ids:
        raw_node = raw_mapping.get(node_id)
        messages = _convert_message_node(
            raw_node,
            f"{path}.mapping.{node_id}",
            node_id,
            context=context,
            errors=errors,
        )
        converted_messages.extend(messages)

    normalized_messages: list[dict[str, Any]] = []
    previous_created_at: datetime | None = None
    for message in converted_messages:
        created_at_for_import = message.created_at
        if previous_created_at is not None and created_at_for_import <= previous_created_at:
            created_at_for_import = previous_created_at + timedelta(microseconds=1)
        previous_created_at = created_at_for_import
        normalized: dict[str, Any] = {
            "external_message_id": message.external_message_id,
            "role": message.role,
            "content": message.content,
            "created_at": _format_timestamp(created_at_for_import),
        }
        if message.metadata:
            normalized["metadata"] = message.metadata
        normalized_messages.append(normalized)

    if not normalized_messages:
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.mapping",
                "DeepSeek conversation must include at least one importable message",
                code="empty_conversation",
            )
        )
        return None
    if not external_id or created_at is None or updated_at is None:
        return None

    conversation: dict[str, Any] = {
        "external_conversation_id": external_id,
        "created_at": _format_timestamp(created_at),
        "updated_at": _format_timestamp(updated_at),
        "messages": normalized_messages,
    }
    if title is not None:
        conversation["title"] = title
    return conversation


def _linear_message_node_ids(
    mapping: Mapping[str, Any],
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> list[str]:
    root = mapping.get("root")
    if root is None:
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.root",
                "DeepSeek mapping must include a root node",
                code="missing_root",
            )
        )
        return []

    node_ids: list[str] = []
    visited: set[str] = set()
    ignored_node_ids: set[str] = set()
    current_id = "root"
    while True:
        if current_id in visited:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{current_id}",
                    "DeepSeek mapping must not contain cycles",
                    code="cyclic_mapping",
                )
            )
            return node_ids
        visited.add(current_id)

        raw_node = mapping.get(current_id)
        if not isinstance(raw_node, dict):
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{current_id}",
                    "DeepSeek mapping node must be an object",
                    code="invalid_mapping_node",
                )
            )
            return node_ids
        _validate_node_id(raw_node, current_id, f"{path}.{current_id}", errors)
        if current_id == "root" and raw_node.get("message") is not None:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.root.message",
                    "DeepSeek root message must be null",
                    code="invalid_root_message",
                )
            )
            return node_ids

        children = _node_children(raw_node, f"{path}.{current_id}.children", errors)
        if children is None:
            return node_ids
        if len(children) > 1:
            children = _filter_server_busy_response_children(
                mapping,
                path,
                current_id,
                children,
                ignored_node_ids,
                errors,
            )
            if children is None:
                return node_ids
            if len(children) > 1:
                for skipped_child_id in children[:-1]:
                    ignored_node_ids.update(_collect_subtree_node_ids(mapping, skipped_child_id))
                children = [children[-1]]
            if not children:
                break
        if not children:
            break

        child_id = children[0]
        child_node = mapping.get(child_id)
        if not isinstance(child_node, dict):
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{current_id}.children[0]",
                    f"child node {child_id!r} is missing from mapping",
                    code="missing_child_node",
                )
            )
            return node_ids
        parent = child_node.get("parent")
        if parent != current_id:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{child_id}.parent",
                    f"child node parent must be {current_id!r}",
                    code="invalid_parent",
                )
            )
            return node_ids
        node_ids.append(child_id)
        current_id = child_id

    unreachable_ids = sorted(set(mapping) - visited - ignored_node_ids)
    if unreachable_ids:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek mapping contains nodes unreachable from root: "
                + ", ".join(unreachable_ids),
                code="unreachable_nodes",
            )
        )
    return node_ids


def _filter_server_busy_response_children(
    mapping: Mapping[str, Any],
    path: str,
    current_id: str,
    children: Sequence[str],
    ignored_node_ids: set[str],
    errors: list[DeepSeekConversionErrorDetail],
) -> list[str] | None:
    remaining_children: list[str] = []
    for index, child_id in enumerate(children):
        child_node = mapping.get(child_id)
        if not isinstance(child_node, dict):
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{current_id}.children[{index}]",
                    f"child node {child_id!r} is missing from mapping",
                    code="missing_child_node",
                )
            )
            return None
        parent = child_node.get("parent")
        if parent != current_id:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}.{child_id}.parent",
                    f"child node parent must be {current_id!r}",
                    code="invalid_parent",
                )
            )
            return None
        if _is_server_busy_response_node(child_node):
            ignored_node_ids.update(_collect_subtree_node_ids(mapping, child_id))
        else:
            remaining_children.append(child_id)
    return remaining_children


def _is_server_busy_response_node(raw_node: Mapping[str, Any]) -> bool:
    raw_message = raw_node.get("message")
    if not isinstance(raw_message, dict):
        return False
    raw_fragments = raw_message.get("fragments")
    if not isinstance(raw_fragments, list):
        return False
    response_parts: list[str] = []
    for raw_fragment in raw_fragments:
        if not isinstance(raw_fragment, dict):
            continue
        fragment_type = raw_fragment.get("type")
        if fragment_type == _REQUEST_FRAGMENT:
            return False
        if fragment_type != _RESPONSE_FRAGMENT:
            continue
        raw_content = raw_fragment.get("content")
        if isinstance(raw_content, str):
            response_parts.append(raw_content)
    if not response_parts:
        return False
    return "\n\n".join(response_parts).lstrip().startswith("服务器繁忙")


def _collect_subtree_node_ids(mapping: Mapping[str, Any], root_id: str) -> set[str]:
    collected: set[str] = set()
    pending = [root_id]
    while pending:
        node_id = pending.pop()
        if node_id in collected:
            continue
        collected.add(node_id)
        raw_node = mapping.get(node_id)
        if not isinstance(raw_node, dict):
            continue
        raw_children = raw_node.get("children")
        if not isinstance(raw_children, list):
            continue
        pending.extend(child_id for child_id in raw_children if isinstance(child_id, str))
    return collected


def _convert_message_node(
    raw_node: Any,
    path: str,
    node_id: str,
    *,
    context: _ConversionContext,
    errors: list[DeepSeekConversionErrorDetail],
) -> list[_ConvertedMessage]:
    if not isinstance(raw_node, dict):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek mapping node must be an object",
                code="invalid_mapping_node",
            )
        )
        return []

    raw_message = raw_node.get("message")
    if not isinstance(raw_message, dict):
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.message",
                "DeepSeek mapping message must be an object",
                code="invalid_message",
            )
        )
        return []

    model = _optional_non_empty_string(raw_message, "model", f"{path}.message.model", errors)
    created_at = _required_timestamp(
        raw_message,
        "inserted_at",
        f"{path}.message.inserted_at",
        errors,
    )
    if created_at is not None:
        _record_message_offset(created_at, f"{path}.message.inserted_at", context, errors)
    file_messages = _convert_file_messages(
        raw_message.get("files", []),
        f"{path}.message.files",
        node_id=node_id,
        model=model,
        created_at=created_at,
        errors=errors,
    )

    raw_fragments = raw_message.get("fragments")
    if not isinstance(raw_fragments, list):
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.message.fragments",
                "DeepSeek message fragments must be an array",
                code="invalid_fragments",
            )
        )
        return file_messages
    if not raw_fragments:
        return file_messages

    request_parts: list[str] = []
    response_parts: list[str] = []
    omitted_fragment_types: list[str] = []
    think_fragment_count = 0
    search_result_count = 0
    for index, raw_fragment in enumerate(raw_fragments):
        fragment_path = f"{path}.message.fragments[{index}]"
        if not isinstance(raw_fragment, dict):
            errors.append(
                DeepSeekConversionErrorDetail(
                    fragment_path,
                    "DeepSeek fragment must be an object",
                    code="invalid_fragment",
                )
            )
            continue
        fragment_type = raw_fragment.get("type")
        if not isinstance(fragment_type, str):
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{fragment_path}.type",
                    "DeepSeek fragment type must be a string",
                    code="invalid_fragment_type",
                )
            )
            continue
        if fragment_type not in _KNOWN_FRAGMENT_TYPES:
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{fragment_path}.type",
                    f"unsupported fragment type {fragment_type!r}",
                    code="unsupported_fragment_type",
                )
            )
            continue
        if fragment_type in {_REQUEST_FRAGMENT, _RESPONSE_FRAGMENT}:
            content = _fragment_content(raw_fragment, f"{fragment_path}.content", errors)
            if content is None:
                continue
            if fragment_type == _REQUEST_FRAGMENT:
                request_parts.append(content)
            else:
                response_parts.append(content)
            continue
        _append_unique(omitted_fragment_types, fragment_type)
        if fragment_type == _THINK_FRAGMENT:
            think_fragment_count += 1
        elif fragment_type == _SEARCH_FRAGMENT:
            search_result_count += _search_result_count(raw_fragment, fragment_path, errors)

    if request_parts and response_parts:
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.message.fragments",
                "DeepSeek message must not contain both REQUEST and RESPONSE fragments",
                code="mixed_message_role",
            )
        )
        return []
    if not request_parts and not response_parts:
        if file_messages:
            return file_messages
        if omitted_fragment_types:
            return []
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.message.fragments",
                "DeepSeek message must contain REQUEST or RESPONSE content",
                code="missing_message_content",
            )
        )
        return []
    if created_at is None:
        return file_messages

    role: _DeepSeekRole = "user" if request_parts else "assistant"
    parts = request_parts if request_parts else response_parts
    metadata = _message_metadata(
        model=model,
        omitted_fragment_types=omitted_fragment_types,
        think_fragment_count=think_fragment_count,
        search_result_count=search_result_count,
    )
    return [
        *file_messages,
        _ConvertedMessage(
            external_message_id=node_id,
            role=role,
            content="\n\n".join(parts),
            created_at=created_at,
            metadata=metadata,
        ),
    ]


def _convert_file_messages(
    raw_files: Any,
    path: str,
    *,
    node_id: str,
    model: str | None,
    created_at: datetime | None,
    errors: list[DeepSeekConversionErrorDetail],
) -> list[_ConvertedMessage]:
    if not isinstance(raw_files, list):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek message files must be an array",
                code="invalid_files",
            )
        )
        return []

    messages: list[_ConvertedMessage] = []
    for index, raw_file in enumerate(raw_files):
        file_path = f"{path}[{index}]"
        if not isinstance(raw_file, dict):
            errors.append(
                DeepSeekConversionErrorDetail(
                    file_path,
                    "DeepSeek file entry must be an object",
                    code="invalid_file",
                )
            )
            continue
        file_name = _optional_non_empty_string(
            raw_file,
            "file_name",
            f"{file_path}.file_name",
            errors,
        )
        raw_file_id = raw_file.get("id")
        file_id: str | None = None
        if raw_file_id is not None:
            if not isinstance(raw_file_id, str) or not raw_file_id.strip():
                errors.append(
                    DeepSeekConversionErrorDetail(
                        f"{file_path}.id",
                        "DeepSeek file id must be a non-empty string when present",
                        code="invalid_file_id",
                    )
                )
            else:
                file_id = raw_file_id.strip()
        if file_name is None or created_at is None:
            continue
        messages.append(
            _ConvertedMessage(
                external_message_id=f"{node_id}:file:{index}",
                role="user",
                content=f"file: {file_name}",
                created_at=created_at,
                metadata=_file_message_metadata(model=model, file_id=file_id),
            )
        )
    return messages


def _file_message_metadata(
    *,
    model: str | None,
    file_id: str | None,
) -> dict[str, Any]:
    deepseek_metadata: dict[str, Any] = {}
    if model is not None:
        deepseek_metadata["model"] = model
    if file_id is not None:
        deepseek_metadata["file_id"] = file_id
    if not deepseek_metadata:
        return {}
    return {"deepseek": deepseek_metadata}


def _message_metadata(
    *,
    model: str | None,
    omitted_fragment_types: list[str],
    think_fragment_count: int,
    search_result_count: int,
) -> dict[str, Any]:
    deepseek_metadata: dict[str, Any] = {}
    if model is not None:
        deepseek_metadata["model"] = model
    if omitted_fragment_types:
        deepseek_metadata["omitted_fragment_types"] = omitted_fragment_types
    if think_fragment_count:
        deepseek_metadata["think_fragment_count"] = think_fragment_count
    if search_result_count:
        deepseek_metadata["search_result_count"] = search_result_count
    if not deepseek_metadata:
        return {}
    return {"deepseek": deepseek_metadata}


def _validate_node_id(
    raw_node: Mapping[str, Any],
    expected_id: str,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> None:
    raw_id = raw_node.get("id")
    if raw_id != expected_id:
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.id",
                f"DeepSeek mapping node id must be {expected_id!r}",
                code="invalid_node_id",
            )
        )


def _node_children(
    raw_node: Mapping[str, Any],
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> list[str] | None:
    raw_children = raw_node.get("children")
    if not isinstance(raw_children, list):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek mapping node children must be an array",
                code="invalid_children",
            )
        )
        return None
    children: list[str] = []
    for index, raw_child in enumerate(raw_children):
        if not isinstance(raw_child, str) or not raw_child.strip():
            errors.append(
                DeepSeekConversionErrorDetail(
                    f"{path}[{index}]",
                    "DeepSeek child node id must be a non-empty string",
                    code="invalid_child_id",
                )
            )
            return None
        children.append(raw_child.strip())
    return children


def _fragment_content(
    fragment: Mapping[str, Any],
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> str | None:
    raw_content = fragment.get("content")
    if not isinstance(raw_content, str):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek REQUEST/RESPONSE fragment content must be a string",
                code="invalid_fragment_content",
            )
        )
        return None
    if not raw_content.strip():
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek REQUEST/RESPONSE fragment content must be non-empty",
                code="empty_fragment_content",
            )
        )
        return None
    return raw_content


def _search_result_count(
    fragment: Mapping[str, Any],
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> int:
    raw_results = fragment.get("results", [])
    if not isinstance(raw_results, list):
        errors.append(
            DeepSeekConversionErrorDetail(
                f"{path}.results",
                "DeepSeek SEARCH fragment results must be an array when present",
                code="invalid_search_results",
            )
        )
        return 0
    return len(raw_results)


def _required_non_empty_string(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                f"DeepSeek {key} must be a non-empty string",
                code="missing_required_string",
            )
        )
        return ""
    return raw.strip()


def _optional_non_empty_string(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> str | None:
    if key not in value:
        return None
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                f"DeepSeek {key} must be a non-empty string when present",
                code="invalid_string",
            )
        )
        return None
    return raw.strip()


def _optional_string(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> str | None:
    if key not in value:
        return None
    raw = value.get(key)
    if not isinstance(raw, str):
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                f"DeepSeek {key} must be a string when present",
                code="invalid_string",
            )
        )
        return None
    return raw


def _required_timestamp(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> datetime | None:
    if key not in value:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                f"DeepSeek {key} is required and must include a timezone",
                code="missing_timestamp",
            )
        )
        return None
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek timestamp must be a non-empty string with an explicit timezone",
                code="invalid_timestamp",
            )
        )
        return None
    raw_timestamp = raw.strip()
    parse_value = f"{raw_timestamp[:-1]}+00:00" if raw_timestamp.endswith("Z") else raw_timestamp
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek timestamp must be valid ISO-8601",
                code="invalid_timestamp",
            )
        )
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek timestamp must include an explicit timezone offset or Z",
                code="naive_timestamp",
            )
        )
        return None
    return parsed


def _record_message_offset(
    timestamp: datetime,
    path: str,
    context: _ConversionContext,
    errors: list[DeepSeekConversionErrorDetail],
) -> None:
    offset = _fixed_offset(timestamp, path, errors)
    if offset is None:
        return
    if context.message_offset is None:
        context.message_offset = offset
    elif context.message_offset != offset:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "message timestamp offsets must be consistent within one DeepSeek export",
                code="inconsistent_timezone_offset",
            )
        )


def _fixed_offset(
    timestamp: datetime,
    path: str,
    errors: list[DeepSeekConversionErrorDetail],
) -> str | None:
    offset = timestamp.utcoffset()
    if offset is None:
        return None
    total_seconds = int(offset.total_seconds())
    if total_seconds % 60 != 0:
        errors.append(
            DeepSeekConversionErrorDetail(
                path,
                "DeepSeek timestamp offset must be minute-aligned",
                code="invalid_timezone_offset",
            )
        )
        return None
    sign = "+" if total_seconds >= 0 else "-"
    absolute_seconds = abs(total_seconds)
    hours, remaining = divmod(absolute_seconds, 3600)
    minutes = remaining // 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.isoformat(timespec="microseconds")


def _append_unique(values: list[str], value: str) -> None:
    if value not in values:
        values.append(value)


def _raise_errors(errors: Sequence[DeepSeekConversionErrorDetail]) -> None:
    raise DeepSeekExportConversionError(errors)
