"""Daemon-side normalized external conversation import service."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Literal, cast

from alpha_agent.runtime.counterpart_router import DEFAULT_COUNTERPART_ID
from alpha_agent.state.models import ImportedConversationRecord, SessionMessageKind
from alpha_agent.state.store import StateStore
from alpha_agent.utils.ids import new_id
from alpha_agent.utils.time import utc_now_iso, validate_timezone

MAX_CONVERSATION_IMPORT_PAYLOAD_BYTES = 50 * 1024 * 1024

ImportRole = Literal["system", "user", "assistant", "tool"]

_ALLOWED_ROLES = frozenset({"system", "user", "assistant", "tool"})
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {"source_provider", "conversations", "metadata", "timezone"}
)
_ALLOWED_CONVERSATION_FIELDS = frozenset(
    {
        "external_conversation_id",
        "title",
        "created_at",
        "updated_at",
        "messages",
        "metadata",
    }
)
_REJECTED_MESSAGE_FIELDS = frozenset(
    {
        "reasoning_content",
        "attachment",
        "attachments",
        "file",
        "files",
        "image",
        "images",
        "audio",
        "video",
        "multimodal",
        "parts",
    }
)
_ROLE_TO_KIND: dict[ImportRole, SessionMessageKind] = {
    "system": "system_message",
    "user": "user_message",
    "assistant": "assistant_message",
    "tool": "tool_message",
}


@dataclass(frozen=True)
class ConversationImportValidationError:
    """One path-aware validation error for a normalized import payload."""

    path: str
    message: str
    code: str = "invalid"

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "message": self.message, "code": self.code}


class ConversationImportValidationFailed(ValueError):
    """Raised when a normalized conversation import cannot be accepted."""

    def __init__(self, errors: Sequence[ConversationImportValidationError]):
        self.errors = tuple(errors)
        super().__init__("Invalid conversation import payload.")


@dataclass(frozen=True)
class ConversationImportSummary:
    """Structured result for dry-run planning or a completed import."""

    source_provider: str
    dry_run: bool
    conversations_seen: int
    messages_seen: int
    conversations_created: int
    conversations_reused: int
    messages_inserted: int
    messages_deduped: int
    batch_id: str | None = None
    status: str = "completed"
    input_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "source_provider": self.source_provider,
            "dry_run": self.dry_run,
            "status": self.status,
            "input_name": self.input_name,
            "conversations_seen": self.conversations_seen,
            "messages_seen": self.messages_seen,
            "conversations_created": self.conversations_created,
            "conversations_reused": self.conversations_reused,
            "messages_inserted": self.messages_inserted,
            "messages_deduped": self.messages_deduped,
        }


@dataclass(frozen=True)
class _TimestampValue:
    original: str
    instant: datetime
    utc_iso: str
    fixed_offset: str


@dataclass(frozen=True)
class _ImportMessage:
    external_message_id: str
    role: ImportRole
    content: str
    created_at: _TimestampValue
    path: str
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ImportConversation:
    external_conversation_id: str
    title: str | None
    messages: list[_ImportMessage]
    session_timezone: str
    path: str
    external_created_at: _TimestampValue | None = None
    external_updated_at: _TimestampValue | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _ImportPayload:
    source_provider: str
    conversations: list[_ImportConversation]
    metadata: dict[str, Any]
    timezone: str | None = None


@dataclass(frozen=True)
class _PlannedConversation:
    conversation: _ImportConversation
    session_id: str
    imported_conversation: ImportedConversationRecord | None
    new_messages: list[_ImportMessage]
    deduped_messages: int

    @property
    def exists(self) -> bool:
        return self.imported_conversation is not None


@dataclass(frozen=True)
class _ImportPlan:
    payload: _ImportPayload
    conversations: list[_PlannedConversation]
    payload_digest: str
    input_name: str | None

    @property
    def conversations_seen(self) -> int:
        return len(self.conversations)

    @property
    def messages_seen(self) -> int:
        return sum(len(item.conversation.messages) for item in self.conversations)

    @property
    def conversations_created(self) -> int:
        return sum(1 for item in self.conversations if not item.exists)

    @property
    def conversations_reused(self) -> int:
        return sum(1 for item in self.conversations if item.exists)

    @property
    def messages_inserted(self) -> int:
        return sum(len(item.new_messages) for item in self.conversations)

    @property
    def messages_deduped(self) -> int:
        return sum(item.deduped_messages for item in self.conversations)


class ConversationImportService:
    """Parse, validate, plan, and persist normalized external conversations."""

    MAX_PAYLOAD_BYTES = MAX_CONVERSATION_IMPORT_PAYLOAD_BYTES

    def __init__(self, store: StateStore):
        self.store = store

    def import_payload(
        self,
        payload_json: str | bytes,
        *,
        input_name: str | None = None,
        dry_run: bool = False,
    ) -> ConversationImportSummary:
        """Import or dry-run one normalized conversation payload."""

        payload_bytes = _payload_bytes(payload_json)
        if len(payload_bytes) > self.MAX_PAYLOAD_BYTES:
            _raise_errors(
                [
                    ConversationImportValidationError(
                        "$",
                        "conversation import payload exceeds the first-version 50 MB limit",
                        code="payload_too_large",
                    )
                ]
            )

        parsed_json = _parse_json_payload(payload_bytes)
        payload = _validate_payload(parsed_json)
        plan = self._plan_import(
            payload,
            payload_digest=hashlib.sha256(payload_bytes).hexdigest(),
            input_name=_safe_input_name(input_name),
        )
        if dry_run:
            return _summary_from_plan(plan, dry_run=True, batch_id=None)
        return self._write_plan(plan)

    def _plan_import(
        self,
        payload: _ImportPayload,
        *,
        payload_digest: str,
        input_name: str | None,
    ) -> _ImportPlan:
        errors: list[ConversationImportValidationError] = []
        planned: list[_PlannedConversation] = []
        with self.store.connect() as conn:
            for conversation in payload.conversations:
                existing = self.store.get_imported_conversation(
                    payload.source_provider,
                    conversation.external_conversation_id,
                    conn=conn,
                )
                imported_ids = self.store.list_imported_external_message_ids(
                    payload.source_provider,
                    conversation.external_conversation_id,
                    conn=conn,
                )
                new_messages = [
                    message
                    for message in conversation.messages
                    if message.external_message_id not in imported_ids
                ]
                if existing is None:
                    _validate_strict_message_order(conversation.messages, errors)
                else:
                    _validate_strict_new_message_order(new_messages, errors)
                    if new_messages:
                        latest = self.store.latest_imported_message_timestamp(
                            payload.source_provider,
                            conversation.external_conversation_id,
                            conn=conn,
                        )
                        if latest is not None:
                            latest_instant = datetime.fromisoformat(latest).astimezone(UTC)
                            first_new = new_messages[0]
                            if first_new.created_at.instant <= latest_instant:
                                errors.append(
                                    ConversationImportValidationError(
                                        f"{first_new.path}.created_at",
                                        "new messages for an existing conversation must be "
                                        "strictly later than the latest imported message",
                                        code="middle_insertion",
                                    )
                                )
                planned.append(
                    _PlannedConversation(
                        conversation=conversation,
                        session_id=(
                            existing.session_id if existing is not None else new_id("session")
                        ),
                        imported_conversation=existing,
                        new_messages=new_messages,
                        deduped_messages=len(conversation.messages) - len(new_messages),
                    )
                )
        if errors:
            _raise_errors(errors)
        return _ImportPlan(
            payload=payload,
            conversations=planned,
            payload_digest=payload_digest,
            input_name=input_name,
        )

    def _write_plan(self, plan: _ImportPlan) -> ConversationImportSummary:
        batch_id = new_id("import_batch")
        imported_at = utc_now_iso()
        with self.store.immediate_transaction() as conn:
            self.store.create_import_batch(
                batch_id=batch_id,
                source_provider=plan.payload.source_provider,
                input_name=plan.input_name,
                payload_digest=plan.payload_digest,
                status="completed",
                conversations_seen=plan.conversations_seen,
                messages_seen=plan.messages_seen,
                conversations_created=plan.conversations_created,
                conversations_reused=plan.conversations_reused,
                messages_inserted=plan.messages_inserted,
                messages_deduped=plan.messages_deduped,
                metadata={"conversations": _conversation_count_metadata(plan)},
                created_at=imported_at,
                updated_at=imported_at,
                conn=conn,
            )
            for planned in plan.conversations:
                self._write_conversation(
                    plan.payload.source_provider,
                    planned,
                    batch_id,
                    imported_at,
                    conn,
                )
        return _summary_from_plan(plan, dry_run=False, batch_id=batch_id)

    def _write_conversation(
        self,
        source_provider: str,
        planned: _PlannedConversation,
        batch_id: str,
        imported_at: str,
        conn: Any,
    ) -> None:
        conversation = planned.conversation
        if not planned.exists:
            first_message = conversation.messages[0]
            last_message = conversation.messages[-1]
            self.store.create_session_record(
                planned.session_id,
                timezone=conversation.session_timezone,
                created_at=first_message.created_at.utc_iso,
                updated_at=last_message.created_at.utc_iso,
                conn=conn,
            )

        imported_conversation, _created = self.store.create_or_reuse_imported_conversation(
            source_provider=source_provider,
            external_conversation_id=conversation.external_conversation_id,
            session_id=planned.session_id,
            title=conversation.title,
            external_created_at=(
                conversation.external_created_at.utc_iso
                if conversation.external_created_at is not None
                else None
            ),
            external_updated_at=(
                conversation.external_updated_at.utc_iso
                if conversation.external_updated_at is not None
                else None
            ),
            first_import_batch_id=batch_id,
            latest_import_batch_id=batch_id,
            metadata=conversation.metadata,
            imported_at=imported_at,
            conn=conn,
        )
        self.store.create_session_counterpart(
            session_id=imported_conversation.session_id,
            counterpart_id=str(DEFAULT_COUNTERPART_ID),
            source_metadata={
                "source": "conversation_import",
                "source_provider": source_provider,
                "external_conversation_id": conversation.external_conversation_id,
            },
            created_at=imported_at,
            conn=conn,
        )

        for message in planned.new_messages:
            session_message = self.store.append_session_message(
                session_id=imported_conversation.session_id,
                kind=_ROLE_TO_KIND[message.role],
                llm_role=message.role,
                raw_content=message.content,
                tool_call_id=message.tool_call_id,
                tool_calls=message.tool_calls,
                source_metadata={
                    "source": "conversation_import",
                    "source_provider": source_provider,
                    "external_conversation_id": conversation.external_conversation_id,
                    "external_message_id": message.external_message_id,
                    "import_batch_id": batch_id,
                },
                metadata={
                    "external_created_at": message.created_at.original,
                    "external_metadata": message.metadata,
                },
                created_at=message.created_at.utc_iso,
                conn=conn,
            )
            self.store.create_imported_message(
                source_provider=source_provider,
                external_conversation_id=conversation.external_conversation_id,
                external_message_id=message.external_message_id,
                imported_conversation_id=imported_conversation.id,
                session_message_id=session_message.id,
                import_batch_id=batch_id,
                role=message.role,
                external_created_at=message.created_at.utc_iso,
                metadata={
                    "external_created_at": message.created_at.original,
                    "external_metadata": message.metadata,
                },
                imported_at=imported_at,
                conn=conn,
            )

        if planned.exists and planned.new_messages:
            self.store.update_session_record_history_times(
                imported_conversation.session_id,
                updated_at=planned.new_messages[-1].created_at.utc_iso,
                conn=conn,
            )


def _payload_bytes(payload_json: str | bytes) -> bytes:
    if isinstance(payload_json, bytes):
        return payload_json
    return payload_json.encode("utf-8")


def _parse_json_payload(payload_bytes: bytes) -> Any:
    try:
        return json.loads(payload_bytes.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConversationImportValidationFailed(
            [
                ConversationImportValidationError(
                    "$",
                    "payload must be UTF-8 encoded JSON",
                    code="malformed_json",
                )
            ]
        ) from exc
    except json.JSONDecodeError as exc:
        raise ConversationImportValidationFailed(
            [
                ConversationImportValidationError(
                    "$",
                    f"payload must be valid JSON: {exc.msg}",
                    code="malformed_json",
                )
            ]
        ) from exc


def _validate_payload(decoded: Any) -> _ImportPayload:
    errors: list[ConversationImportValidationError] = []
    if not isinstance(decoded, dict):
        _raise_errors(
            [
                ConversationImportValidationError(
                    "$",
                    "payload must be a JSON object",
                    code="invalid_top_level",
                )
            ]
        )

    source_provider = _required_non_empty_string(
        decoded,
        "source_provider",
        "source_provider",
        errors,
    )
    _reject_unknown_fields(
        decoded,
        allowed_fields=_ALLOWED_TOP_LEVEL_FIELDS,
        path_prefix="",
        errors=errors,
    )
    metadata = _optional_metadata(decoded, "metadata", "metadata", errors)
    timezone = _optional_timezone(decoded, "timezone", "timezone", errors)

    conversations_value = decoded.get("conversations")
    conversations: list[_ImportConversation] = []
    if not isinstance(conversations_value, list):
        errors.append(
            ConversationImportValidationError(
                "conversations",
                "conversations must be a non-empty array",
                code="invalid_conversations",
            )
        )
    elif not conversations_value:
        errors.append(
            ConversationImportValidationError(
                "conversations",
                "conversations must be a non-empty array",
                code="invalid_conversations",
            )
        )
    else:
        seen_conversation_ids: set[str] = set()
        for index, conversation_value in enumerate(conversations_value):
            path = f"conversations[{index}]"
            conversation = _validate_conversation(
                conversation_value,
                path,
                timezone=timezone,
                seen_conversation_ids=seen_conversation_ids,
                errors=errors,
            )
            if conversation is not None:
                conversations.append(conversation)

    if errors:
        _raise_errors(errors)
    return _ImportPayload(
        source_provider=source_provider,
        conversations=conversations,
        metadata=metadata,
        timezone=timezone,
    )


def _validate_conversation(
    value: Any,
    path: str,
    *,
    timezone: str | None,
    seen_conversation_ids: set[str],
    errors: list[ConversationImportValidationError],
) -> _ImportConversation | None:
    if not isinstance(value, dict):
        errors.append(
            ConversationImportValidationError(
                path,
                "conversation must be an object",
                code="invalid_conversation",
            )
        )
        return None

    _reject_unknown_fields(
        value,
        allowed_fields=_ALLOWED_CONVERSATION_FIELDS,
        path_prefix=path,
        errors=errors,
    )

    external_id = _required_non_empty_string(
        value,
        "external_conversation_id",
        f"{path}.external_conversation_id",
        errors,
    )
    if external_id:
        if external_id in seen_conversation_ids:
            errors.append(
                ConversationImportValidationError(
                    f"{path}.external_conversation_id",
                    "external_conversation_id must be unique within the payload",
                    code="duplicate_conversation_id",
                )
            )
        seen_conversation_ids.add(external_id)

    title = value.get("title")
    if title is not None and not isinstance(title, str):
        errors.append(
            ConversationImportValidationError(
                f"{path}.title",
                "title must be a string when present",
                code="invalid_title",
            )
        )
        title = None

    external_created_at = _optional_timestamp(
        value,
        "created_at",
        f"{path}.created_at",
        errors,
    )
    external_updated_at = _optional_timestamp(
        value,
        "updated_at",
        f"{path}.updated_at",
        errors,
    )
    metadata = _optional_metadata(value, "metadata", f"{path}.metadata", errors)

    messages_value = value.get("messages")
    messages: list[_ImportMessage] = []
    if not isinstance(messages_value, list):
        errors.append(
            ConversationImportValidationError(
                f"{path}.messages",
                "messages must be a non-empty array",
                code="invalid_messages",
            )
        )
    elif not messages_value:
        errors.append(
            ConversationImportValidationError(
                f"{path}.messages",
                "messages must be a non-empty array",
                code="invalid_messages",
            )
        )
    else:
        seen_message_ids: set[str] = set()
        pending_tool_calls: dict[str, str] = {}
        for index, message_value in enumerate(messages_value):
            message_path = f"{path}.messages[{index}]"
            message = _validate_message(
                message_value,
                message_path,
                seen_message_ids=seen_message_ids,
                pending_tool_calls=pending_tool_calls,
                errors=errors,
            )
            if message is None:
                continue
            messages.append(message)
        for tool_call_id, tool_call_path in pending_tool_calls.items():
            errors.append(
                ConversationImportValidationError(
                    tool_call_path,
                    f"tool_call {tool_call_id!r} must be followed by a matching tool message",
                    code="unmatched_tool_call",
                )
            )

    if not external_id or not messages:
        return None
    session_timezone = timezone or messages[0].created_at.fixed_offset
    return _ImportConversation(
        external_conversation_id=external_id,
        title=title,
        messages=messages,
        session_timezone=session_timezone,
        external_created_at=external_created_at,
        external_updated_at=external_updated_at,
        metadata=metadata,
        path=path,
    )


def _validate_message(
    value: Any,
    path: str,
    *,
    seen_message_ids: set[str],
    pending_tool_calls: dict[str, str],
    errors: list[ConversationImportValidationError],
) -> _ImportMessage | None:
    if not isinstance(value, dict):
        errors.append(
            ConversationImportValidationError(
                path,
                "message must be an object",
                code="invalid_message",
            )
        )
        return None

    for field_name in sorted(_REJECTED_MESSAGE_FIELDS):
        if field_name in value:
            errors.append(
                ConversationImportValidationError(
                    f"{path}.{field_name}",
                    f"{field_name} is not supported by the first-version import contract",
                    code="unsupported_field",
                )
            )

    external_message_id = _required_non_empty_string(
        value,
        "external_message_id",
        f"{path}.external_message_id",
        errors,
    )
    if external_message_id:
        if external_message_id in seen_message_ids:
            errors.append(
                ConversationImportValidationError(
                    f"{path}.external_message_id",
                    "external_message_id must be unique within the conversation",
                    code="duplicate_message_id",
                )
            )
        seen_message_ids.add(external_message_id)

    role_value = value.get("role")
    if not isinstance(role_value, str) or role_value not in _ALLOWED_ROLES:
        errors.append(
            ConversationImportValidationError(
                f"{path}.role",
                "role must be one of system, user, assistant, tool",
                code="invalid_role",
            )
        )
        role: ImportRole | None = None
    else:
        role = cast(ImportRole, role_value)

    timestamp = _required_timestamp(value, "created_at", f"{path}.created_at", errors)
    metadata = _optional_metadata(value, "metadata", f"{path}.metadata", errors)
    tool_calls = _validate_tool_calls(value, path, role, errors)
    tool_call_id = _validate_tool_call_id(value, path, role, pending_tool_calls, errors)
    content = _validate_message_content(value, path, role, bool(tool_calls), errors)

    if role == "assistant":
        for index, tool_call in enumerate(tool_calls):
            tool_id = str(tool_call["id"])
            if tool_id in pending_tool_calls:
                errors.append(
                    ConversationImportValidationError(
                        f"{path}.tool_calls[{index}].id",
                        "tool call ids must be unique until matched by a tool message",
                        code="duplicate_tool_call_id",
                    )
                )
            pending_tool_calls[tool_id] = f"{path}.tool_calls[{index}].id"
    elif role == "tool" and tool_call_id is not None:
        if tool_call_id not in pending_tool_calls:
            errors.append(
                ConversationImportValidationError(
                    f"{path}.tool_call_id",
                    "tool message must reference a matching assistant tool call",
                    code="unmatched_tool_result",
                )
            )
        else:
            pending_tool_calls.pop(tool_call_id)

    if not external_message_id or role is None or timestamp is None or content is None:
        return None
    return _ImportMessage(
        external_message_id=external_message_id,
        role=role,
        content=content,
        created_at=timestamp,
        tool_call_id=tool_call_id,
        tool_calls=tool_calls,
        metadata=metadata,
        path=path,
    )


def _reject_unknown_fields(
    value: Mapping[str, Any],
    *,
    allowed_fields: frozenset[str],
    path_prefix: str,
    errors: list[ConversationImportValidationError],
) -> None:
    for field_name in sorted(set(value) - allowed_fields):
        path = field_name if not path_prefix else f"{path_prefix}.{field_name}"
        errors.append(
            ConversationImportValidationError(
                path,
                f"{field_name} is not supported by the first-version import contract",
                code="unsupported_field",
            )
        )


def _validate_strict_message_order(
    messages: Sequence[_ImportMessage],
    errors: list[ConversationImportValidationError],
) -> None:
    previous: _ImportMessage | None = None
    for message in messages:
        if previous is not None and message.created_at.instant <= previous.created_at.instant:
            errors.append(
                ConversationImportValidationError(
                    f"{message.path}.created_at",
                    "message created_at must be strictly later than the previous message",
                    code="non_increasing_timestamp",
                )
            )
        previous = message


def _validate_strict_new_message_order(
    messages: Sequence[_ImportMessage],
    errors: list[ConversationImportValidationError],
) -> None:
    previous: _ImportMessage | None = None
    for message in messages:
        if previous is not None and message.created_at.instant <= previous.created_at.instant:
            errors.append(
                ConversationImportValidationError(
                    f"{message.path}.created_at",
                    "new message created_at must be strictly later than the previous new message",
                    code="non_increasing_timestamp",
                )
            )
        previous = message


def _validate_tool_calls(
    value: Mapping[str, Any],
    path: str,
    role: ImportRole | None,
    errors: list[ConversationImportValidationError],
) -> list[dict[str, Any]]:
    if "tool_calls" not in value:
        return []
    if role != "assistant":
        errors.append(
            ConversationImportValidationError(
                f"{path}.tool_calls",
                "tool_calls are allowed only on assistant messages",
                code="invalid_tool_calls",
            )
        )
        return []
    raw_tool_calls = value.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        errors.append(
            ConversationImportValidationError(
                f"{path}.tool_calls",
                "tool_calls must be an array",
                code="invalid_tool_calls",
            )
        )
        return []
    tool_calls: list[dict[str, Any]] = []
    for index, raw_tool_call in enumerate(raw_tool_calls):
        tool_call_path = f"{path}.tool_calls[{index}]"
        if not isinstance(raw_tool_call, dict):
            errors.append(
                ConversationImportValidationError(
                    tool_call_path,
                    "tool call must be an object",
                    code="invalid_tool_call",
                )
            )
            continue
        has_error = False
        tool_call_id = raw_tool_call.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id.strip():
            errors.append(
                ConversationImportValidationError(
                    f"{tool_call_path}.id",
                    "tool call id must be a non-empty string",
                    code="invalid_tool_call_id",
                )
            )
            has_error = True
        tool_call_type = raw_tool_call.get("type")
        if tool_call_type != "function":
            errors.append(
                ConversationImportValidationError(
                    f"{tool_call_path}.type",
                    "tool call type must be function",
                    code="invalid_tool_call_type",
                )
            )
            has_error = True
        function = raw_tool_call.get("function")
        if not isinstance(function, dict):
            errors.append(
                ConversationImportValidationError(
                    f"{tool_call_path}.function",
                    "tool call function must be an object",
                    code="invalid_tool_call_function",
                )
            )
            has_error = True
        else:
            function_name = function.get("name")
            if not isinstance(function_name, str) or not function_name.strip():
                errors.append(
                    ConversationImportValidationError(
                        f"{tool_call_path}.function.name",
                        "tool call function.name must be a non-empty string",
                        code="invalid_tool_call_function_name",
                    )
                )
                has_error = True
            function_arguments = function.get("arguments")
            if not isinstance(function_arguments, str):
                errors.append(
                    ConversationImportValidationError(
                        f"{tool_call_path}.function.arguments",
                        "tool call function.arguments must be a JSON string",
                        code="invalid_tool_call_arguments",
                    )
                )
                has_error = True
            else:
                try:
                    json.loads(function_arguments)
                except json.JSONDecodeError as exc:
                    errors.append(
                        ConversationImportValidationError(
                            f"{tool_call_path}.function.arguments",
                            f"tool call function.arguments must contain valid JSON: {exc.msg}",
                            code="invalid_tool_call_arguments_json",
                        )
                    )
                    has_error = True
        if has_error:
            continue
        tool_calls.append(dict(raw_tool_call))
    return tool_calls


def _validate_tool_call_id(
    value: Mapping[str, Any],
    path: str,
    role: ImportRole | None,
    pending_tool_calls: Mapping[str, str],
    errors: list[ConversationImportValidationError],
) -> str | None:
    if "tool_call_id" not in value:
        if role == "tool":
            errors.append(
                ConversationImportValidationError(
                    f"{path}.tool_call_id",
                    "tool messages must include a non-empty tool_call_id",
                    code="missing_tool_call_id",
                )
            )
        return None
    raw_tool_call_id = value.get("tool_call_id")
    if role != "tool":
        errors.append(
            ConversationImportValidationError(
                f"{path}.tool_call_id",
                "tool_call_id is allowed only on tool messages",
                code="invalid_tool_call_id",
            )
        )
        return None
    if not isinstance(raw_tool_call_id, str) or not raw_tool_call_id.strip():
        errors.append(
            ConversationImportValidationError(
                f"{path}.tool_call_id",
                "tool_call_id must be a non-empty string",
                code="invalid_tool_call_id",
            )
        )
        return None
    return raw_tool_call_id.strip()


def _validate_message_content(
    value: Mapping[str, Any],
    path: str,
    role: ImportRole | None,
    has_tool_calls: bool,
    errors: list[ConversationImportValidationError],
) -> str | None:
    if role == "assistant" and has_tool_calls and "content" not in value:
        return ""
    raw_content = value.get("content")
    if not isinstance(raw_content, str):
        errors.append(
            ConversationImportValidationError(
                f"{path}.content",
                "content must be a string",
                code="invalid_content",
            )
        )
        return None
    if role == "assistant" and has_tool_calls:
        return raw_content
    if not raw_content.strip():
        errors.append(
            ConversationImportValidationError(
                f"{path}.content",
                "content must be a non-empty string",
                code="empty_content",
            )
        )
        return None
    return raw_content


def _required_non_empty_string(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[ConversationImportValidationError],
) -> str:
    raw = value.get(key)
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            ConversationImportValidationError(
                path,
                f"{key} must be a non-empty string",
                code="missing_required_string",
            )
        )
        return ""
    return raw.strip()


def _optional_metadata(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[ConversationImportValidationError],
) -> dict[str, Any]:
    if key not in value:
        return {}
    raw = value.get(key)
    if not isinstance(raw, dict):
        errors.append(
            ConversationImportValidationError(
                path,
                f"{key} must be an object when present",
                code="invalid_metadata",
            )
        )
        return {}
    return dict(raw)


def _optional_timezone(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[ConversationImportValidationError],
) -> str | None:
    if key not in value:
        return None
    raw = value.get(key)
    if not isinstance(raw, str):
        errors.append(
            ConversationImportValidationError(
                path,
                "timezone must be an IANA timezone or fixed UTC offset string",
                code="invalid_timezone",
            )
        )
        return None
    try:
        return validate_timezone(raw)
    except ValueError as exc:
        errors.append(
            ConversationImportValidationError(
                path,
                str(exc),
                code="invalid_timezone",
            )
        )
        return None


def _optional_timestamp(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[ConversationImportValidationError],
) -> _TimestampValue | None:
    if key not in value:
        return None
    return _timestamp_value(value.get(key), path, errors)


def _required_timestamp(
    value: Mapping[str, Any],
    key: str,
    path: str,
    errors: list[ConversationImportValidationError],
) -> _TimestampValue | None:
    if key not in value:
        errors.append(
            ConversationImportValidationError(
                path,
                f"{key} is required and must include a timezone",
                code="missing_timestamp",
            )
        )
        return None
    return _timestamp_value(value.get(key), path, errors)


def _timestamp_value(
    raw: Any,
    path: str,
    errors: list[ConversationImportValidationError],
) -> _TimestampValue | None:
    if not isinstance(raw, str) or not raw.strip():
        errors.append(
            ConversationImportValidationError(
                path,
                "timestamp must be a non-empty string with an explicit timezone",
                code="invalid_timestamp",
            )
        )
        return None
    original = raw.strip()
    parse_value = f"{original[:-1]}+00:00" if original.endswith("Z") else original
    try:
        parsed = datetime.fromisoformat(parse_value)
    except ValueError:
        errors.append(
            ConversationImportValidationError(
                path,
                "timestamp must be a valid ISO-8601 datetime",
                code="invalid_timestamp",
            )
        )
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        errors.append(
            ConversationImportValidationError(
                path,
                "timestamp must include an explicit timezone offset or Z",
                code="naive_timestamp",
            )
        )
        return None
    instant = parsed.astimezone(UTC)
    return _TimestampValue(
        original=original,
        instant=instant,
        utc_iso=instant.isoformat(),
        fixed_offset=_format_fixed_offset(parsed.utcoffset() or timedelta(0)),
    )


def _format_fixed_offset(offset: timedelta) -> str:
    total_minutes = round(offset.total_seconds() / 60)
    sign = "+" if total_minutes >= 0 else "-"
    absolute_minutes = abs(total_minutes)
    hours, minutes = divmod(absolute_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _summary_from_plan(
    plan: _ImportPlan,
    *,
    dry_run: bool,
    batch_id: str | None,
) -> ConversationImportSummary:
    return ConversationImportSummary(
        batch_id=batch_id,
        source_provider=plan.payload.source_provider,
        dry_run=dry_run,
        input_name=plan.input_name,
        conversations_seen=plan.conversations_seen,
        messages_seen=plan.messages_seen,
        conversations_created=plan.conversations_created,
        conversations_reused=plan.conversations_reused,
        messages_inserted=plan.messages_inserted,
        messages_deduped=plan.messages_deduped,
    )


def _conversation_count_metadata(plan: _ImportPlan) -> list[dict[str, Any]]:
    return [
        {
            "external_conversation_id": item.conversation.external_conversation_id,
            "title": item.conversation.title,
            "messages_inserted": len(item.new_messages),
            "messages_deduped": item.deduped_messages,
            "session_reused": item.exists,
        }
        for item in plan.conversations
    ]


def _safe_input_name(input_name: str | None) -> str | None:
    if input_name is None:
        return None
    stripped = input_name.strip()
    if not stripped:
        return None
    windows_name = PureWindowsPath(stripped).name
    return PurePosixPath(windows_name).name


def _raise_errors(errors: Sequence[ConversationImportValidationError]) -> None:
    raise ConversationImportValidationFailed(errors)
