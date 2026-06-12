"""IPC request and response models for the Alpha daemon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

RequestType = Literal[
    "ask",
    "chat_turn",
    "status",
    "stop",
    "conversation_import",
    "conversation_import_status",
]
StopPolicyValue = Literal["graceful", "immediate"]
STOP_POLICIES: frozenset[str] = frozenset(("graceful", "immediate"))
REQUEST_TYPES: frozenset[str] = frozenset(
    (
        "ask",
        "chat_turn",
        "status",
        "stop",
        "conversation_import",
        "conversation_import_status",
    )
)


class DaemonProtocolError(ValueError):
    """Raised when an IPC request fails boundary validation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class DaemonRequest:
    """Validated daemon IPC request."""

    type: RequestType
    message: str | None = None
    session_id: str | None = None
    source_metadata: dict[str, Any] | None = None
    stop_policy: StopPolicyValue = "graceful"
    input_name: str | None = None
    payload_json: str | None = None
    dry_run: bool = False
    batch_id: str | None = None
    verbose: bool = False


def parse_request(payload: Any) -> DaemonRequest:
    """Validate an untrusted JSON payload at the IPC boundary."""

    if not isinstance(payload, dict):
        raise DaemonProtocolError("INVALID_REQUEST", "Request must be a JSON object.")
    request_type = payload.get("type")
    if not isinstance(request_type, str):
        raise DaemonProtocolError("INVALID_REQUEST", "Request type is required.")
    if request_type not in REQUEST_TYPES:
        raise DaemonProtocolError(
            "UNKNOWN_REQUEST_TYPE",
            f"Unknown daemon request type: {request_type}",
        )

    message = payload.get("message")
    if request_type in {"ask", "chat_turn"}:
        if not isinstance(message, str) or not message.strip():
            raise DaemonProtocolError("INVALID_REQUEST", "Message is required.")
    elif message is not None and not isinstance(message, str):
        raise DaemonProtocolError("INVALID_REQUEST", "Message must be a string.")

    session_id = payload.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise DaemonProtocolError("INVALID_REQUEST", "session_id must be a string or null.")

    source_metadata = payload.get("source_metadata")
    if source_metadata is not None and not isinstance(source_metadata, dict):
        raise DaemonProtocolError(
            "INVALID_REQUEST",
            "source_metadata must be an object or null.",
        )

    stop_policy = payload.get("policy", "graceful")
    if request_type == "stop":
        if not isinstance(stop_policy, str) or stop_policy not in STOP_POLICIES:
            raise DaemonProtocolError(
                "INVALID_REQUEST",
                "Stop policy must be one of: graceful, immediate.",
            )
    elif "policy" in payload:
        raise DaemonProtocolError(
            "INVALID_REQUEST",
            "policy is only supported for stop requests.",
        )

    input_name = payload.get("input_name")
    if request_type == "conversation_import":
        if input_name is not None and not isinstance(input_name, str):
            raise DaemonProtocolError(
                "INVALID_REQUEST",
                "input_name must be a string or null.",
            )
        payload_json = payload.get("payload_json")
        if not isinstance(payload_json, str):
            raise DaemonProtocolError("INVALID_REQUEST", "payload_json must be a string.")
        dry_run = payload.get("dry_run", False)
        if not isinstance(dry_run, bool):
            raise DaemonProtocolError("INVALID_REQUEST", "dry_run must be a boolean.")
    else:
        payload_json = None
        dry_run = False

    batch_id = payload.get("batch_id")
    verbose = payload.get("verbose", False)
    if request_type == "conversation_import_status":
        if not isinstance(batch_id, str) or not batch_id.strip():
            raise DaemonProtocolError(
                "INVALID_REQUEST",
                "batch_id must be a non-empty string.",
            )
        if not isinstance(verbose, bool):
            raise DaemonProtocolError("INVALID_REQUEST", "verbose must be a boolean.")
    else:
        batch_id = None
        verbose = False

    return DaemonRequest(
        type=cast(RequestType, request_type),
        message=message,
        session_id=session_id,
        source_metadata=dict(source_metadata) if source_metadata is not None else None,
        stop_policy=cast(StopPolicyValue, stop_policy),
        input_name=input_name if isinstance(input_name, str) else None,
        payload_json=payload_json,
        dry_run=dry_run,
        batch_id=batch_id.strip() if isinstance(batch_id, str) else None,
        verbose=verbose,
    )


def ok_response(**fields: Any) -> dict[str, Any]:
    """Build a successful IPC response."""

    return {"ok": True, **fields}


def error_response(
    code: str,
    message: str,
    *,
    details: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a stable daemon IPC error response."""

    error: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        error["details"] = details
    return {"ok": False, "error": error}
