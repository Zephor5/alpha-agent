"""IPC request and response models for the Alpha daemon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

RequestType = Literal["ask", "chat_turn", "consolidate_memory", "status", "stop"]
StopPolicyValue = Literal["graceful", "immediate"]
STOP_POLICIES: frozenset[str] = frozenset(("graceful", "immediate"))


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


def parse_request(payload: Any) -> DaemonRequest:
    """Validate an untrusted JSON payload at the IPC boundary."""

    if not isinstance(payload, dict):
        raise DaemonProtocolError("INVALID_REQUEST", "Request must be a JSON object.")
    request_type = payload.get("type")
    if not isinstance(request_type, str):
        raise DaemonProtocolError("INVALID_REQUEST", "Request type is required.")
    if request_type not in {"ask", "chat_turn", "consolidate_memory", "status", "stop"}:
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

    return DaemonRequest(
        type=request_type,  # type: ignore[arg-type]
        message=message,
        session_id=session_id,
        source_metadata=dict(source_metadata) if source_metadata is not None else None,
        stop_policy=stop_policy,  # type: ignore[arg-type]
    )


def ok_response(**fields: Any) -> dict[str, Any]:
    """Build a successful IPC response."""

    return {"ok": True, **fields}


def error_response(code: str, message: str) -> dict[str, Any]:
    """Build a stable daemon IPC error response."""

    return {"ok": False, "error": {"code": code, "message": message}}
