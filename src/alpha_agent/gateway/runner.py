"""Small gateway coordination helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from alpha_agent.gateway.adapters import PlatformAdapter
from alpha_agent.gateway.config import adapter_name
from alpha_agent.gateway.logging import GatewayLogContext, append_gateway_log
from alpha_agent.gateway.models import DeliveryResult, InboundMessage, OutboundMessage
from alpha_agent.gateway.session import GatewayDeduplicator, GatewaySessionStore, SessionMode

BUSY_SESSION_MESSAGE = (
    "This conversation already has an active Alpha turn. Please try again after it finishes."
)
RUNTIME_ERROR_MESSAGE = "Alpha failed while processing that message. Please try again."


@dataclass(frozen=True, slots=True)
class TurnStartResult:
    """Decision returned when a platform turn asks to enter a session."""

    accepted: bool
    bypassed: bool = False
    queued: bool = False
    reason: str | None = None


class GatewayDeliveryError(RuntimeError):
    """Raised when a platform adapter explicitly fails outbound delivery."""

    def __init__(self, adapter_name: str, result: DeliveryResult):
        self.adapter_name = adapter_name
        self.result = result
        detail = f": {result.error}" if result.error else ""
        super().__init__(f"Gateway outbound delivery failed for {adapter_name}{detail}")


class ActiveTurnGuard:
    """In-memory guard that allows at most one active non-control turn per session."""

    def __init__(self, bypass_commands: set[str] | None = None):
        self._active_session_ids: set[str] = set()
        self._bypass_commands = bypass_commands or {"/stop", "/reset", "/status"}
        self._lock = Lock()

    def begin(self, session_id: str, text: str, *, allow_queue: bool = False) -> TurnStartResult:
        """Try to mark a session active for processing."""

        if self._is_bypass_command(text):
            return TurnStartResult(accepted=True, bypassed=True)
        with self._lock:
            if session_id in self._active_session_ids:
                return TurnStartResult(
                    accepted=False,
                    queued=allow_queue,
                    reason="active_turn",
                )
            self._active_session_ids.add(session_id)
        return TurnStartResult(accepted=True)

    def complete(self, session_id: str) -> None:
        """Release a previously active session."""

        with self._lock:
            self._active_session_ids.discard(session_id)

    def is_active(self, session_id: str) -> bool:
        """Return whether a session currently has an active non-control turn."""

        with self._lock:
            return session_id in self._active_session_ids

    def _is_bypass_command(self, text: str) -> bool:
        first_token = text.strip().split(maxsplit=1)[0].lower() if text.strip() else ""
        return first_token in self._bypass_commands


class GatewayRuntimeBridge:
    """Coordinate platform inbound messages with the synchronous Alpha runtime."""

    def __init__(
        self,
        *,
        agent_manager: Any,
        session_store: GatewaySessionStore,
        deduplicator: GatewayDeduplicator,
        turn_guard: ActiveTurnGuard,
        session_mode: SessionMode,
        gateway_log_path: Path | None = None,
        error_log_path: Path | None = None,
        busy_message: str = BUSY_SESSION_MESSAGE,
        runtime_error_message: str = RUNTIME_ERROR_MESSAGE,
    ):
        self.agent_manager = agent_manager
        self.session_store = session_store
        self.deduplicator = deduplicator
        self.turn_guard = turn_guard
        self.session_mode = session_mode
        self.gateway_log_path = gateway_log_path
        self.error_log_path = error_log_path
        self.busy_message = busy_message
        self.runtime_error_message = runtime_error_message

    def connect(self, adapter: PlatformAdapter) -> None:
        """Connect adapter inbound delivery to this bridge."""

        adapter.connect(lambda message: self.handle_message(adapter, message))

    def handle_message(
        self,
        adapter: PlatformAdapter,
        message: InboundMessage,
    ) -> OutboundMessage | None:
        """Process one inbound gateway message and deliver any outbound response."""

        context = GatewayLogContext(
            platform=message.source.platform,
            chat_id=message.source.chat_id,
            user_id=message.source.user_id,
        )
        self._log_gateway(
            "gateway.message.received",
            "Inbound message received.",
            context=context,
        )

        dedup = self.deduplicator.check_and_record(message)
        if dedup.duplicate:
            cached_outbound = self.deduplicator.cached_outbound(dedup.dedup_key)
            if cached_outbound is not None:
                self._log_gateway(
                    "gateway.message.duplicate_retry",
                    "Duplicate inbound message is retrying cached outbound delivery.",
                    context=context,
                    metadata={"dedup_key": dedup.dedup_key},
                )
                self._send(adapter, message, cached_outbound)
                self.deduplicator.mark_outbound_delivered(dedup.dedup_key)
                return cached_outbound
            self._log_gateway(
                "gateway.message.duplicate",
                "Duplicate inbound message suppressed.",
                context=context,
                metadata={"dedup_key": dedup.dedup_key},
            )
            return None

        mapping = self.session_store.get_or_create(message.source, self.session_mode)
        context = GatewayLogContext(
            session_id=mapping.session_id,
            platform=message.source.platform,
            chat_id=message.source.chat_id,
            user_id=message.source.user_id,
        )
        turn = self.turn_guard.begin(mapping.session_id, message.text)
        if not turn.accepted:
            outbound = self._outbound(message, self.busy_message)
            self._log_gateway(
                "gateway.message.busy",
                "Inbound message rejected because the session has an active turn.",
                context=context,
                metadata={"reason": turn.reason, "queued": turn.queued},
            )
            self._cache_and_send(adapter, message, outbound, dedup.dedup_key)
            return outbound

        try:
            self._run_adapter_hook(adapter, "on_processing_start", message, context)
            agent = self._agent_for(mapping.session_id)
            result = agent.respond(
                message.text,
                session_id=mapping.session_id,
                source_metadata=gateway_source_metadata(message),
            )
            outbound = self._outbound(message, result.response)
        except Exception as exc:
            self._log_error(
                "gateway.runtime.error",
                "Runtime failed while processing gateway message.",
                context=context,
                metadata={"error_type": type(exc).__name__},
            )
            outbound = self._outbound(message, self.runtime_error_message)
        finally:
            try:
                self._run_adapter_hook(adapter, "on_processing_complete", message, context)
            finally:
                self.turn_guard.complete(mapping.session_id)

        self._cache_and_send(adapter, message, outbound, dedup.dedup_key)
        return outbound

    def _cache_and_send(
        self,
        adapter: PlatformAdapter,
        message: InboundMessage,
        outbound: OutboundMessage,
        dedup_key: str,
    ) -> DeliveryResult:
        self.deduplicator.cache_outbound(dedup_key, outbound)
        result = self._send(adapter, message, outbound)
        self.deduplicator.mark_outbound_delivered(dedup_key)
        return result

    def _run_adapter_hook(
        self,
        adapter: PlatformAdapter,
        hook_name: str,
        message: InboundMessage,
        context: GatewayLogContext,
    ) -> None:
        hook = getattr(adapter, hook_name, None)
        if hook is None:
            return
        try:
            hook(message.source)
        except Exception as exc:
            self._log_error(
                "gateway.adapter_hook.error",
                f"Adapter hook {hook_name} failed.",
                context=context,
                metadata={
                    "adapter": adapter_name(adapter),
                    "hook": hook_name,
                    "error_type": type(exc).__name__,
                },
            )

    def _send(
        self,
        adapter: PlatformAdapter,
        message: InboundMessage,
        outbound: OutboundMessage,
    ) -> DeliveryResult:
        try:
            result = adapter.send(message.source, outbound)
        except Exception as exc:
            result = DeliveryResult(success=False, error=str(exc), retryable=False)
            error = GatewayDeliveryError(adapter_name(adapter), result)
            self._log_delivery_error(message, error, result)
            raise error from exc
        if not result.success:
            error = GatewayDeliveryError(adapter_name(adapter), result)
            self._log_delivery_error(message, error, result)
            raise error
        self._log_gateway(
            "gateway.outbound.sent",
            "Outbound gateway message delivered.",
            context=GatewayLogContext(
                platform=message.source.platform,
                chat_id=message.source.chat_id,
                user_id=message.source.user_id,
            ),
            metadata={"platform_message_id": result.message_id},
        )
        return result

    def _log_delivery_error(
        self,
        message: InboundMessage,
        error: GatewayDeliveryError,
        result: DeliveryResult,
    ) -> None:
        self._log_error(
            "gateway.outbound.delivery_failed",
            str(error),
            context=GatewayLogContext(
                platform=message.source.platform,
                chat_id=message.source.chat_id,
                user_id=message.source.user_id,
            ),
            metadata={"retryable": result.retryable},
        )

    def _outbound(self, message: InboundMessage, text: str) -> OutboundMessage:
        return OutboundMessage(
            text=text,
            reply_to=message.platform_message_id or message.source.message_id,
        )

    def _agent_for(self, session_id: str) -> Any:
        return self.agent_manager.get_or_create(session_id)

    def _log_gateway(
        self,
        event: str,
        message: str,
        *,
        context: GatewayLogContext,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.gateway_log_path is None:
            return
        append_gateway_log(
            self.gateway_log_path,
            event=event,
            message=message,
            context=context,
            metadata=metadata,
        )

    def _log_error(
        self,
        event: str,
        message: str,
        *,
        context: GatewayLogContext,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.error_log_path is None:
            return
        append_gateway_log(
            self.error_log_path,
            event=event,
            message=message,
            level="error",
            context=context,
            metadata=metadata,
        )


def gateway_source_metadata(message: InboundMessage) -> dict[str, Any]:
    """Return transcript source metadata for one inbound gateway message."""

    source = message.source
    metadata: dict[str, Any] = {
        "channel": "gateway",
        "platform": source.platform,
        "chat_id": source.chat_id,
        "chat_type": source.chat_type,
        "user_id": source.user_id,
        "message_type": message.message_type,
    }
    if source.user_name is not None:
        metadata["user_name"] = source.user_name
    if source.thread_id is not None:
        metadata["thread_id"] = source.thread_id
    message_id = message.platform_message_id or source.message_id
    if message_id is not None:
        metadata["message_id"] = message_id
    if source.metadata:
        metadata["source"] = dict(source.metadata)
    if message.attachments:
        metadata["attachment_count"] = len(message.attachments)
    return metadata
