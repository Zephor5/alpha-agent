"""Daemon lifecycle and request handling."""

from __future__ import annotations

import signal
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from threading import Event, Lock
from types import FrameType
from typing import Any

from alpha_agent.cognition.coordinator import LoopCoordinator
from alpha_agent.cognition.loops import (
    BackgroundCognitionService,
    DirectCompactExtractionService,
    RealtimeFeedbackAttributionService,
)
from alpha_agent.cognition.models.subject import SUBJECT_SELF
from alpha_agent.cognition.processing_ledger import BackgroundSourceRef, BackgroundStage
from alpha_agent.cognition.state_service import CognitionStateStore
from alpha_agent.config import AlphaConfig
from alpha_agent.daemon.conversation_import import (
    ConversationImportService,
    ConversationImportValidationFailed,
)
from alpha_agent.daemon.manager import (
    AgentFactory,
    AgentManager,
    build_provider,
    initialize_store,
)
from alpha_agent.daemon.models import (
    DaemonProtocolError,
    DaemonRequest,
    StopPolicyValue,
    error_response,
    ok_response,
    parse_request,
)
from alpha_agent.daemon.server import JsonLineDaemonServer
from alpha_agent.daemon.status import (
    DaemonRuntimeConfig,
    DaemonRuntimeLock,
    DaemonStatus,
    cleanup_runtime_files,
    daemon_runtime_config,
    error_status,
    idle_status,
    is_pid_running,
    read_daemon_status,
    running_status,
    write_daemon_status,
)
from alpha_agent.gateway.config import (
    adapter_name,
    configured_adapters,
    ensure_gateway_runtime_files,
    gateway_runtime_config,
)
from alpha_agent.gateway.logging import append_gateway_log
from alpha_agent.gateway.runner import ActiveTurnGuard, GatewayRuntimeBridge
from alpha_agent.gateway.session import GatewayDeduplicator, GatewaySessionStore, SessionMode
from alpha_agent.llm.tracing import LLMTraceLogger
from alpha_agent.runtime.session import new_session_id
from alpha_agent.state.models import ImportBatchRecord, ImportStatusSummary
from alpha_agent.state.store import StateStore
from alpha_agent.tools.default import build_tool_registry

LOCAL_BUSY_MESSAGE = "This session already has an active Alpha turn."
IMPORT_SESSION_CHAT_MESSAGE = (
    "Import sessions are hidden source material and cannot be continued with ordinary chat."
)
_IMPORT_EXTRACTION_STATUS_KEYS = (
    "extraction_pending",
    "extraction_claimed",
    "extraction_processed",
    "extraction_failed",
    "extraction_skipped",
)
_BACKGROUND_SERVICE_SHUTDOWN_TIMEOUT_SECONDS = 61.0


class StopPolicy(StrEnum):
    """Daemon shutdown policy."""

    GRACEFUL = "graceful"
    IMMEDIATE = "immediate"


class DaemonAlreadyRunningError(RuntimeError):
    """Raised when a live daemon already owns the configured runtime paths."""


class AlphaDaemon:
    """Single long-running owner for local and gateway runtime turns."""

    def __init__(
        self,
        config: AlphaConfig,
        *,
        store: StateStore | None = None,
        agent_manager: AgentManager | None = None,
        turn_guard: ActiveTurnGuard | None = None,
        runtime: DaemonRuntimeConfig | None = None,
    ):
        self.config = config
        self.runtime = runtime or daemon_runtime_config(config)
        self.store = store or initialize_store(config)
        self.conversation_import_service = ConversationImportService(self.store)
        self._conversation_import_lock = Lock()
        self.loop_coordinator = LoopCoordinator(SUBJECT_SELF)
        background_provider = build_provider(config)
        background_tools = build_tool_registry(config).to_llm_tool_definitions()
        llm_trace_logger = LLMTraceLogger.from_config(config)
        self.direct_compact_extraction = DirectCompactExtractionService(
            store=self.store,
            llm_provider=background_provider,
            tools=background_tools,
            enabled=config.cognition_background.enabled,
            llm_trace_logger=llm_trace_logger,
        )
        self.feedback_attribution = RealtimeFeedbackAttributionService(
            store=self.store,
            llm_provider=background_provider,
            enabled=config.cognition_background.enabled,
            llm_trace_logger=llm_trace_logger,
        )
        self.agent_manager = agent_manager or AgentManager(
            AgentFactory(
                config,
                self.store,
                coordinator=self.loop_coordinator,
                compact_extraction_submitter=self.direct_compact_extraction.submit,
                feedback_attribution_submitter=self.feedback_attribution.submit,
                llm_trace_logger=llm_trace_logger,
            )
        )
        self.turn_guard = turn_guard or ActiveTurnGuard(bypass_commands=set())
        self.background_service = BackgroundCognitionService(
            store=self.store,
            config=config.cognition_background,
            coordinator=self.loop_coordinator,
            llm_provider=background_provider,
            tools=background_tools,
            llm_trace_logger=llm_trace_logger,
        )
        self._server: JsonLineDaemonServer | None = None
        self._stop_requested = Event()
        self._stop_policy: StopPolicy | None = None
        self._state = "running"
        self._status_message = "Daemon is running."
        self._adapter_names: tuple[str, ...] = ()

    def handle_payload(self, payload: Any) -> dict[str, Any]:
        """Handle one untrusted IPC payload."""

        try:
            request = parse_request(payload)
        except DaemonProtocolError as exc:
            return error_response(exc.code, exc.message)

        if request.type == "status":
            return ok_response(status=self.status().to_json())
        if request.type == "stop":
            self.stop(StopPolicy(request.stop_policy))
            return ok_response(status=self.status().to_json())
        if request.type == "conversation_import":
            return self._handle_conversation_import(request)
        if request.type == "conversation_import_status":
            return self._handle_conversation_import_status(request)
        if request.type in {"ask", "chat_turn"}:
            return self._handle_turn(request)
        return error_response("UNKNOWN_REQUEST_TYPE", f"Unknown request type: {request.type}")

    def run(self) -> None:
        """Run IPC server and configured gateway adapters until stopped."""

        self._assert_single_owner()
        runtime_lock = self._acquire_runtime_lock()
        connected_adapters = []
        restore_signal_handlers: Callable[[], None] | None = None
        failed = False
        restore_error: Exception | None = None
        try:
            restore_signal_handlers = self._install_signal_handlers()
            gateway_runtime = gateway_runtime_config(self.config)
            ensure_gateway_runtime_files(gateway_runtime)
            adapters = configured_adapters()
            self._adapter_names = tuple(adapter_name(adapter) for adapter in adapters)
            write_daemon_status(
                self.runtime.status_path,
                self._set_status(state="starting", message="Daemon is starting."),
            )
            self._server = JsonLineDaemonServer(self.runtime.socket_path, self.handle_payload)
            bridge = GatewayRuntimeBridge(
                agent_manager=self.agent_manager,
                session_store=GatewaySessionStore(self.store),
                deduplicator=GatewayDeduplicator(self.store),
                turn_guard=self.turn_guard,
                session_mode=SessionMode.GROUP_PER_USER,
                gateway_log_path=gateway_runtime.log_paths["gateway.log"],
                error_log_path=gateway_runtime.log_paths["errors.log"],
            )
            for adapter in adapters:
                connected_adapters.append(adapter)
                bridge.connect(adapter)
            self.background_service.start()
            write_daemon_status(
                self.runtime.status_path,
                self._set_status(state="running", message="Daemon is running."),
            )
            self._server.serve_forever()
        except Exception as exc:
            failed = True
            write_daemon_status(
                self.runtime.status_path,
                error_status(
                    config=self.config,
                    runtime=self.runtime,
                    adapter_names=self._adapter_names,
                    message=f"Daemon stopped after error: {exc}",
                    background_status=self.background_service.status(),
                ),
            )
            raise
        finally:
            self.background_service.stop(
                immediate=self._stop_policy is StopPolicy.IMMEDIATE,
                wait=True,
                timeout=_BACKGROUND_SERVICE_SHUTDOWN_TIMEOUT_SECONDS,
            )
            self.direct_compact_extraction.shutdown(
                wait=self._stop_policy is not StopPolicy.IMMEDIATE,
                timeout=_BACKGROUND_SERVICE_SHUTDOWN_TIMEOUT_SECONDS,
            )
            self.feedback_attribution.shutdown(
                wait=self._stop_policy is not StopPolicy.IMMEDIATE,
                timeout=_BACKGROUND_SERVICE_SHUTDOWN_TIMEOUT_SECONDS,
            )
            gateway_runtime = gateway_runtime_config(self.config)
            self._disconnect_adapters(connected_adapters, gateway_runtime.log_paths["errors.log"])
            self.agent_manager.evict_all()
            cleanup_runtime_files(self.runtime)
            if not failed:
                write_daemon_status(
                    self.runtime.status_path,
                    idle_status(
                        config=self.config,
                        runtime=self.runtime,
                        adapter_names=self._adapter_names,
                        message="Daemon stopped.",
                        background_status=self.background_service.status(),
                    ),
                )
            runtime_lock.release()
            if restore_signal_handlers is not None:
                try:
                    restore_signal_handlers()
                except Exception as exc:
                    restore_error = exc
            if restore_error is not None and not failed:
                raise restore_error

    def stop(self, policy: StopPolicy | StopPolicyValue = StopPolicy.GRACEFUL) -> None:
        """Request daemon shutdown."""

        self._stop_policy = StopPolicy(policy)
        self._stop_requested.set()
        self.background_service.stop(
            immediate=self._stop_policy is StopPolicy.IMMEDIATE,
            wait=False,
        )
        self.direct_compact_extraction.shutdown(wait=False)
        self.feedback_attribution.shutdown(wait=False)
        write_daemon_status(
            self.runtime.status_path,
            self._set_status(
                state="stopping",
                message=self._stop_message(self._stop_policy),
            ),
        )
        if self._server is not None:
            if self._stop_policy is StopPolicy.IMMEDIATE:
                self._server.stop_immediately()
            else:
                self._server.stop()

    @property
    def stop_policy(self) -> StopPolicy | None:
        """Return the shutdown policy requested for the daemon."""

        return self._stop_policy

    def status(
        self,
        *,
        state: str | None = None,
        message: str = "Daemon is running.",
    ) -> DaemonStatus:
        """Build current daemon status."""

        if state is None:
            state = self._state
            message = self._status_message
        if self._stop_requested.is_set() and state == "running":
            state = "stopping"
            message = self._status_message
        return running_status(
            config=self.config,
            runtime=self.runtime,
            adapter_names=self._adapter_names,
            state=state,
            message=message,
            background_status=self.background_service.status(),
        )

    def _set_status(self, *, state: str, message: str) -> DaemonStatus:
        self._state = state
        self._status_message = message
        return self.status(state=state, message=message)

    def _stop_message(self, policy: StopPolicy) -> str:
        if policy is StopPolicy.IMMEDIATE:
            return "Daemon is stopping immediately."
        return "Daemon is draining the current request before stopping."

    def _handle_turn(self, request: DaemonRequest) -> dict[str, Any]:
        session_id = request.session_id or new_session_id()
        message = request.message or ""
        if self.store.is_import_session(session_id):
            return error_response("IMPORT_SESSION_NOT_CHAT", IMPORT_SESSION_CHAT_MESSAGE)
        turn = self.turn_guard.begin(session_id, message)
        if not turn.accepted:
            return error_response("ACTIVE_TURN_IN_PROGRESS", LOCAL_BUSY_MESSAGE)

        try:
            agent = self.agent_manager.get_or_create(session_id)
            result = agent.respond(
                message,
                session_id=session_id,
                source_metadata=self._local_source_metadata(request),
            )
            return ok_response(session_id=result.session_id, response=result.response)
        finally:
            self.turn_guard.complete(session_id)

    def _local_source_metadata(self, request: DaemonRequest) -> dict[str, Any]:
        command = "chat" if request.type == "chat_turn" else request.type
        metadata: dict[str, Any] = {"channel": "cli", "command": command}
        if request.source_metadata:
            metadata["client"] = dict(request.source_metadata)
        return metadata

    def _handle_conversation_import(self, request: DaemonRequest) -> dict[str, Any]:
        payload_json = request.payload_json
        if payload_json is None:
            return error_response("INVALID_REQUEST", "payload_json must be a string.")
        try:
            with self._conversation_import_lock:
                summary = self.conversation_import_service.import_payload(
                    payload_json,
                    input_name=request.input_name,
                    dry_run=request.dry_run,
                )
        except ConversationImportValidationFailed as exc:
            return error_response(
                "VALIDATION_ERROR",
                str(exc),
                details=[error.to_dict() for error in exc.errors],
            )
        if not request.dry_run and summary.messages_inserted > 0:
            self.background_service.wake()
        return ok_response(summary=summary.to_dict())

    def _handle_conversation_import_status(self, request: DaemonRequest) -> dict[str, Any]:
        batch_id = request.batch_id
        if batch_id is None:
            return error_response("INVALID_REQUEST", "batch_id must be a non-empty string.")
        summary = self.store.get_import_status_summary(batch_id)
        if summary is None:
            return error_response(
                "IMPORT_BATCH_NOT_FOUND",
                f"Import batch not found: {batch_id}",
            )
        response = ok_response(status=_import_status_summary_dict(summary))
        if request.verbose:
            response["conversations"] = self._import_status_conversation_details(batch_id)
        return response

    def _import_status_conversation_details(self, batch_id: str) -> list[dict[str, Any]]:
        batch = self.store.get_import_batch(batch_id)
        metadata_items = _import_status_conversation_metadata(batch)
        details: dict[str, dict[str, Any]] = {}
        order: list[str] = []

        for item in metadata_items:
            external_id = str(item.get("external_conversation_id") or "")
            if not external_id:
                continue
            details[external_id] = {
                "external_conversation_id": external_id,
                "title": item.get("title") if isinstance(item.get("title"), str) else None,
                "session_id": None,
                "messages_inserted": _int_value(item.get("messages_inserted")),
                "messages_deduped": _int_value(item.get("messages_deduped")),
                "session_reused": bool(item.get("session_reused", False)),
                **_empty_import_extraction_counts(),
            }
            order.append(external_id)

        state_service = CognitionStateStore(self.store)
        for message in self.store.list_imported_messages(import_batch_id=batch_id):
            external_id = message.external_conversation_id
            conversation = self.store.get_imported_conversation(
                message.source_provider,
                external_id,
            )
            if external_id not in details:
                details[external_id] = {
                    "external_conversation_id": external_id,
                    "title": conversation.title if conversation is not None else None,
                    "session_id": (
                        conversation.session_id if conversation is not None else None
                    ),
                    "messages_inserted": 0,
                    "messages_deduped": 0,
                    "session_reused": False,
                    **_empty_import_extraction_counts(),
                }
                order.append(external_id)
            if conversation is not None:
                details[external_id]["title"] = conversation.title
                details[external_id]["session_id"] = conversation.session_id
                status = _imported_message_extraction_status(
                    state_service,
                    session_id=conversation.session_id,
                    session_message_id=message.session_message_id,
                )
            else:
                status = "pending"
            details[external_id][f"extraction_{status}"] += 1

        return [details[external_id] for external_id in order]

    def _assert_single_owner(self) -> None:
        existing = read_daemon_status(self.runtime.status_path)
        if existing is not None and existing.running and is_pid_running(existing.pid):
            raise DaemonAlreadyRunningError(
                "Daemon is already running for this runtime path."
            )

    def _acquire_runtime_lock(self) -> DaemonRuntimeLock:
        try:
            return DaemonRuntimeLock.acquire(self.runtime)
        except FileExistsError as exc:
            raise DaemonAlreadyRunningError(
                "Daemon is already running for this runtime path."
            ) from exc

    def _install_signal_handlers(
        self,
        signal_module: Any = signal,
    ) -> Callable[[], None]:
        previous_handlers: dict[int, Any] = {}

        def handle_stop_signal(_signum: int, _frame: FrameType | None) -> None:
            self.stop(StopPolicy.GRACEFUL)

        for signum in (signal_module.SIGTERM, signal_module.SIGINT):
            try:
                previous_handler = signal_module.getsignal(signum)
                signal_module.signal(signum, handle_stop_signal)
                previous_handlers[signum] = previous_handler
            except ValueError:
                for installed_signum, handler in previous_handlers.items():
                    try:
                        signal_module.signal(installed_signum, handler)
                    except ValueError:
                        pass
                return lambda: None

        def restore() -> None:
            for signum, handler in previous_handlers.items():
                signal_module.signal(signum, handler)

        return restore

    def _disconnect_adapters(self, adapters: list[Any], error_log_path: Path) -> None:
        for adapter in adapters:
            try:
                adapter.disconnect()
            except Exception as exc:
                if not error_log_path.parent.exists():
                    error_log_path.parent.mkdir(parents=True, exist_ok=True)
                append_gateway_log(
                    error_log_path,
                    event="gateway.adapter.disconnect_failed",
                    message="Gateway adapter disconnect failed.",
                    level="error",
                    metadata={
                        "adapter": adapter_name(adapter),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )


def _import_status_summary_dict(summary: ImportStatusSummary) -> dict[str, Any]:
    return {
        "batch_id": summary.batch_id,
        "source_provider": summary.source_provider,
        "status": summary.status,
        "conversations_seen": summary.conversations_seen,
        "messages_seen": summary.messages_seen,
        "conversations_created": summary.conversations_created,
        "conversations_reused": summary.conversations_reused,
        "messages_inserted": summary.messages_inserted,
        "messages_deduped": summary.messages_deduped,
        "extraction_pending": summary.extraction_pending,
        "extraction_claimed": summary.extraction_claimed,
        "extraction_processed": summary.extraction_processed,
        "extraction_failed": summary.extraction_failed,
        "extraction_skipped": summary.extraction_skipped,
        "created_at": summary.created_at,
        "updated_at": summary.updated_at,
        "error_summary": summary.error_summary,
    }


def _import_status_conversation_metadata(
    batch: ImportBatchRecord | None,
) -> list[dict[str, Any]]:
    if batch is None:
        return []
    conversations = batch.metadata.get("conversations")
    if not isinstance(conversations, list):
        return []
    return [dict(item) for item in conversations if isinstance(item, dict)]


def _empty_import_extraction_counts() -> dict[str, int]:
    return {key: 0 for key in _IMPORT_EXTRACTION_STATUS_KEYS}


def _imported_message_extraction_status(
    state_service: CognitionStateStore,
    *,
    session_id: str,
    session_message_id: str,
) -> str:
    try:
        progress = state_service.ledger.get_source_progress(
            BackgroundSourceRef("session_message", session_message_id),
            stage=BackgroundStage.EXTRACTION,
            target_unit=f"session:{session_id}",
        )
    except KeyError:
        return "pending"
    status = str(progress.status)
    if status not in {"pending", "claimed", "processed", "failed", "skipped"}:
        return "pending"
    return status


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0
