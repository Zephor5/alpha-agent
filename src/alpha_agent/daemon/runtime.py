"""Daemon lifecycle and request handling."""

from __future__ import annotations

import signal
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from threading import Event
from types import FrameType
from typing import Any

from alpha_agent.config import AlphaConfig
from alpha_agent.daemon.manager import AgentFactory, AgentManager, initialize_store
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
from alpha_agent.gateway.runner import ActiveTurnGuard, GatewayRuntimeBridge
from alpha_agent.gateway.session import GatewayDeduplicator, GatewaySessionStore, SessionMode
from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.session import new_session_id

LOCAL_BUSY_MESSAGE = "This session already has an active Alpha turn."


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
        store: MemoryStore | None = None,
        agent_manager: AgentManager | None = None,
        turn_guard: ActiveTurnGuard | None = None,
        runtime: DaemonRuntimeConfig | None = None,
    ):
        self.config = config
        self.runtime = runtime or daemon_runtime_config(config)
        self.store = store or initialize_store(config)
        self.agent_manager = agent_manager or AgentManager(AgentFactory(config, self.store))
        self.turn_guard = turn_guard or ActiveTurnGuard(bypass_commands=set())
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
        if request.type == "consolidate_memory":
            report = ConsolidationService(self.store).consolidate()
            return ok_response(response=report.render())
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
                ),
            )
            raise
        finally:
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
            except Exception:
                if not error_log_path.parent.exists():
                    error_log_path.parent.mkdir(parents=True, exist_ok=True)
