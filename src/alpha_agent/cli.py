"""Command line interface for Alpha Agent."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Annotated, Any, Protocol

import typer
from prompt_toolkit import prompt as _terminal_prompt
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.goals import GoalRegistry
from alpha_agent.cognition.loops import (
    CheckpointStore,
    ConsolidationConfig,
    ConsolidationLoop,
    DriveConfig,
    DriveLoop,
    Scheduler,
    WorkerReport,
)
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
)
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.config import (
    AlphaConfig,
    default_config_path,
    load_config,
    read_config_value,
    set_config_value,
    write_default_config,
)
from alpha_agent.conversation_import.deepseek import (
    DeepSeekExportConversionError,
    convert_deepseek_export,
)
from alpha_agent.daemon.client import DaemonClient
from alpha_agent.daemon.conversation_import import (
    MAX_CONVERSATION_IMPORT_PAYLOAD_BYTES,
    ConversationImportService,
    ConversationImportValidationFailed,
)
from alpha_agent.daemon.manager import build_provider, initialize_store
from alpha_agent.daemon.runtime import AlphaDaemon, DaemonAlreadyRunningError
from alpha_agent.daemon.status import (
    DaemonRuntimeConfig,
    DaemonStatus,
    daemon_runtime_config,
    read_daemon_status,
)
from alpha_agent.daemon.status import (
    idle_status as daemon_idle_status,
)
from alpha_agent.daemon.status import (
    is_pid_running as daemon_pid_running,
)
from alpha_agent.gateway.config import (
    configured_adapter_names,
    ensure_gateway_runtime_files,
    gateway_runtime_config,
)
from alpha_agent.gateway.logging import append_gateway_log
from alpha_agent.gateway.status import gateway_tables_available
from alpha_agent.llm.codex import CODEX_DEFAULT_MODEL
from alpha_agent.llm.deepseek import DEEPSEEK_DEFAULT_MODEL
from alpha_agent.llm.mimo import MIMO_DEFAULT_MODEL
from alpha_agent.llm.openai_compatible import OPENAI_COMPATIBLE_DEFAULT_MODEL
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.runtime.chat_messages import session_message_to_chat, strip_system_reminder
from alpha_agent.runtime.prompt_builder import build_answer_prompt_messages
from alpha_agent.runtime.session import new_session_id
from alpha_agent.runtime.session_context import SessionContextAssembler
from alpha_agent.skills.manager import SkillManager
from alpha_agent.state.models import RuntimeTrace, SessionMessage
from alpha_agent.state.store import StateStore
from alpha_agent.tools.default import build_tool_registry

console = Console()
app = typer.Typer(help="Alpha Agent cognition runtime.")
skills_app = typer.Typer(help="Skill commands.")
debug_app = typer.Typer(help="Debug commands.")
gateway_app = typer.Typer(help="Gateway operational commands.")
config_app = typer.Typer(help="Configuration commands.")
daemon_app = typer.Typer(help="Daemon runtime commands.")
cognition_app = typer.Typer(help="Cognition inspection commands.")
cognition_import_app = typer.Typer(help="External conversation import commands.")
cognition_import_convert_app = typer.Typer(help="Source export conversion commands.")
goals_app = typer.Typer(help="Drive Loop goal commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(cognition_app, name="cognition")
cognition_app.add_typer(cognition_import_app, name="import")
cognition_import_app.add_typer(cognition_import_convert_app, name="convert")
cognition_app.add_typer(goals_app, name="goals")

DAEMON_START_TIMEOUT_SECONDS = 5.0
DAEMON_STOP_TIMEOUT_SECONDS = 5.0
DAEMON_START_POLL_INTERVAL_SECONDS = 0.1
CHAT_HISTORY_PREVIEW_LIMIT = 8
CHAT_HISTORY_MESSAGE_MAX_CHARS = 900
CHAT_TURN_PROGRESS_POLL_INTERVAL_SECONDS = 0.1
CHAT_DISPLAY_MESSAGE_KINDS = {
    "user_message",
    "assistant_message",
    "tool_message",
    "compressed_message",
}
CHAT_TURN_DISPLAY_MESSAGE_KINDS = {
    "assistant_message",
    "tool_message",
}


class _DaemonRequestClient(Protocol):
    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one daemon request."""
        ...


def _display_model(config: AlphaConfig) -> str:
    if config.llm_model:
        return config.llm_model
    if config.llm_provider == "deepseek":
        return f"{DEEPSEEK_DEFAULT_MODEL} (provider default)"
    if config.llm_provider == "mimo":
        return f"{MIMO_DEFAULT_MODEL} (provider default)"
    if config.llm_provider in {"codex", "openai-codex", "openai_codex"}:
        return f"{CODEX_DEFAULT_MODEL} (provider default)"
    if config.llm_provider in {"openai-compatible", "openai", "compatible"}:
        return f"{OPENAI_COMPATIBLE_DEFAULT_MODEL} (provider default)"
    return ""


def _store(config: AlphaConfig) -> StateStore:
    return initialize_store(config)


def _read_chat_message() -> str:
    if sys.stdin.isatty() and sys.stdout.isatty():
        return _terminal_prompt("You: ")
    return typer.prompt("You")


def _render_chat_header(session_id: str) -> None:
    header = Table.grid(expand=True)
    header.add_column(style="bold cyan")
    header.add_column(justify="right", style="dim")
    header.add_row("Alpha Chat", f"session {session_id}")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))


def _render_chat_history_preview(
    store: StateStore,
    session_id: str,
    *,
    limit: int | None = None,
) -> None:
    display_messages = _displayable_session_messages(store, session_id)
    if not display_messages:
        console.print(
            Panel(
                Text("No prior messages for this session.", style="dim"),
                title="Recent Session Context",
                box=box.ROUNDED,
                border_style="dim",
                padding=(0, 1),
            )
        )
        return

    table = _build_chat_history_table(display_messages, limit=limit)
    console.print(table)


def _build_chat_history_table(
    display_messages: list[SessionMessage],
    *,
    limit: int | None = None,
) -> Table:
    preview_limit = CHAT_HISTORY_PREVIEW_LIMIT if limit is None else limit
    visible_messages = display_messages[-max(1, preview_limit) :]
    omitted_count = len(display_messages) - len(visible_messages)
    table = Table(
        title="Recent Session Context",
        box=box.SIMPLE_HEAVY,
        show_header=False,
        show_lines=True,
        expand=True,
        padding=(0, 1),
    )
    table.add_column("Role", no_wrap=True, style="bold cyan", width=9)
    table.add_column("Message", overflow="fold")
    if omitted_count > 0:
        table.caption = (
            f"Showing last {len(visible_messages)} messages; {omitted_count} older omitted."
        )
    for message in visible_messages:
        table.add_row(
            _chat_history_role_label(message),
            Text(_chat_history_content(message), overflow="fold"),
        )
    return table


def _render_assistant_reply(content: str) -> None:
    console.print(
        Panel(
            Text(content),
            title="Alpha",
            box=box.ROUNDED,
            border_style="green",
            padding=(0, 1),
        )
    )


def _render_tool_reply(content: str) -> None:
    console.print(
        Panel(
            Text(content),
            title="Tool",
            box=box.ROUNDED,
            border_style="yellow",
            padding=(0, 1),
        )
    )


def _render_chat_turn_messages(
    messages: list[SessionMessage],
    *,
    fallback_response: str,
) -> None:
    if not messages:
        _render_assistant_reply(fallback_response)
        return

    for message in messages:
        content = _chat_turn_content(message)
        if message.llm_role == "tool":
            _render_tool_reply(content)
        else:
            _render_assistant_reply(content)


def _chat_turn_content(message: SessionMessage) -> str:
    if message.kind == "assistant_message" and not message.tool_calls:
        return message.model_content if message.model_content is not None else message.raw_content
    return _chat_history_content(message)


def _displayable_session_messages(store: StateStore, session_id: str) -> list[SessionMessage]:
    projection = SessionContextAssembler(store).load(session_id)
    return [
        message
        for message in projection.source_messages
        if message.kind in CHAT_DISPLAY_MESSAGE_KINDS
    ]


def _displayable_chat_turn_messages(
    store: StateStore,
    session_id: str,
    *,
    after_ordinal: int,
) -> list[SessionMessage]:
    return [
        message
        for message in store.list_session_messages(session_id, after_ordinal=after_ordinal)
        if message.kind in CHAT_TURN_DISPLAY_MESSAGE_KINDS
    ]


def _chat_turn_messages_include_response(
    messages: list[SessionMessage],
    response_text: str,
) -> bool:
    if not response_text:
        return False
    return any(
        message.kind == "assistant_message"
        and not message.tool_calls
        and _chat_turn_content(message) == response_text
        for message in messages
    )


def _render_chat_turn_progress(
    store: StateStore,
    session_id: str,
    *,
    after_ordinal: int,
) -> int:
    messages = store.list_session_messages(session_id, after_ordinal=after_ordinal)
    if not messages:
        return after_ordinal
    turn_messages = [
        message
        for message in messages
        if message.kind in CHAT_TURN_DISPLAY_MESSAGE_KINDS
    ]
    if turn_messages:
        _render_chat_turn_messages(turn_messages, fallback_response="")
    return messages[-1].ordinal


def _request_chat_turn_with_progress(
    client: _DaemonRequestClient,
    payload: dict[str, Any],
    *,
    store: StateStore,
    session_id: str,
    after_ordinal: int,
) -> tuple[dict[str, Any], int]:
    done = threading.Event()
    responses: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def run_request() -> None:
        try:
            responses.append(client.request(payload))
        except BaseException as exc:
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run_request, daemon=True)
    thread.start()

    rendered_after_ordinal = after_ordinal
    while not done.wait(CHAT_TURN_PROGRESS_POLL_INTERVAL_SECONDS):
        rendered_after_ordinal = _render_chat_turn_progress(
            store,
            session_id,
            after_ordinal=rendered_after_ordinal,
        )

    thread.join()
    rendered_after_ordinal = _render_chat_turn_progress(
        store,
        session_id,
        after_ordinal=rendered_after_ordinal,
    )
    if errors:
        raise errors[0]
    if not responses:
        raise RuntimeError("Daemon request finished without a response.")
    return responses[0], rendered_after_ordinal


def _latest_session_ordinal(store: StateStore, session_id: str) -> int:
    messages = store.list_session_messages(session_id)
    if not messages:
        return 0
    return messages[-1].ordinal


def _chat_history_role_label(message: SessionMessage) -> str:
    if message.kind == "compressed_message":
        return "Context"
    if message.llm_role == "tool":
        return "Tool"
    if message.llm_role == "assistant":
        return "Alpha"
    return "You"


def _chat_history_content(message: SessionMessage) -> str:
    content = message.model_content if message.model_content is not None else message.raw_content
    if message.kind == "compressed_message":
        content = strip_system_reminder(content)
    elif message.kind == "tool_message":
        content = _tool_result_display(message, content)
    elif message.tool_calls:
        content = _assistant_tool_call_display(message, content)
    return _truncate_chat_history_content(content)


def _assistant_tool_call_display(message: SessionMessage, content: str) -> str:
    parts = [content.strip()] if content.strip() else []
    parts.extend(_tool_call_display(tool_call) for tool_call in message.tool_calls)
    return "\n".join(parts)


def _tool_call_display(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = str(function.get("name") or "unknown")
        arguments = function.get("arguments")
    else:
        name = str(tool_call.get("name") or tool_call.get("id") or "unknown")
        arguments = tool_call.get("arguments")
    if arguments is None or arguments == "":
        return f"Tool call: {name}"
    return f"Tool call: {name}\n{_display_jsonish(arguments)}"


def _tool_result_display(message: SessionMessage, content: str) -> str:
    tool_name = message.provider_metadata.get("tool_name")
    name = str(tool_name or message.tool_call_id or "unknown")
    if content.strip():
        return f"Tool result: {name}\n{content.strip()}"
    return f"Tool result: {name}"


def _display_jsonish(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def _truncate_chat_history_content(content: str) -> str:
    if len(content) <= CHAT_HISTORY_MESSAGE_MAX_CHARS:
        return content
    return f"{content[: CHAT_HISTORY_MESSAGE_MAX_CHARS - 3].rstrip()}..."


def _render_daemon_status(status: DaemonStatus) -> None:
    process = "running" if status.running else "not running"
    table = Table(title="Daemon Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("State", status.state)
    table.add_row("Process", process)
    table.add_row("PID", str(status.pid) if status.pid is not None else "-")
    table.add_row("Socket path", status.socket_path)
    table.add_row("Status path", status.status_path)
    table.add_row("DB path", status.db_path)
    table.add_row("Log dir", status.log_dir)
    table.add_row("Adapters", ", ".join(status.adapters) if status.adapters else "none")
    table.add_row("Background enabled", str(status.background_enabled).lower())
    table.add_row("Background state", status.background_state)
    table.add_row("Background last tick", status.background_last_tick or "-")
    table.add_row("Background last success", status.background_last_success or "-")
    table.add_row("Background last error", status.background_last_error or "-")
    table.add_row("Background next tick", status.background_next_tick or "-")
    table.add_row("Message", status.message)
    console.print(table)
    typer.echo(f"Socket path: {status.socket_path}")
    typer.echo(f"Status path: {status.status_path}")
    typer.echo(f"DB path: {status.db_path}")
    typer.echo(f"Log dir: {status.log_dir}")


def _daemon_not_running_message() -> str:
    return "Daemon is not running. Run alpha daemon start."


def _daemon_status_is_running(response: dict[str, Any]) -> bool:
    status = response.get("status")
    if not isinstance(status, dict):
        return False
    return bool(status.get("running")) and str(status.get("state", "")) == "running"


def _daemon_status_is_stopped(response: dict[str, Any]) -> bool:
    error = response.get("error")
    if isinstance(error, dict) and error.get("code") == "DAEMON_NOT_RUNNING":
        return True
    status = response.get("status")
    if not isinstance(status, dict):
        return False
    return not bool(status.get("running"))


def _daemon_log_path(runtime: DaemonRuntimeConfig) -> Path:
    return runtime.log_dir / "daemon.log"


def _spawn_daemon_background(runtime: DaemonRuntimeConfig) -> tuple[subprocess.Popen[bytes], Path]:
    log_path = _daemon_log_path(runtime)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "alpha_agent.cli", "daemon", "run"]
    log_file = log_path.open("ab")
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_file.close()
    return process, log_path


def _wait_for_daemon_running(
    client: DaemonClient,
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float = DAEMON_START_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        response = client.status()
        if _daemon_status_is_running(response):
            status = response.get("status")
            return dict(status) if isinstance(status, dict) else {}
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Daemon exited before startup completed with exit code {return_code}."
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for daemon startup.")
        time.sleep(DAEMON_START_POLL_INTERVAL_SECONDS)


def _wait_for_daemon_stopped(
    client: DaemonClient,
    *,
    timeout_seconds: float = DAEMON_STOP_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if _daemon_status_is_stopped(client.status()):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for daemon shutdown.")
        time.sleep(DAEMON_START_POLL_INTERVAL_SECONDS)


def _start_daemon_background_and_report(
    *,
    runtime: DaemonRuntimeConfig,
    client: DaemonClient,
    message_prefix: str,
) -> None:
    process, log_path = _spawn_daemon_background(runtime)
    try:
        status = _wait_for_daemon_running(client, process)
    except (RuntimeError, TimeoutError) as exc:
        console.print(str(exc))
        console.print(f"Daemon log: {log_path}")
        raise typer.Exit(1) from exc
    pid = status.get("pid") or process.pid
    console.print(f"{message_prefix} with PID {pid}.")
    console.print(f"Daemon log: {log_path}")


def _client_response_or_exit(response: dict[str, Any]) -> dict[str, Any]:
    if response.get("ok") is True:
        return response
    error = response.get("error")
    if isinstance(error, dict):
        message = str(error.get("message") or error.get("code") or "Daemon request failed.")
        if error.get("code") == "DAEMON_NOT_RUNNING":
            message = _daemon_not_running_message()
        console.print(message)
        _render_error_details(error)
    else:
        console.print("Daemon request failed.")
    raise typer.Exit(1)


def _render_error_details(error: dict[str, Any]) -> None:
    details = error.get("details")
    if not isinstance(details, list) or not details:
        return
    table = Table(title="Validation Details", box=box.SIMPLE_HEAVY)
    table.add_column("Path")
    table.add_column("Code")
    table.add_column("Message")
    for item in details:
        if not isinstance(item, dict):
            continue
        table.add_row(
            str(item.get("path") or "-"),
            str(item.get("code") or "-"),
            str(item.get("message") or ""),
        )
        typer.echo(
            "validation_error "
            f"path={item.get('path') or '-'} "
            f"code={item.get('code') or '-'} "
            f"message={item.get('message') or ''}"
        )
    console.print(table)


def _source_metadata(command: str) -> dict[str, str]:
    return {"channel": "cli", "command": command}


@app.command("init")
def init_command() -> None:
    """Initialize the local data directory and SQLite database."""

    config = load_config()
    wrote_config = write_default_config()
    _store(config)
    console.print(f"Initialized Alpha Agent database at [bold]{config.db_path}[/bold]")
    if wrote_config:
        console.print(f"Created config file at [bold]{default_config_path()}[/bold]")
    else:
        console.print(f"Config file already exists at [bold]{default_config_path()}[/bold]")


@config_app.command("init")
def config_init(
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing config file."),
    ] = False,
) -> None:
    """Create the default TOML config file."""

    config_path = default_config_path()
    wrote = write_default_config(config_path, overwrite=force)
    if wrote:
        console.print(f"Created config file at [bold]{config_path}[/bold]")
    else:
        console.print(f"Config file already exists at [bold]{config_path}[/bold]")


@config_app.command("show")
def config_show() -> None:
    """Show the effective configuration without printing secret values."""

    config = load_config()
    table = Table(title="Alpha Config")
    table.add_column("Key")
    table.add_column("Value")
    rows = {
        "config_path": str(default_config_path()),
        "db_path": str(config.db_path),
        "log_dir": str(config.log_dir),
        "gateway_status_path": str(config.gateway_status_path),
        "daemon_socket_path": str(config.daemon_socket_path),
        "daemon_status_path": str(config.daemon_status_path),
        "llm_provider": config.llm_provider,
        "llm_model": _display_model(config),
        "llm_debug_logging": str(config.llm_debug_logging).lower(),
        "llm_context_tool_truncate_threshold_ratio": str(
            config.llm_context.tool_truncate_threshold_ratio
        ),
        "llm_context_handover_compress_threshold_ratio": str(
            config.llm_context.handover_compress_threshold_ratio
        ),
        "llm_context_minimum_remaining_tokens": str(
            config.llm_context.minimum_remaining_tokens
        ),
        "llm_provider_max_context_tokens": str(
            config.max_context_tokens_for_provider(config.llm_provider)
        ),
        "bash_tool_enabled": str(config.bash_tool.enabled).lower(),
        "cognition_background_enabled": str(config.cognition_background.enabled).lower(),
        "cognition_background_startup_delay_seconds": str(
            config.cognition_background.startup_delay_seconds
        ),
        "cognition_background_interval_seconds": str(
            config.cognition_background.interval_seconds
        ),
        "cognition_background_tick_timeout_seconds": str(
            config.cognition_background.tick_timeout_seconds
        ),
    }
    if config.llm_provider in {"openai-compatible", "openai", "compatible"}:
        rows["compatible_base_url"] = config.compatible_base_url or ""
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
    for key, value in rows.items():
        typer.echo(f"{key}={value}")
    typer.echo(f"Config path: {default_config_path()}")


@config_app.command("get")
def config_get(
    key: Annotated[str, typer.Argument(help="Dotted config key, e.g. llm.provider.")],
    reveal_secret: Annotated[
        bool,
        typer.Option("--reveal-secret", help="Print secret values instead of masking them."),
    ] = False,
) -> None:
    """Read one config value."""

    try:
        value = read_config_value(key, reveal_secret=reveal_secret)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(value)


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Dotted config key, e.g. llm.provider.")],
    value: Annotated[str, typer.Argument(help="Value to write.")],
) -> None:
    """Set one config value in the TOML config file."""

    try:
        parsed_value = set_config_value(key, value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    printable = "***" if key.strip().lower().endswith(("api_key", "access_token")) else parsed_value
    console.print(f"Set [bold]{key}[/bold] = {printable}")


@daemon_app.command("run")
def daemon_run() -> None:
    """Run the daemon runtime owner in the foreground."""

    try:
        AlphaDaemon(load_config()).run()
    except DaemonAlreadyRunningError as exc:
        console.print(str(exc))
        raise typer.Exit(1) from exc


@daemon_app.command("start")
def daemon_start() -> None:
    """Start the daemon runtime owner in the background."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    client = DaemonClient(runtime.socket_path)
    if _daemon_status_is_running(client.status()):
        console.print("Daemon is already running.")
        return
    _start_daemon_background_and_report(
        runtime=runtime,
        client=client,
        message_prefix="Daemon started",
    )


@daemon_app.command("restart")
def daemon_restart(
    immediate: Annotated[
        bool,
        typer.Option("--immediate", help="Stop accepting daemon requests immediately."),
    ] = False,
) -> None:
    """Restart the daemon runtime owner."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    client = DaemonClient(runtime.socket_path)
    if _daemon_status_is_running(client.status()):
        policy = "immediate" if immediate else "graceful"
        response = _client_response_or_exit(client.stop(policy=policy))
        raw_status = response.get("status")
        if isinstance(raw_status, dict):
            console.print(str(raw_status.get("message") or "Daemon stopping."))
        else:
            console.print("Daemon stopping.")
        try:
            _wait_for_daemon_stopped(client)
        except TimeoutError as exc:
            console.print(str(exc))
            raise typer.Exit(1) from exc
        message_prefix = "Daemon restarted"
    else:
        console.print("Daemon is not running; starting it.")
        message_prefix = "Daemon started"

    _start_daemon_background_and_report(
        runtime=runtime,
        client=client,
        message_prefix=message_prefix,
    )


@daemon_app.command("status")
def daemon_status() -> None:
    """Show daemon runtime status."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    response = DaemonClient(runtime.socket_path).status()
    if response.get("ok") is True and isinstance(response.get("status"), dict):
        raw = response["status"]
        status = DaemonStatus(
            state=str(raw.get("state", "unknown")),
            running=bool(raw.get("running", False)),
            pid=int(raw["pid"]) if raw.get("pid") is not None else None,
            socket_path=str(raw.get("socket_path", runtime.socket_path)),
            status_path=str(raw.get("status_path", runtime.status_path)),
            updated_at=str(raw.get("updated_at", "")),
            adapters=[str(adapter) for adapter in raw.get("adapters", [])],
            db_path=str(raw.get("db_path", config.db_path)),
            log_dir=str(raw.get("log_dir", config.log_dir)),
            message=str(raw.get("message", "")),
            started_at=str(raw["started_at"]) if raw.get("started_at") is not None else None,
            background_enabled=bool(raw.get("background_enabled", False)),
            background_state=str(raw.get("background_state", "disabled")),
            background_last_tick=(
                str(raw["background_last_tick"])
                if raw.get("background_last_tick") is not None
                else None
            ),
            background_last_success=(
                str(raw["background_last_success"])
                if raw.get("background_last_success") is not None
                else None
            ),
            background_last_error=(
                str(raw["background_last_error"])
                if raw.get("background_last_error") is not None
                else None
            ),
            background_next_tick=(
                str(raw["background_next_tick"])
                if raw.get("background_next_tick") is not None
                else None
            ),
        )
        _render_daemon_status(status)
        return

    maybe_status = read_daemon_status(runtime.status_path)
    if maybe_status is None:
        status = daemon_idle_status(config=config, runtime=runtime)
    elif maybe_status.running and not daemon_pid_running(maybe_status.pid):
        status = daemon_idle_status(
            config=config,
            runtime=runtime,
            adapter_names=tuple(maybe_status.adapters),
            message="Daemon status file exists, but the recorded process is not running.",
        )
    else:
        status = maybe_status
    _render_daemon_status(status)


@daemon_app.command("stop")
def daemon_stop(
    immediate: Annotated[
        bool,
        typer.Option("--immediate", help="Stop accepting daemon requests immediately."),
    ] = False,
) -> None:
    """Request daemon shutdown."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    policy = "immediate" if immediate else "graceful"
    response = _client_response_or_exit(DaemonClient(runtime.socket_path).stop(policy=policy))
    raw_status = response.get("status")
    if isinstance(raw_status, dict):
        console.print(str(raw_status.get("message") or "Daemon stopping."))
    else:
        console.print("Daemon stopping.")


@gateway_app.command("status")
def gateway_status() -> None:
    """Show gateway runtime status."""

    from alpha_agent.gateway.status import idle_status, read_gateway_status

    config = load_config()
    runtime = gateway_runtime_config(config)
    status = read_gateway_status(runtime.status_path)
    if status is None:
        status = idle_status(db_path=config.db_path, log_dir=config.log_dir)
    table = Table(title="Gateway Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("State", status.state)
    table.add_row("Process", "running" if status.running else "not running")
    table.add_row("PID", str(status.pid) if status.pid is not None else "-")
    table.add_row("DB path", status.db_path)
    table.add_row("Log dir", status.log_dir)
    table.add_row("Adapters", ", ".join(status.adapters) if status.adapters else "none")
    table.add_row("Message", status.message)
    console.print(table)
    typer.echo(f"DB path: {status.db_path}")
    typer.echo(f"Log dir: {status.log_dir}")


@gateway_app.command("doctor")
def gateway_doctor() -> None:
    """Initialize and inspect local gateway runtime files."""

    config = load_config()
    runtime = gateway_runtime_config(config)
    ensure_gateway_runtime_files(runtime)
    _store(config)
    table = Table(title="Gateway Doctor")
    table.add_column("Check")
    table.add_column("Value")
    table.add_row("db_path", str(config.db_path))
    table.add_row("log_dir", str(config.log_dir))
    table.add_row("llm_provider", config.llm_provider)
    for name, available in gateway_tables_available(config.db_path).items():
        table.add_row(name, "ok" if available else "missing")
    adapter_names = configured_adapter_names()
    table.add_row("adapters", ", ".join(adapter_names) if adapter_names else "none")
    console.print(table)
    typer.echo(f"DB path: {config.db_path}")
    typer.echo(f"Log dir: {config.log_dir}")
    if not adapter_names:
        console.print("No real platform adapters configured.")
    append_gateway_log(
        runtime.log_paths["gateway.log"],
        event="gateway.doctor",
        message="Gateway doctor inspected local runtime files.",
        metadata={"adapters": adapter_names},
    )


@app.command()
def ask(message: Annotated[str, typer.Argument(help="Message to send to the agent.")]) -> None:
    """Run a single daemon-owned turn."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    response = _client_response_or_exit(
        DaemonClient(runtime.socket_path).request(
            {
                "type": "ask",
                "message": message,
                "session_id": None,
                "source_metadata": _source_metadata("ask"),
            }
        )
    )
    console.print(str(response.get("response", "")))


@app.command()
def chat(
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Reuse an existing session id."),
    ] = None,
) -> None:
    """Start an interactive daemon-owned chat."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    client = DaemonClient(runtime.socket_path)
    store = _store(config)
    session_id = session or new_session_id()
    if session is not None and store.is_import_session(session_id):
        console.print("Import sessions are hidden source material and cannot be opened in chat.")
        raise typer.Exit(1)
    _render_chat_header(session_id)
    if session is not None:
        _render_chat_history_preview(store, session_id)
    while True:
        try:
            message = _read_chat_message()
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip().lower() in {"/exit", "/quit"}:
            break
        request_session_id = session_id
        before_ordinal = _latest_session_ordinal(store, request_session_id)
        response, rendered_after_ordinal = _request_chat_turn_with_progress(
            client,
            {
                "type": "chat_turn",
                "message": message,
                "session_id": request_session_id,
                "source_metadata": _source_metadata("chat"),
            },
            store=store,
            session_id=request_session_id,
            after_ordinal=before_ordinal,
        )
        response = _client_response_or_exit(response)
        session_id = str(response.get("session_id") or request_session_id)
        response_text = str(response.get("response", ""))
        all_turn_messages = (
            _displayable_chat_turn_messages(
                store,
                session_id,
                after_ordinal=before_ordinal,
            )
            if session_id == request_session_id
            else []
        )
        turn_messages = [
            message
            for message in all_turn_messages
            if message.ordinal > rendered_after_ordinal
        ]
        fallback_response = (
            ""
            if _chat_turn_messages_include_response(all_turn_messages, response_text)
            else response_text
        )
        if turn_messages or fallback_response:
            _render_chat_turn_messages(
                turn_messages,
                fallback_response=fallback_response,
            )


@skills_app.command("list")
def skills_list() -> None:
    """List built-in skills."""

    table = Table(title="Built-in Skills")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Description")
    for skill in SkillManager().load_builtin_skills():
        table.add_row(skill.id, skill.name, skill.description)
        console.print(f"Skill: id={skill.id} name={skill.name}")
    console.print(table)


@debug_app.command()
def prompt(
    message: Annotated[str, typer.Argument(help="Message to render into a prompt.")],
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Existing session id to include."),
    ] = None,
    trace: Annotated[
        bool,
        typer.Option("--trace", help="Include recent cognitive event trace for the session."),
    ] = False,
    include_reminders: Annotated[
        bool,
        typer.Option(
            "--include-reminders",
            help="Include raw durable system reminder source messages in the prompt preview.",
        ),
    ] = False,
) -> None:
    """Print the runtime prompt preview and optional cognitive event trace."""

    config = load_config()
    store = _store(config)
    session_id = session or new_session_id()
    if session is not None and store.is_import_session(session_id):
        console.print(
            "Import sessions are hidden source material and cannot be rendered with debug prompt."
        )
        raise typer.Exit(1)
    context = SessionContextAssembler(store).load(session_id)
    session_history = (
        context.chat_messages
        if include_reminders
        else [
            session_message_to_chat(message)
            for message in context.source_messages
            if message.kind != "system_reminder"
        ]
    )
    messages = build_answer_prompt_messages(
        session_history=session_history,
        current_turn_messages=[{"role": "user", "content": message}],
    )
    for index, prompt_message in enumerate(messages, start=1):
        role = prompt_message["role"]
        content = prompt_message.get("content") or ""
        console.print(f"Message {index} [{role}]\n{content}", markup=False)
    if trace:
        _render_cognitive_trace(store, session_id)


def _render_cognitive_trace(store: StateStore, session_id: str) -> None:
    all_events = list(SQLiteEventLog(store).iter())
    turn_ids = {
        str(event.payload["turn_id"])
        for event in all_events
        if _event_belongs_to_session(event, session_id) and "turn_id" in event.payload
    }
    linked_event_ids = _linked_cognitive_event_ids(all_events, turn_ids)
    events = [
        event
        for event in all_events
        if (
            event.payload.get("turn_id") is not None
            and str(event.payload.get("turn_id")) in turn_ids
        )
        or str(event.id) in linked_event_ids
    ]
    console.print("Cognitive Trace", markup=False)
    if not events:
        console.print("(none)", markup=False)
    else:
        for event in events[-20:]:
            turn_id = event.payload.get("turn_id", "-")
            event_session_id = event.payload.get("session_id", "-")
            parents = ",".join(str(parent) for parent in event.causal_parents) or "-"
            console.print(
                f"{event.timestamp} kind={event.kind.value} turn_id={turn_id} "
                f"session_id={event_session_id} id={event.id} parents={parents}",
                markup=False,
            )
    memory_traces = _memory_tool_traces(store, session_id)
    if memory_traces:
        console.print("Memory Tool Trace", markup=False)
        for trace in memory_traces[-20:]:
            console.print(_format_memory_tool_trace(trace), markup=False)


def _memory_tool_traces(store: StateStore, session_id: str) -> list[RuntimeTrace]:
    return [
        trace
        for trace in store.list_runtime_traces(session_id)
        if trace.event_type in {"tool.completed", "tool.failed"}
        and _trace_tool_name(trace) in {"memory_recall", "memory_propose"}
    ]


def _format_memory_tool_trace(trace: RuntimeTrace) -> str:
    tool_name = _trace_tool_name(trace)
    output = _trace_output(trace)
    parts = [
        f"{trace.timestamp} tool={tool_name}",
        f"trace={trace.id}",
        f"event={trace.event_type}",
    ]
    status = output.get("status")
    if isinstance(status, str) and status:
        parts.append(f"status={status}")
    next_action = output.get("next_action")
    if isinstance(next_action, str) and next_action:
        parts.append(f"next_action={next_action}")
    if tool_name == "memory_recall":
        result_ids = _ids_from_items(output.get("results"))
        if result_ids:
            parts.append(f"results={','.join(result_ids)}")
    elif tool_name == "memory_propose":
        updates, targets, reviewed, candidates, new_beliefs = _memory_update_trace_fields(output)
        if updates:
            parts.append(f"updates={';'.join(updates)}")
        if targets:
            parts.append(f"targets={','.join(targets)}")
        if reviewed:
            parts.append(f"reviewed={','.join(reviewed)}")
        if candidates:
            parts.append(f"candidates={','.join(candidates)}")
        if new_beliefs:
            parts.append(f"new_beliefs={','.join(new_beliefs)}")
    return " ".join(parts)


def _trace_tool_name(trace: RuntimeTrace) -> str:
    name = trace.metadata.get("tool_name")
    if isinstance(name, str) and name:
        return name
    result = _trace_result(trace)
    result_name = result.get("name")
    return str(result_name) if result_name is not None else ""


def _trace_result(trace: RuntimeTrace) -> dict[str, Any]:
    result = trace.metadata.get("result")
    return result if isinstance(result, dict) else {}


def _trace_output(trace: RuntimeTrace) -> dict[str, Any]:
    output = _trace_result(trace).get("output")
    return output if isinstance(output, dict) else {}


def _memory_update_trace_fields(
    output: dict[str, Any],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    updates: list[str] = []
    targets: list[str] = []
    reviewed: list[str] = []
    candidates: list[str] = []
    new_beliefs: list[str] = []
    for result in _dict_items(output.get("results")):
        operation = str(result.get("operation") or "-")
        decision = str(result.get("decision") or "-")
        updates.append(f"{operation}:{decision}")
        targets.extend(_strings_from_items(result.get("target_belief_ids")))
        reviewed.extend(_strings_from_items(result.get("reviewed_candidate_ids")))
        candidates.extend(_ids_from_items(result.get("candidates")))
        new_belief = result.get("new_belief_id")
        if new_belief is not None:
            new_beliefs.append(str(new_belief))
    return (
        _unique_preserving_order(updates),
        _unique_preserving_order(targets),
        _unique_preserving_order(reviewed),
        _unique_preserving_order(candidates),
        _unique_preserving_order(new_beliefs),
    )


def _dict_items(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _ids_from_items(value: object) -> list[str]:
    return [
        str(item["id"])
        for item in _dict_items(value)
        if item.get("id") is not None
    ]


def _strings_from_items(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if item is not None]


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _linked_cognitive_event_ids(
    events: list[CognitiveEvent],
    turn_ids: set[str],
) -> set[str]:
    linked: set[str] = set()
    for event in events:
        if event.kind != CognitiveEventKind.TURN_SOURCES_RECORDED:
            continue
        if str(event.payload.get("turn_id") or "") not in turn_ids:
            continue
        for field in ("cognitive_event_ids", "tool_cognitive_event_ids"):
            values = event.payload.get(field)
            if isinstance(values, list):
                linked.update(str(value) for value in values)
    return linked


def _event_belongs_to_session(event: CognitiveEvent, session_id: str) -> bool:
    raw_session_id = event.payload.get("session_id")
    return isinstance(raw_session_id, str) and raw_session_id == session_id


@cognition_import_convert_app.command("deepseek")
def cognition_import_convert_deepseek(
    source: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Raw DeepSeek conversation export JSON file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Argument(help="Normalized conversation import JSON output file."),
    ],
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing output file."),
    ] = False,
) -> None:
    """Convert a raw DeepSeek export into normalized import JSON."""

    if output.exists():
        if output.is_dir():
            console.print("Output path is a directory.")
            raise typer.Exit(1)
        if not force:
            console.print("Output file already exists. Use --force to overwrite.")
            raise typer.Exit(1)
    output_parent = output.parent
    if output_parent and not output_parent.exists():
        console.print("Output directory does not exist.")
        raise typer.Exit(1)
    raw_json = _read_deepseek_export_file(source)
    try:
        payload = convert_deepseek_export(raw_json)
    except DeepSeekExportConversionError as exc:
        console.print(str(exc))
        _render_error_details({"details": [error.to_dict() for error in exc.errors]})
        raise typer.Exit(1) from exc

    payload_json = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _validate_converted_conversation_import_payload(payload_json, input_name=output.name)
    try:
        output.write_text(payload_json, encoding="utf-8")
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _render_deepseek_conversion_summary(payload, output_name=output.name)


@cognition_import_app.command("conversations")
def cognition_import_conversations(
    file: Annotated[
        Path,
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Normalized conversation import JSON file.",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and report planned counts without writes."),
    ] = False,
) -> None:
    """Import normalized external conversation JSON through the daemon."""

    payload_json = _read_conversation_import_file(file)
    config = load_config()
    runtime = daemon_runtime_config(config)
    response = _client_response_or_exit(
        DaemonClient(runtime.socket_path).request(
            {
                "type": "conversation_import",
                "input_name": file.name,
                "payload_json": payload_json,
                "dry_run": dry_run,
            }
        )
    )
    summary = response.get("summary")
    if not isinstance(summary, dict):
        console.print("Daemon response did not include an import summary.")
        raise typer.Exit(1)
    _render_conversation_import_summary(summary)


@cognition_import_app.command("status")
def cognition_import_status(
    batch_id: Annotated[str, typer.Argument(help="Import batch id.")],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", help="Show conversation-level import status."),
    ] = False,
) -> None:
    """Show daemon-owned conversation import and extraction status."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    response = _client_response_or_exit(
        DaemonClient(runtime.socket_path).request(
            {
                "type": "conversation_import_status",
                "batch_id": batch_id,
                "verbose": verbose,
            }
        )
    )
    status = response.get("status")
    if not isinstance(status, dict):
        console.print("Daemon response did not include import status.")
        raise typer.Exit(1)
    _render_conversation_import_status(status)
    if verbose:
        conversations = response.get("conversations")
        _render_conversation_import_status_details(
            conversations if isinstance(conversations, list) else []
        )


def _read_conversation_import_file(file: Path) -> str:
    try:
        size = file.stat().st_size
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if size > MAX_CONVERSATION_IMPORT_PAYLOAD_BYTES:
        console.print("Conversation import file exceeds the 50 MB conversation import limit.")
        raise typer.Exit(1)
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter("Import file must be UTF-8 encoded JSON.") from exc
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _read_deepseek_export_file(file: Path) -> str:
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise typer.BadParameter("DeepSeek export file must be UTF-8 encoded JSON.") from exc
    except OSError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _validate_converted_conversation_import_payload(
    payload_json: str,
    *,
    input_name: str,
) -> None:
    with tempfile.TemporaryDirectory(prefix="alpha-import-convert-") as tmp_dir:
        store = StateStore(Path(tmp_dir) / "alpha.db")
        store.initialize()
        try:
            ConversationImportService(store).import_payload(
                payload_json,
                input_name=input_name,
                dry_run=True,
            )
        except ConversationImportValidationFailed as exc:
            console.print("Converted payload failed normalized import validation.")
            _render_error_details({"details": [error.to_dict() for error in exc.errors]})
            raise typer.Exit(1) from exc


def _render_deepseek_conversion_summary(
    payload: dict[str, Any],
    *,
    output_name: str,
) -> None:
    conversations = payload.get("conversations")
    conversation_count = len(conversations) if isinstance(conversations, list) else 0
    message_count = 0
    if isinstance(conversations, list):
        for conversation in conversations:
            if isinstance(conversation, dict) and isinstance(conversation.get("messages"), list):
                message_count += len(conversation["messages"])

    rows = {
        "source_provider": str(payload.get("source_provider") or ""),
        "conversations": str(conversation_count),
        "messages": str(message_count),
        "output": output_name,
    }
    table = Table(title="DeepSeek Conversion")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
    typer.echo(
        "conversion "
        f"source_provider={rows['source_provider']} "
        f"conversations={rows['conversations']} "
        f"messages={rows['messages']} "
        f"output={rows['output']}"
    )


def _render_conversation_import_summary(summary: dict[str, Any]) -> None:
    table = Table(title="Conversation Import")
    table.add_column("Field")
    table.add_column("Value")
    rows = {
        "batch_id": _display_optional(summary.get("batch_id")),
        "source_provider": str(summary.get("source_provider") or ""),
        "dry_run": _display_bool(summary.get("dry_run")),
        "status": str(summary.get("status") or ""),
        "input_name": _display_optional(summary.get("input_name")),
        "conversations_seen": _display_count(summary.get("conversations_seen")),
        "messages_seen": _display_count(summary.get("messages_seen")),
        "conversations_created": _display_count(summary.get("conversations_created")),
        "conversations_reused": _display_count(summary.get("conversations_reused")),
        "messages_inserted": _display_count(summary.get("messages_inserted")),
        "messages_deduped": _display_count(summary.get("messages_deduped")),
        "background_cognition": "eligible",
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
    typer.echo(
        "import "
        f"batch_id={rows['batch_id']} "
        f"source_provider={rows['source_provider']} "
        f"dry_run={rows['dry_run']} "
        f"status={rows['status']} "
        f"conversations_seen={rows['conversations_seen']} "
        f"messages_seen={rows['messages_seen']} "
        f"conversations_created={rows['conversations_created']} "
        f"conversations_reused={rows['conversations_reused']} "
        f"messages_inserted={rows['messages_inserted']} "
        f"messages_deduped={rows['messages_deduped']} "
        f"background_cognition={rows['background_cognition']}"
    )


def _render_conversation_import_status(status: dict[str, Any]) -> None:
    table = Table(title="Conversation Import Status")
    table.add_column("Field")
    table.add_column("Value")
    rows = {
        "batch_id": str(status.get("batch_id") or ""),
        "source_provider": str(status.get("source_provider") or ""),
        "status": str(status.get("status") or ""),
        "conversations_seen": _display_count(status.get("conversations_seen")),
        "messages_seen": _display_count(status.get("messages_seen")),
        "conversations_created": _display_count(status.get("conversations_created")),
        "conversations_reused": _display_count(status.get("conversations_reused")),
        "messages_inserted": _display_count(status.get("messages_inserted")),
        "messages_deduped": _display_count(status.get("messages_deduped")),
        "extraction_pending": _display_count(status.get("extraction_pending")),
        "extraction_claimed": _display_count(status.get("extraction_claimed")),
        "extraction_processed": _display_count(status.get("extraction_processed")),
        "extraction_failed": _display_count(status.get("extraction_failed")),
        "extraction_skipped": _display_count(status.get("extraction_skipped")),
        "created_at": str(status.get("created_at") or ""),
        "updated_at": str(status.get("updated_at") or ""),
    }
    error_summary = status.get("error_summary")
    if error_summary:
        rows["error_summary"] = str(error_summary)
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
    typer.echo(
        "import_status "
        f"batch_id={rows['batch_id']} "
        f"source_provider={rows['source_provider']} "
        f"status={rows['status']} "
        f"messages_inserted={rows['messages_inserted']} "
        f"messages_deduped={rows['messages_deduped']} "
        f"extraction_pending={rows['extraction_pending']} "
        f"extraction_claimed={rows['extraction_claimed']} "
        f"extraction_processed={rows['extraction_processed']} "
        f"extraction_failed={rows['extraction_failed']} "
        f"extraction_skipped={rows['extraction_skipped']}"
    )


def _render_conversation_import_status_details(conversations: list[object]) -> None:
    table = Table(title="Conversation Import Details")
    table.add_column("External ID")
    table.add_column("Title")
    table.add_column("Session ID")
    table.add_column("Inserted", justify="right")
    table.add_column("Deduped", justify="right")
    table.add_column("Pending", justify="right")
    table.add_column("Processed", justify="right")
    table.add_column("Failed", justify="right")
    for item in conversations:
        if not isinstance(item, dict):
            continue
        table.add_row(
            str(item.get("external_conversation_id") or ""),
            str(item.get("title") or ""),
            str(item.get("session_id") or ""),
            _display_count(item.get("messages_inserted")),
            _display_count(item.get("messages_deduped")),
            _display_count(item.get("extraction_pending")),
            _display_count(item.get("extraction_processed")),
            _display_count(item.get("extraction_failed")),
        )
        typer.echo(
            "import_conversation "
            f"external_conversation_id={item.get('external_conversation_id') or ''} "
            f"session_id={item.get('session_id') or ''} "
            f"messages_inserted={_display_count(item.get('messages_inserted'))} "
            f"messages_deduped={_display_count(item.get('messages_deduped'))} "
            f"extraction_pending={_display_count(item.get('extraction_pending'))} "
            f"extraction_processed={_display_count(item.get('extraction_processed'))} "
            f"extraction_failed={_display_count(item.get('extraction_failed'))}"
        )
    console.print(table)


def _display_optional(value: object) -> str:
    if value is None or value == "":
        return "-"
    return str(value)


def _display_bool(value: object) -> str:
    return str(bool(value)).lower()


def _display_count(value: object) -> str:
    if isinstance(value, bool):
        return "0"
    if isinstance(value, int):
        return str(value)
    return "0"


@cognition_app.command("consolidate")
def cognition_consolidate(
    now: Annotated[
        bool,
        typer.Option("--now", help="Run one synchronous consolidation pass."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview worker reports without writing events."),
    ] = False,
) -> None:
    """Run the Phase 06 consolidation loop once."""

    if not now:
        raise typer.BadParameter("Only --now is supported by the v1 in-process scheduler.")
    config = load_config()
    consolidation_config = ConsolidationConfig(
        enabled=config.cognition_consolidation_enabled,
        interval_seconds=config.cognition_consolidation_interval_seconds,
        dry_run=dry_run,
    )
    if dry_run:
        with tempfile.TemporaryDirectory(prefix="alpha-consolidation-dry-run-") as tmp_dir:
            store = _dry_run_store(config, Path(tmp_dir))
            reports = _run_consolidation_once(store, consolidation_config)
    else:
        reports = _run_consolidation_once(_store(config), consolidation_config)
    table = Table(title="Cognition Consolidation")
    table.add_column("Worker")
    table.add_column("Inspected", justify="right")
    table.add_column("Emitted", justify="right")
    table.add_column("Status")
    for item in reports:
        table.add_row(
            item.worker,
            str(item.inspected),
            str(item.emitted),
            item.new_checkpoint.last_status,
        )
    console.print(table)
    typer.echo(
        "dry_run="
        f"{str(dry_run).lower()} workers={len(reports)} "
        f"emitted={sum(item.emitted for item in reports)}"
    )


def _run_consolidation_once(
    store: StateStore,
    consolidation_config: ConsolidationConfig,
) -> list[WorkerReport]:
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    scheduler = Scheduler(log, CheckpointStore(store))
    return ConsolidationLoop(
        scheduler=scheduler,
        log=log,
        projections=projections,
        config=consolidation_config,
    ).run_once()


def _dry_run_store(config: AlphaConfig, tmp_dir: Path) -> StateStore:
    dry_db = tmp_dir / "alpha-dry-run.db"
    if config.db_path.exists():
        shutil.copy2(config.db_path, dry_db)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(f"{config.db_path}{suffix}")
            if sidecar.exists():
                shutil.copy2(sidecar, Path(f"{dry_db}{suffix}"))
    store = StateStore(dry_db)
    store.initialize()
    return store


@goals_app.command("list")
def cognition_goals_list(
    active: Annotated[
        bool,
        typer.Option("--active", help="Show only active goals."),
    ] = False,
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Reserved for the single-subject runtime."),
    ] = None,
) -> None:
    """List Drive Loop goals."""

    del subject
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = GoalProjection(store, event_log=log, auto_rebuild=True)
    goals = projection.active() if active else projection.list_all()
    table = Table(title="Cognition Goals")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Priority", justify="right")
    table.add_column("Description")
    table.add_column("Last Drive")
    for goal in goals:
        table.add_row(
            str(goal.id),
            goal.status,
            str(goal.priority),
            goal.description,
            str(goal.last_drive_at or ""),
        )
    console.print(table)
    for goal in goals:
        typer.echo(
            f"goal={goal.id} status={goal.status} "
            f"priority={goal.priority} last_drive_at={goal.last_drive_at or ''}"
        )
    if not goals:
        typer.echo(f"goals=0 active={str(active).lower()}")


@goals_app.command("set")
def cognition_goals_set(
    description: Annotated[str, typer.Option("--description", help="Goal description.")],
    priority: Annotated[int, typer.Option("--priority", help="Higher priority runs first.")] = 0,
    target_outcome: Annotated[
        str,
        typer.Option("--target-outcome", help="Expected outcome for this goal."),
    ] = "",
) -> None:
    """Set a new active Drive Loop goal."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = GoalProjection(
        store,
        event_log=log,
        auto_rebuild=True,
        active_limit=config.cognition_drive_active_goal_limit,
    )
    registry = GoalRegistry(
        log,
        projection=projection,
        active_limit=config.cognition_drive_active_goal_limit,
    )
    try:
        event = registry.set_goal(
            description=description,
            target_outcome=target_outcome,
            priority=priority,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raw_goal = event.payload.get("goal")
    goal_id = str(raw_goal.get("id", "")) if isinstance(raw_goal, dict) else ""
    typer.echo(f"goal_set event_id={event.id} goal={goal_id}")


@goals_app.command("satisfy")
def cognition_goals_satisfy(
    goal_id: Annotated[str, typer.Argument(help="Goal id to mark satisfied.")],
    evidence: Annotated[str, typer.Option("--evidence", help="Evidence for satisfaction.")],
) -> None:
    """Mark a goal as satisfied."""

    projection, registry = _goal_registry_from_config()
    event = registry.satisfy(goal_id, evidence=evidence)
    goal = projection.get(goal_id)
    status = goal.status if goal is not None else "unknown"
    typer.echo(f"goal_satisfied event_id={event.id} goal={goal_id} status={status}")


@goals_app.command("abandon")
def cognition_goals_abandon(
    goal_id: Annotated[str, typer.Argument(help="Goal id to abandon.")],
    reason: Annotated[str, typer.Option("--reason", help="Reason for abandoning the goal.")],
) -> None:
    """Mark a goal as abandoned."""

    projection, registry = _goal_registry_from_config()
    event = registry.abandon(goal_id, reason=reason)
    goal = projection.get(goal_id)
    status = goal.status if goal is not None else "unknown"
    typer.echo(f"goal_abandoned event_id={event.id} goal={goal_id} status={status}")


@cognition_app.command("drive")
def cognition_drive(
    once: Annotated[
        bool,
        typer.Option("--once", help="Run one manual Drive Loop pass."),
    ] = False,
) -> None:
    """Run the Phase 10 Drive Loop once."""

    if not once:
        raise typer.BadParameter("Only --once is supported by the v1 Drive Loop CLI.")
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projections = ProjectionRegistry()
    projections.register(
        GoalProjection(
            store,
            event_log=log,
            auto_rebuild=True,
            active_limit=config.cognition_drive_active_goal_limit,
        )
    )
    emitter = EventEmitter(log)
    agent = AlphaAgent(
        store=store,
        llm_provider=build_provider(config),
        tool_registry=build_tool_registry(config),
        event_log=log,
    )
    report = DriveLoop(
        log=log,
        projections=projections,
        runtime_turn_runner=agent,
        emitter=emitter,
        config=DriveConfig(
            enabled=config.cognition_drive_enabled,
            interval_seconds=config.cognition_drive_interval_seconds,
            goal_cooldown_seconds=config.cognition_drive_goal_cooldown_seconds,
            active_goal_limit=config.cognition_drive_active_goal_limit,
        ),
    ).run_once(force=True)
    typer.echo(
        "drive "
        f"triggered={str(report.triggered).lower()} "
        f"dropped={str(report.dropped).lower()} "
        f"goal={report.selected_goal_id or ''} "
        f"reason={report.skipped_reason}"
    )


def _goal_registry_from_config() -> tuple[GoalProjection, GoalRegistry]:
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = GoalProjection(
        store,
        event_log=log,
        auto_rebuild=True,
        active_limit=config.cognition_drive_active_goal_limit,
    )
    return (
        projection,
        GoalRegistry(
            log,
            projection=projection,
            active_limit=config.cognition_drive_active_goal_limit,
        ),
    )


def main() -> None:
    """Typer entrypoint."""

    app()


if __name__ == "__main__":
    main()
