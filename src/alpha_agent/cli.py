"""Command line interface for Alpha Agent."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

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
    Situation,
    SituationId,
    ValueKind,
)
from alpha_agent.cognition.projections.belief import BeliefProjection
from alpha_agent.cognition.projections.goal import GoalProjection
from alpha_agent.cognition.projections.reflection import ReflectionProjection, target_to_parts
from alpha_agent.cognition.projections.registry import ProjectionRegistry
from alpha_agent.cognition.projections.strategy import StrategyProjection
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.reflectors.l3 import ReflectorL3
from alpha_agent.cognition.render import (
    CognitionView,
    DiffRenderer,
    EvidenceRenderer,
    GraphSnapshotRenderer,
    RenderBudget,
    render_counterpart_profile,
    wrap_system_reminder,
)
from alpha_agent.cognition.render.build_view import build_view
from alpha_agent.cognition.value.lens import default_value_lens, load_lens, save_lens
from alpha_agent.config import (
    AlphaConfig,
    default_config_path,
    load_config,
    read_config_value,
    set_config_value,
    write_default_config,
)
from alpha_agent.daemon.client import DaemonClient
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
from alpha_agent.llm.base import ChatMessage
from alpha_agent.llm.codex import CODEX_DEFAULT_MODEL
from alpha_agent.llm.deepseek import DEEPSEEK_DEFAULT_MODEL
from alpha_agent.llm.openai_compatible import OPENAI_COMPATIBLE_DEFAULT_MODEL
from alpha_agent.runtime.agent import AlphaAgent, default_runtime_system_message
from alpha_agent.runtime.session import new_session_id
from alpha_agent.runtime.session_context import (
    SYSTEM_REMINDER_CLOSE,
    SYSTEM_REMINDER_OPEN,
    SessionContextAssembler,
)
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
lens_app = typer.Typer(help="Subject value lens commands.")
goals_app = typer.Typer(help="Drive Loop goal commands.")
self_model_app = typer.Typer(help="Subject self-model commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(cognition_app, name="cognition")
cognition_app.add_typer(lens_app, name="lens")
cognition_app.add_typer(goals_app, name="goals")
cognition_app.add_typer(self_model_app, name="self-model")

DAEMON_START_TIMEOUT_SECONDS = 5.0
DAEMON_STOP_TIMEOUT_SECONDS = 5.0
DAEMON_START_POLL_INTERVAL_SECONDS = 0.1
CHAT_HISTORY_PREVIEW_LIMIT = 8
CHAT_HISTORY_MESSAGE_MAX_CHARS = 900
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


def _display_model(config: AlphaConfig) -> str:
    if config.llm_model:
        return config.llm_model
    if config.llm_provider == "deepseek":
        return f"{DEEPSEEK_DEFAULT_MODEL} (provider default)"
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
        content = _strip_system_reminder(content)
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


def _strip_system_reminder(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith(SYSTEM_REMINDER_OPEN) and stripped.endswith(SYSTEM_REMINDER_CLOSE):
        return stripped[len(SYSTEM_REMINDER_OPEN) : -len(SYSTEM_REMINDER_CLOSE)].strip()
    return stripped


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
    else:
        console.print("Daemon request failed.")
    raise typer.Exit(1)


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
        response = _client_response_or_exit(
            client.request(
                {
                    "type": "chat_turn",
                    "message": message,
                    "session_id": request_session_id,
                    "source_metadata": _source_metadata("chat"),
                }
            )
        )
        session_id = str(response.get("session_id") or request_session_id)
        turn_messages = (
            _displayable_chat_turn_messages(
                store,
                session_id,
                after_ordinal=before_ordinal,
            )
            if session_id == request_session_id
            else []
        )
        _render_chat_turn_messages(
            turn_messages,
            fallback_response=str(response.get("response", "")),
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
) -> None:
    """Print the runtime prompt preview and optional cognitive event trace."""

    config = load_config()
    store = _store(config)
    session_id = session or new_session_id()
    context = SessionContextAssembler(store).load(session_id)
    messages = [
        default_runtime_system_message(),
        *_debug_profile_context_messages(store, session_id),
        *context.chat_messages,
        {"role": "user", "content": message},
    ]
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


def _debug_profile_context_messages(store: StateStore, session_id: str) -> list[ChatMessage]:
    snapshot = store.get_session_profile_snapshot(session_id)
    if snapshot is None:
        return []
    return [
        {
            "role": "user",
            "content": wrap_system_reminder(render_counterpart_profile(snapshot.content)),
        }
    ]


@cognition_app.command("graph")
def cognition_graph(
    format: Annotated[
        str,
        typer.Option("--format", help="Graph format: mermaid or dot."),
    ] = "mermaid",
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Reserved for future multi-subject inspection."),
    ] = None,
) -> None:
    """Render a minimal active-belief cognition graph."""

    del subject
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    view = build_view(
        session_id="inspection",
        situation=Situation(id=SituationId("situation:cognition-graph")),
        projections=projections,
    )
    rendered = GraphSnapshotRenderer(format=format).render(
        view,
        RenderBudget(max_tokens=128),
        beliefs=projections.get_typed(BeliefProjection).list_active(),
    )
    console.print(str(rendered.payload), markup=False)


@cognition_app.command("diff")
def cognition_diff(
    turn_id_a: Annotated[str, typer.Argument(help="Earlier turn id.")],
    turn_id_b: Annotated[str, typer.Argument(help="Later turn id.")],
) -> None:
    """Render event-kind changes between two turns."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    view = _inspection_view(store)
    rendered = DiffRenderer(log, turn_id_a=turn_id_a, turn_id_b=turn_id_b).render(
        view,
        RenderBudget(max_tokens=256),
    )
    console.print(str(rendered.payload), markup=False)


@cognition_app.command("evidence")
def cognition_evidence(
    belief_id: Annotated[str, typer.Argument(help="Belief id to trace.")],
) -> None:
    """Render evidence events for one belief id."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    view = _inspection_view(store)
    rendered = EvidenceRenderer(log, belief_id=belief_id).render(
        view,
        RenderBudget(max_tokens=256),
    )
    console.print(str(rendered.payload), markup=False)


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
        context_foreground_max=config.cognition_consolidation_context_foreground_max,
        context_absorb_batch=config.cognition_consolidation_context_absorb_batch,
        context_summary_chars=config.cognition_consolidation_context_summary_chars,
        counterpart_digest_min_beliefs=config.cognition_consolidation_counterpart_digest_min_beliefs,
        counterpart_digest_min_new_beliefs=(
            config.cognition_consolidation_counterpart_digest_min_new_beliefs
        ),
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


def _inspection_view(store: StateStore) -> CognitionView:
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    return build_view(
        session_id="inspection",
        situation=Situation(id=SituationId("situation:cognition-inspection")),
        projections=projections,
    )


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


@self_model_app.callback(invoke_without_command=True)
def cognition_self_model(
    ctx: typer.Context,
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Reserved for the single agent subject."),
    ] = None,
) -> None:
    """Show the current subject self-model."""

    if ctx.invoked_subcommand is not None:
        return
    del subject
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    subject_value = SubjectProjection(log, store).current()
    record = subject_value.self_model.to_record()
    table = Table(title="Subject SelfModel")
    table.add_column("Field")
    table.add_column("Value")
    for key in sorted(record):
        value = record[key]
        table.add_row(key, _compact_record(value))
    console.print(table)
    for key in sorted(record):
        typer.echo(f"{key}={_compact_record(record[key])}")


@self_model_app.command("history")
def cognition_self_model_history(
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Reserved for the single agent subject."),
    ] = None,
    last: Annotated[int, typer.Option("--last", min=1, help="Number of updates.")] = 10,
) -> None:
    """List recent self-model updates."""

    del subject
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    events = list(log.iter(kinds=[CognitiveEventKind.SELF_MODEL_UPDATED]))[-last:]
    table = Table(title="SelfModel History")
    table.add_column("Timestamp")
    table.add_column("Event")
    table.add_column("Changed Fields")
    for event in events:
        diff = event.payload.get("diff")
        changed = ",".join(sorted(diff)) if isinstance(diff, dict) else ""
        table.add_row(str(event.timestamp), str(event.id), changed)
    console.print(table)
    for event in events:
        diff = event.payload.get("diff")
        changed = ",".join(sorted(diff)) if isinstance(diff, dict) else ""
        typer.echo(f"self_model_updated event_id={event.id} changed={changed}")
    if not events:
        typer.echo("self_model_history=0")


@cognition_app.command("reflect-l3")
def cognition_reflect_l3(
    once: Annotated[
        bool,
        typer.Option("--once", help="Run one manual Reflector L3 pass."),
    ] = False,
    subject: Annotated[
        str | None,
        typer.Option("--subject", help="Reserved for the single agent subject."),
    ] = None,
) -> None:
    """Run the Phase 11 L3 self-model reflector once."""

    del subject
    if not once:
        raise typer.BadParameter("Only --once is supported by the v1 Reflector L3 CLI.")
    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projections = default_projection_registry(log)
    report = ReflectorL3().run_once(log=log, projections=projections)
    typer.echo(
        "reflect_l3 "
        f"emitted={report.emitted} "
        f"status={report.new_checkpoint.last_status} "
        f"notes={','.join(report.notes)}"
    )


def _compact_record(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "{}"
        return ",".join(f"{key}:{value[key]}" for key in sorted(value))
    if isinstance(value, list):
        if not value:
            return "[]"
        return ",".join(str(item) for item in value)
    return str(value)


@lens_app.command("show")
def cognition_lens_show(
    subject: Annotated[
        str | None,
        typer.Argument(help="Subject id. Defaults to the single agent subject."),
    ] = None,
) -> None:
    """Show the current subject value lens."""

    config = load_config()
    store = _store(config)
    lens = load_lens(store, subject or "agent:self")
    table = Table(title="Subject Value Lens")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Subject", subject or "agent:self")
    table.add_row("Priority", ", ".join(value.value for value in lens.priorities))
    table.add_row(
        "Sensitivity",
        ", ".join(
            f"{value.value}={lens.sensitivity.get(value, 1.0):.3f}"
            for value in lens.priorities
        ),
    )
    console.print(table)
    typer.echo("priority=" + ",".join(value.value for value in lens.priorities))


@lens_app.command("set")
def cognition_lens_set(
    subject: Annotated[
        str | None,
        typer.Argument(help="Subject id. Defaults to the single agent subject."),
    ] = None,
    priority: Annotated[
        str,
        typer.Option(
            "--priority",
            help="Comma-separated ValueKind priority, e.g. safety,honesty,efficiency.",
        ),
    ] = "",
) -> None:
    """Set the subject value lens priority order."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    emitter = EventEmitter(log)
    current = load_lens(store, subject or "agent:self")
    priorities = _parse_lens_priority(priority) if priority else default_value_lens().priorities
    updated = current.__class__(
        priorities=priorities,
        weights=current.weights,
        sensitivity=current.sensitivity,
    )
    event = save_lens(
        store,
        emitter,
        updated,
        subject_id=subject or "agent:self",
        trigger="cli lens set",
        before=current,
    )
    typer.echo(f"value_lens_shifted event_id={event.id}")
    saved = load_lens(store, subject or "agent:self")
    typer.echo("priority=" + ",".join(value.value for value in saved.priorities))


def _parse_lens_priority(raw: str) -> list[ValueKind]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise typer.BadParameter("--priority must contain at least one ValueKind")
    try:
        return [ValueKind(value) for value in values]
    except ValueError as exc:
        allowed = ", ".join(value.value for value in ValueKind)
        raise typer.BadParameter(f"unknown ValueKind; allowed: {allowed}") from exc


@cognition_app.command("strategies")
def cognition_strategies(
    active: Annotated[
        bool,
        typer.Option("--active", help="Show active strategies."),
    ] = True,
    all_: Annotated[
        bool,
        typer.Option("--all", help="Show all strategies."),
    ] = False,
) -> None:
    """List strategy overrides."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = StrategyProjection(store, event_log=log, auto_rebuild=True)
    rows = projection.list_all() if all_ else projection.active()
    table = Table(title="Cognition Strategies")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Domains")
    table.add_column("Counterpart")
    table.add_column("Valid Until")
    for strategy in rows:
        table.add_row(
            str(strategy.id),
            strategy.name,
            ",".join(strategy.target_domains),
            strategy.for_counterpart.id if strategy.for_counterpart else "global",
            str(strategy.valid_until),
        )
    console.print(table)
    for strategy in rows:
        typer.echo(
            f"strategy={strategy.id} name={strategy.name} "
            f"domains={','.join(strategy.target_domains)}"
        )
    if not rows:
        typer.echo("strategies=0 active=" + str(active and not all_).lower())


@cognition_app.command("strategy-expire")
def cognition_strategy_expire(
    strategy_id: Annotated[str, typer.Argument(help="Strategy id to expire.")],
) -> None:
    """Manually expire a strategy override."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = StrategyProjection(store, event_log=log, auto_rebuild=True)
    emitter = EventEmitter(log)
    event = emitter.emit(
        CognitiveEventKind.STRATEGY_EXPIRED,
        payload={"strategy_id": strategy_id, "reason": "manual"},
    )
    projection.apply(event)
    typer.echo(f"strategy_expired event_id={event.id} strategy={strategy_id}")


@cognition_app.command("reflections")
def cognition_reflections(
    severity: Annotated[
        str | None,
        typer.Option("--severity", help="Filter by severity, e.g. info, warning, blocker."),
    ] = None,
    last: Annotated[
        int,
        typer.Option("--last", min=1, help="Maximum number of recent reflections to show."),
    ] = 20,
) -> None:
    """List recent L1 reflection findings."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    projection = ReflectionProjection(store, event_log=log, auto_rebuild=True)
    rows = projection.list_recent(last=last, severity=severity)
    table = Table(title="Cognition Reflections")
    table.add_column("Created")
    table.add_column("Severity")
    table.add_column("Kind")
    table.add_column("Target")
    table.add_column("Finding")
    for item in rows:
        target_kind, target_id = target_to_parts(item.target)
        table.add_row(
            str(item.created_at),
            str(item.severity),
            str(item.kind),
            f"{target_kind}:{target_id}",
            str(item.finding),
        )
    console.print(table)
    for item in rows:
        target_kind, target_id = target_to_parts(item.target)
        typer.echo(
            f"{item.created_at} severity={item.severity} kind={item.kind} "
            f"target={target_kind}:{target_id} finding={item.finding}"
        )
    if not rows:
        console.print("(none)", markup=False)


def main() -> None:
    """Typer entrypoint."""

    app()


if __name__ == "__main__":
    main()
