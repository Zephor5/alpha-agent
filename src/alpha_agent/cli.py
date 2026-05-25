"""Command line interface for Alpha Agent."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import CognitiveEvent
from alpha_agent.cognition.projections.reflection import ReflectionProjection, target_to_parts
from alpha_agent.config import (
    AlphaConfig,
    default_config_path,
    load_config,
    read_config_value,
    set_config_value,
    write_default_config,
)
from alpha_agent.daemon.client import DaemonClient
from alpha_agent.daemon.manager import initialize_store
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
from alpha_agent.llm.openai_compatible import OPENAI_COMPATIBLE_DEFAULT_MODEL
from alpha_agent.runtime.prompt_builder import PromptBuilder
from alpha_agent.runtime.session import new_session_id
from alpha_agent.runtime.session_context import SessionContextManager
from alpha_agent.skills.manager import SkillManager
from alpha_agent.state.store import StateStore

console = Console()
app = typer.Typer(help="Alpha Agent cognition runtime.")
skills_app = typer.Typer(help="Skill commands.")
debug_app = typer.Typer(help="Debug commands.")
gateway_app = typer.Typer(help="Gateway operational commands.")
config_app = typer.Typer(help="Configuration commands.")
daemon_app = typer.Typer(help="Daemon runtime commands.")
cognition_app = typer.Typer(help="Cognition inspection commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(cognition_app, name="cognition")

DAEMON_START_TIMEOUT_SECONDS = 5.0
DAEMON_START_POLL_INTERVAL_SECONDS = 0.1


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
        "context_recent_tail_messages": str(config.context_recent_tail_messages),
    }
    if config.llm_provider in {"openai-compatible", "openai", "compatible"}:
        rows["compatible_base_url"] = config.compatible_base_url or ""
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(table)
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
    process, log_path = _spawn_daemon_background(runtime)
    try:
        status = _wait_for_daemon_running(client, process)
    except (RuntimeError, TimeoutError) as exc:
        console.print(str(exc))
        console.print(f"Daemon log: {log_path}")
        raise typer.Exit(1) from exc
    pid = status.get("pid") or process.pid
    console.print(f"Daemon started with PID {pid}.")
    console.print(f"Daemon log: {log_path}")


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
    session_id = session or new_session_id()
    console.print(f"Session: {session_id}")
    while True:
        try:
            message = typer.prompt("You")
        except (EOFError, KeyboardInterrupt):
            break
        if message.strip().lower() in {"/exit", "/quit"}:
            break
        response = _client_response_or_exit(
            client.request(
                {
                    "type": "chat_turn",
                    "message": message,
                    "session_id": session_id,
                    "source_metadata": _source_metadata("chat"),
                }
            )
        )
        session_id = str(response.get("session_id") or session_id)
        console.print(str(response.get("response", "")))


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
    """Print a baseline prompt preview and optional cognitive event trace."""

    config = load_config()
    store = _store(config)
    session_id = session or new_session_id()
    context = SessionContextManager(
        store,
        recent_tail_messages=config.context_recent_tail_messages,
    ).load(session_id)
    messages = PromptBuilder().build(message, context.messages)
    for index, prompt_message in enumerate(messages, start=1):
        role = prompt_message["role"]
        content = prompt_message.get("content") or ""
        console.print(f"Message {index} [{role}]\n{content}", markup=False)
    if trace:
        _render_cognitive_trace(store, session_id)


def _render_cognitive_trace(store: StateStore, session_id: str) -> None:
    all_events = list(SQLiteEventLog(store).iter())
    tick_ids = {
        str(event.payload["tick_id"])
        for event in all_events
        if _event_belongs_to_session(event, session_id) and "tick_id" in event.payload
    }
    events = [
        event
        for event in all_events
        if event.payload.get("tick_id") is not None
        and str(event.payload.get("tick_id")) in tick_ids
    ]
    console.print("Cognitive Trace", markup=False)
    if not events:
        console.print("(none)", markup=False)
        return
    for event in events[-20:]:
        tick_id = event.payload.get("tick_id", "-")
        parents = ",".join(str(parent) for parent in event.causal_parents) or "-"
        console.print(
            f"{event.timestamp} kind={event.kind.value} tick_id={tick_id} "
            f"id={event.id} parents={parents}",
            markup=False,
        )


def _event_belongs_to_session(event: CognitiveEvent, session_id: str) -> bool:
    raw_thread = event.payload.get("thread_id")
    return isinstance(raw_thread, dict) and raw_thread.get("key") == f"session:{session_id}"


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
