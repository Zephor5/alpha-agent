"""Command line interface for Alpha Agent."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from alpha_agent.cognition.controller import default_projection_registry
from alpha_agent.cognition.emitter import EventEmitter
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.loops import (
    CheckpointStore,
    ConsolidationConfig,
    ConsolidationLoop,
    Scheduler,
)
from alpha_agent.cognition.models import (
    CognitiveEvent,
    CognitiveEventKind,
    ContextWindow,
    Instant,
    Situation,
    SituationId,
    Subject,
    ThreadId,
    ValueKind,
    situation_ref,
    subject_ref,
)
from alpha_agent.cognition.projections.reflection import ReflectionProjection, target_to_parts
from alpha_agent.cognition.projections.strategy import StrategyProjection
from alpha_agent.cognition.projections.subject import SubjectProjection
from alpha_agent.cognition.render import (
    CognitionView,
    DiffRenderer,
    EvidenceRenderer,
    GraphSnapshotRenderer,
    RenderBudget,
    TextChatRenderer,
    conversation_message_to_chat,
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
lens_app = typer.Typer(help="Subject value lens commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")
app.add_typer(cognition_app, name="cognition")
cognition_app.add_typer(lens_app, name="lens")

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
    renderer: Annotated[
        str,
        typer.Option("--renderer", help="Renderer name. Currently supports text_chat."),
    ] = "text_chat",
) -> None:
    """Print a renderer prompt preview and optional cognitive event trace."""

    config = load_config()
    store = _store(config)
    session_id = session or new_session_id()
    context = SessionContextManager(
        store,
        recent_tail_messages=config.context_recent_tail_messages,
    ).load(session_id)
    if renderer != TextChatRenderer.name:
        raise typer.BadParameter("supported renderer: text_chat")
    view = _debug_prompt_view(
        session_id=session_id,
        message=message,
        chat_history=[conversation_message_to_chat(item) for item in context.messages],
    )
    rendered = TextChatRenderer().render(view, RenderBudget())
    messages = rendered.payload
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


def _debug_prompt_view(
    session_id: str,
    message: str,
    chat_history: list[dict[str, Any]],
) -> CognitionView:
    situation = Situation(id=SituationId("situation:debug-prompt"))
    thread_id = ThreadId.from_session(session_id)
    subject_value = Subject()
    window = ContextWindow(
        thread_id=thread_id,
        counterpart=None,
        foreground=[],
        background=None,
        recalled=[],
        recent_judgments=[],
        matched_procedures=[],
        subject_at=subject_ref(subject_value.id),
        situation_at=situation_ref(situation.id),
        assembled_at=Instant(""),
    )
    return CognitionView(
        subject=subject_value,
        counterpart=None,
        situation=situation,
        window=window,
        assembled_at=Instant(""),
        current_query=message,
        chat_history=chat_history,
    )


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
    subject_value = projections.get_typed(SubjectProjection).current()
    thread_id = ThreadId.cognition(subject_value.id, "inspection")
    view = build_view(
        thread_id=thread_id,
        situation=Situation(id=SituationId("situation:cognition-graph")),
        projections=projections,
    )
    rendered = GraphSnapshotRenderer(format=format).render(view, RenderBudget(max_tokens=128))
    console.print(str(rendered.payload), markup=False)


@cognition_app.command("diff")
def cognition_diff(
    tick_id_a: Annotated[str, typer.Argument(help="Earlier tick id.")],
    tick_id_b: Annotated[str, typer.Argument(help="Later tick id.")],
) -> None:
    """Render event-kind changes between two ticks."""

    config = load_config()
    store = _store(config)
    log = SQLiteEventLog(store)
    view = _inspection_view(store)
    rendered = DiffRenderer(log, tick_id_a=tick_id_a, tick_id_b=tick_id_b).render(
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
        judgment_repeat_window=config.cognition_consolidation_judgment_repeat_window,
        judgment_repeat_threshold=config.cognition_consolidation_judgment_repeat_threshold,
        procedure_success_threshold=config.cognition_consolidation_procedure_success_threshold,
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
):
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
    subject_value = projections.get_typed(SubjectProjection).current()
    return build_view(
        thread_id=ThreadId.cognition(subject_value.id, "inspection"),
        situation=Situation(id=SituationId("situation:cognition-inspection")),
        projections=projections,
    )


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
    table.add_column("Stages")
    table.add_column("Counterpart")
    table.add_column("Valid Until")
    for strategy in rows:
        table.add_row(
            str(strategy.id),
            strategy.name,
            ",".join(strategy.target_stages),
            strategy.for_counterpart.id if strategy.for_counterpart else "global",
            str(strategy.valid_until),
        )
    console.print(table)
    for strategy in rows:
        typer.echo(
            f"strategy={strategy.id} name={strategy.name} "
            f"stages={','.join(strategy.target_stages)}"
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
