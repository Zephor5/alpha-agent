"""Command line interface for Alpha Agent."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

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
from alpha_agent.gateway.models import ConversationSource
from alpha_agent.gateway.status import gateway_tables_available
from alpha_agent.llm.codex import CODEX_DEFAULT_MODEL
from alpha_agent.llm.deepseek import DEEPSEEK_DEFAULT_MODEL
from alpha_agent.llm.openai_compatible import OPENAI_COMPATIBLE_DEFAULT_MODEL
from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.review import MemoryReviewService, edit_candidate
from alpha_agent.memory.store import MemoryStore
from alpha_agent.runtime.prompt_builder import PromptBuilder
from alpha_agent.runtime.session import new_session_id
from alpha_agent.runtime.session_context import SessionContextManager

console = Console()
app = typer.Typer(help="Alpha Agent personal memory runtime.")
memory_app = typer.Typer(help="Memory inspection and consolidation commands.")
skills_app = typer.Typer(help="Procedural skill commands.")
debug_app = typer.Typer(help="Debug commands.")
gateway_app = typer.Typer(help="Gateway operational commands.")
config_app = typer.Typer(help="Configuration commands.")
daemon_app = typer.Typer(help="Daemon runtime commands.")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")
app.add_typer(daemon_app, name="daemon")

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


def _store(config: AlphaConfig) -> MemoryStore:
    return initialize_store(config)


def _initialize_gateway(config: AlphaConfig) -> int:
    store = _store(config)
    return len(ProceduralMemoryManager(store).load_builtin_skills())


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


def _render_candidates(candidates: list[Any]) -> None:
    table = Table(title="Memory Review Candidates")
    table.add_column("Candidate", justify="right")
    table.add_column("Type")
    table.add_column("Content")
    table.add_column("Subject")
    table.add_column("Predicate")
    table.add_column("Object")
    table.add_column("Salience", justify="right")
    table.add_column("Confidence", justify="right")
    for index, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            candidate.type,
            candidate.content,
            candidate.subject or "",
            candidate.predicate or "",
            candidate.object or "",
            f"{candidate.salience:.2f}",
            f"{candidate.confidence:.2f}",
        )
    console.print(table)
    for index, candidate in enumerate(candidates, start=1):
        console.print(
            "Candidate "
            f"{index}: type={candidate.type} content={candidate.content} "
            f"subject={candidate.subject or ''} predicate={candidate.predicate or ''} "
            f"object={candidate.object or ''} salience={candidate.salience:.2f} "
            f"confidence={candidate.confidence:.2f}"
        )


def _memory_access_rows(
    store: MemoryStore,
    *,
    query: str,
    retrieved_ids: dict[str, list[str]],
) -> list[dict[str, Any]]:
    ids_by_type = {
        memory_type: ids
        for memory_type, ids in retrieved_ids.items()
        if memory_type != "working" and ids
    }
    if not ids_by_type:
        return []
    rows: list[dict[str, Any]] = []
    with store.connect() as conn:
        for memory_type, memory_ids in ids_by_type.items():
            placeholders = ",".join("?" for _ in memory_ids)
            query_rows = conn.execute(
                f"""
                SELECT memory_id, memory_type, score, accessed_at
                FROM memory_access_log
                WHERE query = ?
                  AND memory_type = ?
                  AND memory_id IN ({placeholders})
                ORDER BY accessed_at DESC
                """,
                (query, memory_type, *memory_ids),
            ).fetchall()
            seen: set[str] = set()
            for row in query_rows:
                memory_id = str(row["memory_id"])
                if memory_id in seen:
                    continue
                seen.add(memory_id)
                access_count = _memory_access_count(conn, str(row["memory_type"]), memory_id)
                rows.append(
                    {
                        "memory_id": memory_id,
                        "memory_type": str(row["memory_type"]),
                        "retrieval_score": float(row["score"]),
                        "access_count": access_count,
                        "accessed_at": str(row["accessed_at"]),
                    }
                )
    order = {
        (memory_type, memory_id): index
        for memory_type, ids in ids_by_type.items()
        for index, memory_id in enumerate(ids)
    }
    rows.sort(key=lambda row: order.get((row["memory_type"], row["memory_id"]), 0))
    return rows


def _memory_access_count(conn: Any, memory_type: str, memory_id: str) -> int:
    if memory_type == "episodic":
        row = conn.execute(
            "SELECT access_count FROM episodic_memories WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return int(row["access_count"]) if row else 0
    return 0


def _source_from_gateway_session(store: MemoryStore, session_id: str) -> ConversationSource | None:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT * FROM gateway_session_mappings
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        memory_scope = json.loads(row["memory_scope"] or "{}")
    except json.JSONDecodeError:
        memory_scope = {}
    return ConversationSource(
        platform=row["platform"],
        chat_id=row["chat_id"],
        chat_type=row["chat_type"],
        user_id=row["user_id"],
        user_name=memory_scope.get("user_name"),
        thread_id=row["thread_id"],
        metadata=memory_scope.get("external_metadata", {}),
    )


def _merge_source_context(
    base: ConversationSource | None,
    *,
    session_id: str,
    platform: str | None,
    chat_id: str | None,
    chat_type: str | None,
    user_id: str | None,
    user_name: str | None,
    thread_id: str | None,
    message_id: str | None,
) -> ConversationSource | None:
    if base is None and not any(
        value is not None
        for value in (platform, chat_id, chat_type, user_id, user_name, thread_id, message_id)
    ):
        return None
    return ConversationSource(
        platform=platform or (base.platform if base else "debug"),
        chat_id=chat_id or (base.chat_id if base else session_id),
        chat_type=chat_type or (base.chat_type if base else "dm"),
        user_id=user_id or (base.user_id if base else "debug-user"),
        user_name=user_name if user_name is not None else (base.user_name if base else None),
        thread_id=thread_id if thread_id is not None else (base.thread_id if base else None),
        message_id=message_id,
        metadata=dict(base.metadata) if base else {},
    )


def _render_source_context(source: ConversationSource | None) -> None:
    if source is None:
        return
    table = Table(title="Gateway Source Context")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("platform", source.platform)
    table.add_row("chat_id", source.chat_id)
    table.add_row("chat_type", str(source.chat_type))
    table.add_row("user_id", source.user_id)
    table.add_row("user_name", source.user_name or "")
    table.add_row("thread_id", source.thread_id or "")
    table.add_row("message_id", source.message_id or "")
    console.print(table)


@app.command("init")
def init_command() -> None:
    """Initialize the local data directory and SQLite database."""

    config = load_config()
    wrote_config = write_default_config()
    store = _store(config)
    loaded = ProceduralMemoryManager(store).load_builtin_skills()
    console.print(f"Initialized Alpha Agent database at [bold]{config.db_path}[/bold]")
    if wrote_config:
        console.print(f"Created config file at [bold]{default_config_path()}[/bold]")
    else:
        console.print(f"Config file already exists at [bold]{default_config_path()}[/bold]")
    console.print(f"Loaded {len(loaded)} builtin skills")


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
    table.add_row("config_path", str(default_config_path()))
    table.add_row("db_path", str(config.db_path))
    table.add_row("log_dir", str(config.log_dir))
    table.add_row("gateway_status_path", str(config.gateway_status_path))
    table.add_row("daemon_socket_path", str(config.daemon_socket_path))
    table.add_row("daemon_status_path", str(config.daemon_status_path))
    table.add_row("llm_provider", config.llm_provider)
    if config.llm_provider in {"openai-compatible", "openai", "compatible"}:
        table.add_row("compatible_base_url", config.compatible_base_url or "")
    table.add_row("llm_model", _display_model(config))
    table.add_row("llm_debug_logging", str(config.llm_debug_logging).lower())
    table.add_row("retrieval_limit", str(config.retrieval_limit))
    table.add_row("context_max_prompt_tokens", str(config.context_max_prompt_tokens))
    table.add_row(
        "context_compression_threshold_ratio",
        str(config.context_compression_threshold_ratio),
    )
    table.add_row("context_recent_tail_messages", str(config.context_recent_tail_messages))
    table.add_row("context_min_summary_tokens", str(config.context_min_summary_tokens))
    table.add_row("context_max_summary_tokens", str(config.context_max_summary_tokens))
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
    """Show gateway adapter status from the daemon runtime."""

    config = load_config()
    runtime = daemon_runtime_config(config)
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


@gateway_app.command("doctor")
def gateway_doctor() -> None:
    """Initialize gateway prerequisites and report local operational readiness."""

    config = load_config()
    runtime = gateway_runtime_config(config)
    ensure_gateway_runtime_files(runtime)
    loaded_skills = _initialize_gateway(config)
    tables = gateway_tables_available(config.db_path)
    adapter_names = configured_adapter_names()
    append_gateway_log(
        runtime.log_paths["gateway.log"],
        event="gateway.doctor",
        message="Gateway doctor completed.",
        metadata={
            "adapter_count": len(adapter_names),
            "gateway_tables": tables,
            "provider": config.llm_provider,
        },
    )

    table = Table(title="Gateway Doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("DB path", str(config.db_path))
    table.add_row(
        "Gateway tables",
        "\n".join(f"{name}: {'available' if ok else 'missing'}" for name, ok in tables.items()),
    )
    table.add_row("Log dir", str(runtime.log_dir))
    table.add_row(
        "Log paths",
        "\n".join(str(path) for path in runtime.log_paths.values()),
    )
    table.add_row("Provider", config.llm_provider)
    table.add_row("Builtin skills", str(loaded_skills))
    table.add_row(
        "Adapters",
        ", ".join(adapter_names) if adapter_names else "No real platform adapters configured",
    )
    console.print(table)
    typer.echo(f"DB path: {config.db_path}")
    typer.echo(f"Log dir: {runtime.log_dir}")
    typer.echo(f"Status path: {runtime.status_path}")
    for name, path in runtime.log_paths.items():
        typer.echo(f"{name}: {path}")


@app.command()
def ask(message: Annotated[str, typer.Argument(help="Message to send to the agent.")]) -> None:
    """Send a single-turn ask to the daemon."""

    config = load_config()
    runtime = daemon_runtime_config(config)
    response = _client_response_or_exit(
        DaemonClient(runtime.socket_path).request(
            {
                "type": "ask",
                "message": message,
                "session_id": None,
                "source_metadata": {"channel": "cli", "command": "ask"},
            }
        )
    )
    console.print(str(response.get("response", "")))


@app.command()
def chat(
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Existing session id to continue."),
    ] = None,
) -> None:
    """Start an interactive chat session."""

    config = load_config()
    session_id = session or new_session_id()
    client = DaemonClient(daemon_runtime_config(config).socket_path)
    console.print(f"[dim]Session: {session_id}[/dim]")
    console.print("[dim]Type /exit to quit, /consolidate to consolidate memory.[/dim]")
    while True:
        try:
            user_message = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user_message:
            continue
        if user_message in {"/exit", "/quit"}:
            break
        if user_message == "/consolidate":
            response = _client_response_or_exit(client.request({"type": "consolidate_memory"}))
            console.print(str(response.get("response", "")))
            continue
        response = _client_response_or_exit(
            client.request(
                {
                    "type": "chat_turn",
                    "message": user_message,
                    "session_id": session_id,
                    "source_metadata": {"channel": "cli", "command": "chat"},
                }
            )
        )
        if isinstance(response.get("session_id"), str):
            session_id = str(response["session_id"])
        console.print(f"[bold green]alpha>[/bold green] {response.get('response', '')}")


@memory_app.command("list")
def memory_list(limit: Annotated[int, typer.Option("--limit", "-n")] = 20) -> None:
    """Show recent memories."""

    config = load_config()
    store = _store(config)
    table = Table(title="Recent Memories")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Content")
    for semantic_memory in store.list_semantic_memories(limit):
        table.add_row("semantic", semantic_memory.id, semantic_memory.content)
    for episodic_memory in store.list_episodic_memories(limit):
        table.add_row("episodic", episodic_memory.id, episodic_memory.summary)
    for procedural_memory in store.list_procedural_memories(limit):
        table.add_row("procedural", procedural_memory.id, procedural_memory.name)
    console.print(table)


@memory_app.command()
def search(
    query: Annotated[str, typer.Argument(help="Query for non-vector memory search.")],
) -> None:
    """Search memories using non-vector retrieval."""

    config = load_config()
    store = _store(config)
    context = MemoryRetriever(store).retrieve_context(
        query=query,
        session_id="memory-search",
        limit=config.retrieval_limit,
        record_access=False,
    )
    table = Table(title=f"Memory search: {query}")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Content")
    for semantic_memory in context.semantic_memories:
        table.add_row("semantic", semantic_memory.id, semantic_memory.content)
    for episodic_memory in context.episodic_memories:
        table.add_row("episodic", episodic_memory.id, episodic_memory.summary)
    for procedural_memory in context.procedural_memories:
        table.add_row("procedural", procedural_memory.id, procedural_memory.name)
    console.print(table)


@memory_app.command()
def consolidate() -> None:
    """Run manual memory consolidation."""

    config = load_config()
    store = _store(config)
    report = ConsolidationService(store).consolidate()
    console.print(report.render())


@memory_app.command("review")
def memory_review(
    message: Annotated[str, typer.Argument(help="Message to extract memory candidates from.")],
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="Session id for approved review source events."),
    ] = "memory-review",
    approve_all: Annotated[
        bool,
        typer.Option("--approve-all", help="Store all extracted candidates."),
    ] = False,
    approve: Annotated[
        list[int] | None,
        typer.Option("--approve", help="1-based candidate index to store; can be repeated."),
    ] = None,
    reject_all: Annotated[
        bool,
        typer.Option("--reject-all", help="Reject all extracted candidates without storing."),
    ] = False,
    reject: Annotated[
        list[int] | None,
        typer.Option("--reject", help="1-based candidate index to skip; can be repeated."),
    ] = None,
    candidate_index: Annotated[
        int,
        typer.Option("--candidate", help="1-based candidate index to edit and approve."),
    ] = 1,
    edit_content: Annotated[
        str | None,
        typer.Option("--edit-content", help="Replacement content for the edited candidate."),
    ] = None,
    edit_subject: Annotated[
        str | None,
        typer.Option("--edit-subject", help="Replacement semantic subject."),
    ] = None,
    edit_predicate: Annotated[
        str | None,
        typer.Option("--edit-predicate", help="Replacement semantic predicate."),
    ] = None,
    edit_object: Annotated[
        str | None,
        typer.Option("--edit-object", help="Replacement semantic object."),
    ] = None,
) -> None:
    """Preview extracted memory candidates and store only explicit approvals."""

    edit_requested = any(
        value is not None for value in (edit_content, edit_subject, edit_predicate, edit_object)
    )
    approve_indices = set(approve or [])
    reject_indices = set(reject or [])
    if reject_all and (approve_all or approve_indices or reject_indices or edit_requested):
        raise typer.BadParameter("--reject-all cannot be combined with other review actions.")
    if approve_all and approve_indices:
        raise typer.BadParameter("Use either --approve-all or --approve, not both.")
    if approve_indices & reject_indices:
        raise typer.BadParameter("A candidate cannot be both approved and rejected.")

    config = load_config()
    service = MemoryReviewService(_store(config))
    candidates = service.preview(message)
    _render_candidates(candidates)
    if not candidates:
        console.print("No memory candidates extracted.")
        return
    if reject_all:
        console.print(f"Rejected {len(candidates)} candidate(s).")
        return

    valid_indices = set(range(1, len(candidates) + 1))
    requested_indices = approve_indices | reject_indices
    if requested_indices - valid_indices:
        raise typer.BadParameter("Candidate indexes must refer to extracted candidates.")

    reviewed_candidates = list(candidates)
    if edit_requested:
        if candidate_index < 1 or candidate_index > len(candidates):
            raise typer.BadParameter("--candidate must refer to an extracted candidate.")
        reviewed_candidates[candidate_index - 1] = edit_candidate(
            candidates[candidate_index - 1],
            content=edit_content,
            subject=edit_subject,
            predicate=edit_predicate,
            object_value=edit_object,
        )
        console.print(f"Edited candidate {candidate_index}.")
        if not approve_all and not approve_indices:
            approve_indices.add(candidate_index)

    if approve_all:
        selected_indices = valid_indices - reject_indices
    else:
        selected_indices = approve_indices - reject_indices
    if reject_indices:
        console.print(f"Rejected {len(reject_indices)} candidate(s).")
    if selected_indices:
        selected = [
            candidate
            for index, candidate in enumerate(reviewed_candidates, start=1)
            if index in selected_indices
        ]
        stored = service.approve(message=message, session_id=session, candidates=selected)
        console.print(f"Approved {len(stored)} candidate(s).")
        return
    console.print(
        "Preview only; no candidates stored. Use --approve-all, --reject-all, or edit flags."
    )


@memory_app.command()
def stats() -> None:
    """Show counts by memory type."""

    config = load_config()
    store = _store(config)
    table = Table(title="Memory Stats")
    table.add_column("Type")
    table.add_column("Count", justify="right")
    for key, count in store.stats().items():
        table.add_row(key, str(count))
    console.print(table)


@skills_app.command("list")
def skills_list() -> None:
    """List stored procedural memories."""

    config = load_config()
    store = _store(config)
    table = Table(title="Skills")
    table.add_column("Name")
    table.add_column("Trigger")
    table.add_column("Description")
    for memory in store.list_procedural_memories(limit=100):
        table.add_row(memory.name, memory.trigger, memory.description)
    console.print(table)


@debug_app.command()
def prompt(
    message: Annotated[str, typer.Argument(help="Message to build a prompt for.")],
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="Session id whose session context should be used."),
    ] = "debug",
    platform: Annotated[
        str | None,
        typer.Option("--platform", help="Gateway platform for source context."),
    ] = None,
    chat_id: Annotated[
        str | None,
        typer.Option("--chat-id", help="Gateway chat id for source context."),
    ] = None,
    chat_type: Annotated[
        str | None,
        typer.Option("--chat-type", help="Gateway chat type for source context."),
    ] = None,
    user_id: Annotated[
        str | None,
        typer.Option("--user-id", help="Gateway user id for source context."),
    ] = None,
    user_name: Annotated[
        str | None,
        typer.Option("--user-name", help="Gateway user name for source context."),
    ] = None,
    thread_id: Annotated[
        str | None,
        typer.Option("--thread-id", help="Gateway thread id for source context."),
    ] = None,
    message_id: Annotated[
        str | None,
        typer.Option("--message-id", help="Gateway message id for source context."),
    ] = None,
) -> None:
    """Print the prompt that would be sent to the LLM without calling the LLM."""

    config = load_config()
    store = _store(config)
    retriever = MemoryRetriever(store)
    source = _merge_source_context(
        _source_from_gateway_session(store, session),
        session_id=session,
        platform=platform,
        chat_id=chat_id,
        chat_type=chat_type,
        user_id=user_id,
        user_name=user_name,
        thread_id=thread_id,
        message_id=message_id,
    )
    context = retriever.retrieve_context(
        message,
        session_id=session,
        limit=config.retrieval_limit,
        record_access=False,
    )
    retrieved_ids = {
        "episodic": [memory.id for memory in context.episodic_memories],
        "semantic": [memory.id for memory in context.semantic_memories],
        "procedural": [memory.id for memory in context.procedural_memories],
    }
    access_rows = _memory_access_rows(store, query=message, retrieved_ids=retrieved_ids)
    latest_ordinal = store.latest_conversation_ordinal(session)
    session_context = SessionContextManager(store).load(
        session,
        before_ordinal=latest_ordinal + 1 if latest_ordinal else None,
    )
    messages = PromptBuilder().build(
        message,
        context,
        session_context=session_context,
    )
    console.print(f"Session: {session}")
    _render_source_context(source)
    context_table = Table(title="Session Context")
    context_table.add_column("Field")
    context_table.add_column("Value")
    context_table.add_row(
        "compressed_until_ordinal",
        str(session_context.compressed_until_ordinal),
    )
    context_table.add_row(
        "summary",
        "present" if session_context.summary else "",
    )
    context_table.add_row(
        "uncompressed_messages",
        str(len(session_context.uncompressed_messages)),
    )
    console.print(context_table)
    retrieved_table = Table(title="Retrieved Memory Trace")
    retrieved_table.add_column("Type")
    retrieved_table.add_column("ID")
    retrieved_table.add_column("Retrieval Score", justify="right")
    retrieved_table.add_column("Access Count", justify="right")
    retrieved_table.add_column("Accessed At")
    scored = {(row["memory_type"], row["memory_id"]): row for row in access_rows}
    for memory_type, memory_ids in retrieved_ids.items():
        for memory_id in memory_ids:
            row = scored.get((memory_type, memory_id))
            retrieved_table.add_row(
                memory_type,
                memory_id,
                f"{row['retrieval_score']:.4f}" if row else "",
                str(row["access_count"]) if row else "",
                row["accessed_at"] if row else "",
            )
    console.print(retrieved_table)
    for memory_type, memory_ids in retrieved_ids.items():
        for memory_id in memory_ids:
            row = scored.get((memory_type, memory_id))
            retrieval_score = f"{row['retrieval_score']:.4f}" if row else ""
            access_count = str(row["access_count"]) if row else ""
            accessed_at = row["accessed_at"] if row else ""
            console.print(
                f"Retrieved: type={memory_type} id={memory_id} "
                f"retrieval_score={retrieval_score} access_count={access_count} "
                f"accessed_at={accessed_at}"
            )
    for chat_message in messages:
        console.rule(chat_message["role"])
        console.print(chat_message["content"])


def main() -> None:
    """Typer entrypoint."""

    app()


if __name__ == "__main__":
    main()
