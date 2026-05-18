"""Command line interface for Alpha Agent."""

from __future__ import annotations

from typing import Annotated

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
from alpha_agent.gateway.config import (
    configured_adapter_names,
    ensure_gateway_runtime_files,
    gateway_runtime_config,
)
from alpha_agent.gateway.logging import append_gateway_log
from alpha_agent.gateway.status import (
    GatewayStatus,
    gateway_tables_available,
    idle_status,
    is_pid_running,
    read_gateway_status,
    running_status,
    write_gateway_status,
)
from alpha_agent.llm.base import LLMProvider
from alpha_agent.llm.codex import CODEX_DEFAULT_MODEL, CodexResponsesProvider
from alpha_agent.llm.deepseek import DEEPSEEK_DEFAULT_MODEL, DeepSeekProvider
from alpha_agent.llm.mock import MockLLMProvider
from alpha_agent.llm.openai_compatible import (
    OPENAI_COMPATIBLE_DEFAULT_MODEL,
    OpenAICompatibleProvider,
)
from alpha_agent.memory.consolidation import ConsolidationService
from alpha_agent.memory.procedural import ProceduralMemoryManager
from alpha_agent.memory.retrieval import MemoryRetriever
from alpha_agent.memory.store import MemoryStore
from alpha_agent.memory.working import WorkingMemoryManager
from alpha_agent.runtime.agent import AlphaAgent
from alpha_agent.runtime.prompt_builder import PromptBuilder
from alpha_agent.runtime.session import new_session_id

console = Console()
app = typer.Typer(help="Alpha Agent personal memory runtime.")
memory_app = typer.Typer(help="Memory inspection and consolidation commands.")
skills_app = typer.Typer(help="Procedural skill commands.")
debug_app = typer.Typer(help="Debug commands.")
gateway_app = typer.Typer(help="Gateway operational commands.")
config_app = typer.Typer(help="Configuration commands.")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(debug_app, name="debug")
app.add_typer(gateway_app, name="gateway")
app.add_typer(config_app, name="config")


def _provider(config: AlphaConfig) -> LLMProvider:
    if config.llm_provider in {"mock", ""}:
        return MockLLMProvider()
    if config.llm_provider in {"openai", "openai-compatible", "compatible"}:
        return OpenAICompatibleProvider(config)
    if config.llm_provider in {"deepseek"}:
        return DeepSeekProvider(config)
    if config.llm_provider in {"codex", "openai-codex", "openai_codex"}:
        return CodexResponsesProvider(config)
    raise typer.BadParameter(f"Unknown ALPHA_LLM_PROVIDER: {config.llm_provider}")


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
    store = MemoryStore(config.db_path)
    store.initialize()
    return store


def _build_agent(config: AlphaConfig) -> AlphaAgent:
    store = _store(config)
    working = WorkingMemoryManager(store, limit=config.working_memory_limit)
    procedural = ProceduralMemoryManager(store)
    procedural.load_builtin_skills()
    retriever = MemoryRetriever(store, working)
    return AlphaAgent(
        store=store,
        llm_provider=_provider(config),
        working_memory=working,
        retriever=retriever,
        retrieval_limit=config.retrieval_limit,
    )


def _initialize_gateway(config: AlphaConfig) -> int:
    store = _store(config)
    return len(ProceduralMemoryManager(store).load_builtin_skills())


def _render_gateway_status(status: GatewayStatus, *, status_path: str) -> None:
    process = "running" if status.running else "not running"
    table = Table(title="Gateway Status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("State", status.state)
    table.add_row("Process", process)
    table.add_row("PID", str(status.pid) if status.pid is not None else "-")
    table.add_row("DB path", status.db_path)
    table.add_row("Log dir", status.log_dir)
    table.add_row("Status path", status_path)
    table.add_row("Adapters", ", ".join(status.adapters) if status.adapters else "none")
    table.add_row("Message", status.message)
    console.print(table)
    typer.echo(f"DB path: {status.db_path}")
    typer.echo(f"Log dir: {status.log_dir}")
    typer.echo(f"Status path: {status_path}")


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
    table.add_row("llm_provider", config.llm_provider)
    if config.llm_provider in {"openai-compatible", "openai", "compatible"}:
        table.add_row("compatible_base_url", config.compatible_base_url or "")
    table.add_row("llm_model", _display_model(config))
    table.add_row("working_memory_limit", str(config.working_memory_limit))
    table.add_row("retrieval_limit", str(config.retrieval_limit))
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


@gateway_app.command("status")
def gateway_status() -> None:
    """Show gateway runtime status."""

    config = load_config()
    runtime = gateway_runtime_config(config)
    status = read_gateway_status(runtime.status_path)
    if status is None:
        status = idle_status(db_path=config.db_path, log_dir=runtime.log_dir)
    elif status.running and not is_pid_running(status.pid):
        status = idle_status(
            db_path=config.db_path,
            log_dir=runtime.log_dir,
            message="Gateway status file exists, but the recorded process is not running.",
        )
    _render_gateway_status(status, status_path=str(runtime.status_path))


@gateway_app.command("doctor")
def gateway_doctor() -> None:
    """Initialize gateway prerequisites and report local operational readiness."""

    config = load_config()
    runtime = gateway_runtime_config(config)
    ensure_gateway_runtime_files(runtime)
    loaded_skills = _initialize_gateway(config)
    tables = gateway_tables_available(config.db_path)
    adapters = configured_adapter_names()
    append_gateway_log(
        runtime.log_paths["gateway.log"],
        event="gateway.doctor",
        message="Gateway doctor completed.",
        metadata={
            "adapter_count": len(adapters),
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
        ", ".join(adapters) if adapters else "No real platform adapters configured",
    )
    console.print(table)
    typer.echo(f"DB path: {config.db_path}")
    typer.echo(f"Log dir: {runtime.log_dir}")
    typer.echo(f"Status path: {runtime.status_path}")
    for name, path in runtime.log_paths.items():
        typer.echo(f"{name}: {path}")


@gateway_app.command("run")
def gateway_run(
    once: Annotated[
        bool,
        typer.Option("--once", help="Run one startup smoke cycle and exit."),
    ] = False,
) -> None:
    """Run the gateway development stub."""

    config = load_config()
    runtime = gateway_runtime_config(config)
    ensure_gateway_runtime_files(runtime)
    _initialize_gateway(config)

    adapters = configured_adapter_names()
    append_gateway_log(
        runtime.log_paths["gateway.log"],
        event="gateway.run.start",
        message="Gateway run invoked.",
        metadata={"adapter_count": len(adapters), "once": once},
    )
    write_gateway_status(
        runtime.status_path,
        running_status(
            db_path=config.db_path,
            log_dir=runtime.log_dir,
            adapter_names=adapters,
            message="Gateway process starting.",
        ),
    )

    if not adapters:
        message = "No platform adapters configured; gateway run exited cleanly."
        append_gateway_log(
            runtime.log_paths["gateway.log"],
            event="gateway.run.no_adapters",
            message=message,
        )
        write_gateway_status(
            runtime.status_path,
            idle_status(db_path=config.db_path, log_dir=runtime.log_dir, message=message),
        )
        console.print(
            "No platform adapters configured; initialized DB and runtime files, "
            "then exited cleanly."
        )
        return

    if once:
        message = "Gateway smoke cycle completed."
        write_gateway_status(
            runtime.status_path,
            idle_status(db_path=config.db_path, log_dir=runtime.log_dir, message=message),
        )
        console.print(message)


@app.command()
def ask(message: Annotated[str, typer.Argument(help="Message to send to the agent.")]) -> None:
    """Run a single-turn ask with the configured provider."""

    config = load_config()
    agent = _build_agent(config)
    result = agent.respond(message, session_id=new_session_id())
    console.print(result.response)


@app.command()
def chat(
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Existing session id to continue."),
    ] = None,
) -> None:
    """Start an interactive chat session."""

    config = load_config()
    agent = _build_agent(config)
    session_id = session or new_session_id()
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
            report = ConsolidationService(agent.store).consolidate()
            console.print(report.render())
            continue
        result = agent.respond(user_message, session_id=session_id)
        console.print(f"[bold green]alpha>[/bold green] {result.response}")


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
    working = WorkingMemoryManager(store, limit=config.working_memory_limit)
    context = MemoryRetriever(store, working).retrieve_context(
        query=query,
        session_id="memory-search",
        limit=config.retrieval_limit,
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
    """List builtin and stored procedural memories."""

    config = load_config()
    store = _store(config)
    ProceduralMemoryManager(store).load_builtin_skills()
    table = Table(title="Skills")
    table.add_column("Name")
    table.add_column("Trigger")
    table.add_column("Description")
    for memory in store.list_procedural_memories(limit=100):
        table.add_row(memory.name, memory.trigger, memory.description)
    console.print(table)


@debug_app.command()
def prompt(message: Annotated[str, typer.Argument(help="Message to build a prompt for.")]) -> None:
    """Print the prompt that would be sent to the LLM without calling the LLM."""

    config = load_config()
    store = _store(config)
    ProceduralMemoryManager(store).load_builtin_skills()
    working = WorkingMemoryManager(store, limit=config.working_memory_limit)
    retriever = MemoryRetriever(store, working)
    context = retriever.retrieve_context(message, session_id="debug", limit=config.retrieval_limit)
    messages = PromptBuilder().build(message, context)
    for chat_message in messages:
        console.rule(chat_message["role"])
        console.print(chat_message["content"])


def main() -> None:
    """Typer entrypoint."""

    app()


if __name__ == "__main__":
    main()
