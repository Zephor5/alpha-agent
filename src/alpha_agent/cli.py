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
from alpha_agent.memory.models import MemoryScope
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


def _render_stored_candidates(candidates: list[Any]) -> None:
    table = Table(title="Stored Memory Candidates")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Type")
    table.add_column("Scope")
    table.add_column("Content")
    table.add_column("Source Messages")
    for candidate in candidates:
        table.add_row(
            candidate.id,
            candidate.status,
            candidate.candidate_type,
            candidate.scope.scope_key,
            candidate.content,
            ", ".join(candidate.source_message_ids),
        )
    console.print(table)
    for candidate in candidates:
        console.print(
            f"Candidate: id={candidate.id} status={candidate.status} "
            f"type={candidate.candidate_type} scope={candidate.scope.scope_key} "
            f"source_message_ids={','.join(candidate.source_message_ids)} "
            f"content={candidate.content}"
        )


def _render_candidate_audit(audit: Any) -> None:
    candidate = audit.candidate
    console.print(
        f"Candidate: id={candidate.id} status={candidate.status} "
        f"type={candidate.candidate_type} scope={candidate.scope.scope_key} "
        f"source_message_ids={','.join(candidate.source_message_ids)} "
        f"content={candidate.content}"
    )
    source_table = Table(title="Source Transcript Evidence")
    source_table.add_column("ID")
    source_table.add_column("Session")
    source_table.add_column("Ordinal", justify="right")
    source_table.add_column("Role")
    source_table.add_column("Content")
    for message in audit.source_messages:
        source_table.add_row(
            message.id,
            message.session_id,
            str(message.ordinal),
            message.role,
            message.raw_content,
        )
    console.print(source_table)
    decision_table = Table(title="Decision History")
    decision_table.add_column("Action")
    decision_table.add_column("Reviewer")
    decision_table.add_column("Memory")
    decision_table.add_column("Rationale")
    decision_table.add_column("Metadata")
    for decision in audit.decisions:
        memory_ref = ""
        if decision.memory_type or decision.memory_id:
            memory_ref = f"{decision.memory_type or ''}:{decision.memory_id or ''}"
        metadata = ", ".join(
            f"{key}={value}" for key, value in decision.metadata.items()
        )
        decision_table.add_row(
            decision.action,
            decision.reviewer or "",
            memory_ref,
            decision.rationale,
            metadata,
        )
    console.print(decision_table)
    for message in audit.source_messages:
        console.print(
            f"Source: id={message.id} session={message.session_id} "
            f"ordinal={message.ordinal} role={message.role} content={message.raw_content}"
        )
    for decision in audit.decisions:
        metadata = " ".join(
            f"{key}={value}" for key, value in decision.metadata.items()
        )
        console.print(
            f"Decision: action={decision.action} reviewer={decision.reviewer or ''} "
            f"memory_type={decision.memory_type or ''} memory_id={decision.memory_id or ''} "
            f"rationale={decision.rationale} {metadata}".rstrip()
        )


def _render_memory_audit(audit: Any) -> None:
    memory = audit.memory
    console.print(
        f"Memory: id={memory.id} status={memory.status} type={memory.memory_type} "
        f"scope={memory.scope.scope_key} source_ids={','.join(audit.source_message_ids)} "
        f"content={memory.content}"
    )
    memory_table = Table(title="Semantic Memory")
    memory_table.add_column("Field")
    memory_table.add_column("Value")
    memory_table.add_row("ID", memory.id)
    memory_table.add_row("Status", memory.status)
    memory_table.add_row("Type", memory.memory_type)
    memory_table.add_row("Scope", memory.scope.scope_key)
    memory_table.add_row("Subject", memory.subject or "")
    memory_table.add_row("Predicate", memory.predicate or "")
    memory_table.add_row("Object", memory.object or "")
    memory_table.add_row("Sources", ", ".join(audit.source_message_ids))
    memory_table.add_row("Supersedes", memory.supersedes_id or "")
    memory_table.add_row("Superseded By", memory.superseded_by_id or "")
    memory_table.add_row("Content", memory.content)
    console.print(memory_table)

    chain_table = Table(title="Supersession Chain")
    chain_table.add_column("ID")
    chain_table.add_column("Status")
    chain_table.add_column("Object")
    chain_table.add_column("Sources")
    chain_table.add_column("Content")
    for item in audit.supersession_chain:
        chain_table.add_row(
            item.id,
            item.status,
            item.object or "",
            ", ".join(item.source_memory_ids),
            item.content,
        )
        console.print(
            f"Supersession: id={item.id} status={item.status} "
            f"source_ids={','.join(item.source_memory_ids)} content={item.content}"
        )
    console.print(chain_table)

    source_table = Table(title="Source Transcript Evidence")
    source_table.add_column("ID")
    source_table.add_column("Session")
    source_table.add_column("Ordinal", justify="right")
    source_table.add_column("Role")
    source_table.add_column("Content")
    for message in audit.source_messages:
        source_table.add_row(
            message.id,
            message.session_id,
            str(message.ordinal),
            message.role,
            message.raw_content,
        )
    console.print(source_table)

    if audit.projection_memories:
        projection_table = Table(title="Projection Drill-Down Memories")
        projection_table.add_column("ID")
        projection_table.add_column("Status")
        projection_table.add_column("Type")
        projection_table.add_column("Sources")
        projection_table.add_column("Content")
        for item in audit.projection_memories:
            projection_table.add_row(
                item.id,
                item.status,
                item.memory_type,
                ", ".join(item.source_memory_ids),
                item.content,
            )
            console.print(
                f"ProjectionSource: id={item.id} status={item.status} "
                f"type={item.memory_type} source_ids={','.join(item.source_memory_ids)} "
                f"content={item.content}"
            )
        console.print(projection_table)

    if audit.relation_edges:
        relation_table = Table(title="Relation Audit Edges")
        relation_table.add_column("Edge")
        relation_table.add_column("Relation")
        relation_table.add_column("Source")
        relation_table.add_column("Target")
        relation_table.add_column("Evidence")
        for item in audit.relation_edges:
            evidence_ids = [memory.id for memory in item.evidence_memories]
            relation_table.add_row(
                item.edge.id,
                item.edge.relation_type,
                item.source_node.name,
                item.target_node.name,
                ", ".join(evidence_ids),
            )
            console.print(
                f"RelationAudit: edge_id={item.edge.id} relation={item.edge.relation_type} "
                f"source={item.source_node.name} target={item.target_node.name} "
                f"evidence_memory_ids={','.join(evidence_ids)}"
            )
        console.print(relation_table)


def _render_scope_inspection(inspection: Any) -> None:
    console.print(f"MemoryInspection: query={inspection.query}")
    console.print(f"Scope: key={inspection.scope.scope_key} kind={inspection.scope.kind}")
    table = Table(title="Scoped Memory Inspection")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")
    table.add_column("Scope")
    table.add_column("Sources")
    table.add_column("Content")
    for memory in inspection.semantic_memories:
        table.add_row(
            f"semantic:{memory.memory_type}",
            memory.id,
            memory.status,
            f"{memory.confidence:.2f}",
            memory.scope.scope_key,
            ", ".join(memory.source_memory_ids),
            memory.content,
        )
        console.print(
            f"MemoryInspect: type=semantic id={memory.id} status={memory.status} "
            f"confidence={memory.confidence:.2f} scope={memory.scope.scope_key} "
            f"source_ids={','.join(memory.source_memory_ids)} content={memory.content}"
        )
    for memory in inspection.episodic_memories:
        table.add_row(
            "episodic",
            memory.id,
            "active",
            f"{memory.confidence:.2f}",
            memory.scope.scope_key,
            ", ".join(memory.source_event_ids),
            memory.summary,
        )
    for memory in inspection.procedural_memories:
        source_ids = memory.metadata.get("source_ids", [])
        if isinstance(source_ids, str):
            rendered_sources = source_ids
        elif isinstance(source_ids, list):
            rendered_sources = ", ".join(str(item) for item in source_ids)
        else:
            rendered_sources = ""
        table.add_row(
            "procedural",
            memory.id,
            str(memory.metadata.get("status") or "active"),
            f"{memory.confidence:.2f}",
            memory.scope.scope_key,
            rendered_sources,
            memory.name,
        )
    console.print(table)
    if inspection.candidates:
        _render_stored_candidates(inspection.candidates)


def _render_retrieval_diagnostics(
    diagnostics: Any,
    *,
    budgets: dict[str, int],
) -> None:
    console.print(f"RetrievalDiagnostics: query={diagnostics.query}")
    console.print(f"Scope: key={diagnostics.scope.scope_key} kind={diagnostics.scope.kind}")
    budget_table = Table(title="Prompt Budget Impact")
    budget_table.add_column("Section")
    budget_table.add_column("Budget Group")
    budget_table.add_column("Used Tokens", justify="right")
    budget_table.add_column("Budget Tokens", justify="right")
    for section, used_tokens in diagnostics.prompt_section_tokens.items():
        budget_group = diagnostics.prompt_section_budget_groups.get(section, section)
        budget_table.add_row(
            section,
            budget_group,
            str(used_tokens),
            str(budgets.get(budget_group, 0)),
        )
        console.print(
            f"PromptBudget: section={section} budget_group={budget_group} "
            f"used_tokens={used_tokens} budget_tokens={budgets.get(budget_group, 0)}"
        )
    console.print(budget_table)

    table = Table(title="Retrieval Score Breakdown")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Score", justify="right")
    table.add_column("Keyword", justify="right")
    table.add_column("FTS", justify="right")
    table.add_column("Recency", justify="right")
    table.add_column("Salience", justify="right")
    table.add_column("Access", justify="right")
    table.add_column("Scope", justify="right")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")
    table.add_column("Prompt Section")
    table.add_column("Sources")
    table.add_column("Content")
    for item in diagnostics.memories:
        explanation = item.explanation
        components = explanation.components if explanation is not None else {}
        table.add_row(
            item.memory_type,
            item.memory_id,
            f"{explanation.total:.3f}" if explanation else "",
            f"{components.get('keyword', 0):.2f}",
            f"{components.get('fts', 0):.2f}",
            f"{components.get('recency', 0):.2f}",
            f"{components.get('salience', 0):.2f}",
            f"{components.get('access', 0):.2f}",
            f"{components.get('scope_priority', 0):.2f}",
            item.status,
            f"{item.confidence:.2f}" if item.confidence is not None else "",
            item.prompt_section,
            ", ".join(item.source_ids),
            item.content,
        )
        console.print(
            f"RetrievalDiagnostic: type={item.memory_type} id={item.memory_id} "
            f"score={explanation.total if explanation else 0.0:.4f} "
            f"keyword={components.get('keyword', 0):.2f} "
            f"fts={components.get('fts', 0):.2f} "
            f"recency={components.get('recency', 0):.2f} "
            f"salience={components.get('salience', 0):.2f} "
            f"access={components.get('access', 0):.2f} "
            f"scope_priority={components.get('scope_priority', 0):.2f} "
            f"prompt_section={item.prompt_section} prompt_tokens={item.prompt_tokens} "
            f"status={item.status} "
            f"confidence={item.confidence if item.confidence is not None else ''} "
            f"source_ids={','.join(item.source_ids)}"
        )
    console.print(table)


def _render_maintenance_report(report: Any) -> None:
    table = Table(title="Memory Maintenance")
    table.add_column("Area")
    table.add_column("Count", justify="right")
    table.add_row("stale_candidates", str(len(report.stale_candidates)))
    table.add_row("inactive_memories", str(len(report.inactive_memories)))
    table.add_row("rejected_stale", str(report.rejected_stale_count))
    table.add_row("cleaned_search_index", str(report.cleaned_search_index_count))
    console.print(table)
    console.print(
        f"Maintenance: stale_candidates={len(report.stale_candidates)} "
        f"inactive_memories={len(report.inactive_memories)} "
        f"rejected_stale={report.rejected_stale_count} "
        f"cleaned_search_index={report.cleaned_search_index_count}"
    )
    if report.stale_candidates:
        _render_stored_candidates(report.stale_candidates)
    if report.cleaned_search_index_memories:
        cleaned_table = Table(title="Cleaned Inactive Search Index Rows")
        cleaned_table.add_column("ID")
        cleaned_table.add_column("Status")
        cleaned_table.add_column("Scope")
        cleaned_table.add_column("Content")
        for memory in report.cleaned_search_index_memories:
            cleaned_table.add_row(
                memory.id,
                memory.status,
                memory.scope.scope_key,
                memory.content,
            )
            console.print(
                f"CleanedInactiveIndex: id={memory.id} status={memory.status} "
                f"scope={memory.scope.scope_key} content={memory.content}"
            )
        console.print(cleaned_table)


def _render_operational_metrics(metrics: dict[str, int | float]) -> None:
    table = Table(title="Memory Operational Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key, value in metrics.items():
        rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
        table.add_row(key, rendered)
        console.print(f"MemoryMetric: {key}={rendered}")
    console.print(table)


def _retrieved_memory_metadata(memory_type: str, memory: Any) -> dict[str, str]:
    if memory_type == "semantic":
        source_ids = getattr(memory, "source_memory_ids", [])
        status = str(getattr(memory, "status", "active"))
    elif memory_type == "episodic":
        source_ids = getattr(memory, "source_event_ids", [])
        status = "active"
    else:
        metadata = getattr(memory, "metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_ids = (
            metadata.get("source_ids")
            or metadata.get("source_memory_ids")
            or metadata.get("source_event_ids")
            or []
        )
        status = str(metadata.get("status") or "active")
    if isinstance(source_ids, str):
        rendered_sources = source_ids
    else:
        rendered_sources = ",".join(str(item) for item in source_ids)
    confidence = getattr(memory, "confidence", None)
    if isinstance(confidence, int | float):
        rendered_confidence = f"{confidence:.2f}"
    else:
        rendered_confidence = ""
    return {
        "source_ids": rendered_sources,
        "status": status,
        "confidence": rendered_confidence,
    }


def _print_retrieved_memory_line(
    prefix: str,
    memory_type: str,
    memory_id: str,
    metadata: dict[str, str],
    explanation: Any,
    *,
    access_count: str = "",
) -> None:
    components = explanation.components if explanation is not None else {}
    retrieval_score = f"{explanation.total:.4f}" if explanation else ""
    reason = ",".join(explanation.reasons) if explanation else ""
    access_part = f" access_count={access_count}" if access_count else ""
    console.print(
        f"{prefix}: type={memory_type} id={memory_id} "
        f"source_ids={metadata['source_ids']} "
        f"memory_status={metadata['status']} "
        f"memory_confidence={metadata['confidence']} "
        f"retrieval_score={retrieval_score} "
        f"keyword={components.get('keyword', 0):.2f} "
        f"fts={components.get('fts', 0):.2f} "
        f"recency={components.get('recency', 0):.2f} "
        f"salience={components.get('salience', 0):.2f} "
        f"stability={components.get('stability', 0):.2f} "
        f"access={components.get('access', 0):.2f} "
        f"scope_priority={components.get('scope_priority', 0):.2f} "
        f"status={components.get('status', 0):.2f} "
        f"source_confidence={components.get('source_confidence', 0):.2f}"
        f"{access_part} accessed_at= reason={reason}"
    )


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


def _memory_scope_from_gateway_session(
    store: MemoryStore,
    session_id: str,
) -> MemoryScope | None:
    with store.connect() as conn:
        row = conn.execute(
            """
            SELECT memory_scope FROM gateway_session_mappings
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    try:
        record = json.loads(row["memory_scope"] or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(record, dict) or not record.get("scope_key"):
        return None
    return MemoryScope.from_record(record)


def _memory_scope_from_option(scope_key: str | None) -> MemoryScope:
    if not scope_key:
        return MemoryScope.default()
    normalized = scope_key.strip()
    if normalized.startswith("user:"):
        return MemoryScope(
            kind="global_user",
            scope_key=normalized,
            user_id=normalized.split(":", 1)[1] or None,
        )
    if normalized.startswith("project:"):
        parts = normalized.split(":")
        project_id = parts[1] if len(parts) > 1 else None
        user_id = parts[3] if len(parts) > 3 and parts[2] == "user" else None
        return MemoryScope(
            kind="project",
            scope_key=normalized,
            project_id=project_id,
            user_id=user_id,
        )
    if normalized.startswith("platform:") and ":chat:" in normalized:
        parts = normalized.split(":")
        return MemoryScope(
            kind="chat_thread",
            scope_key=normalized,
            platform=parts[1] if len(parts) > 1 else None,
            chat_id=parts[3] if len(parts) > 3 else None,
            thread_id=parts[5] if len(parts) > 5 else None,
            user_id=parts[7] if len(parts) > 7 else None,
        )
    if normalized.startswith("platform:"):
        parts = normalized.split(":")
        return MemoryScope(
            kind="platform_user",
            scope_key=normalized,
            platform=parts[1] if len(parts) > 1 else None,
            user_id=parts[3] if len(parts) > 3 else None,
        )
    return MemoryScope(kind="global_user", scope_key=normalized)


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


def _source_metadata(source: ConversationSource | None) -> dict[str, Any] | None:
    if source is None:
        return None
    metadata = dict(source.metadata)
    metadata.update(
        {
            "platform": source.platform,
            "chat_id": source.chat_id,
            "chat_type": source.chat_type,
            "user_id": source.user_id,
            "user_name": source.user_name,
            "thread_id": source.thread_id,
            "message_id": source.message_id,
        }
    )
    return metadata


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
    table.add_row("memory_capture_mode", config.memory_capture_mode)
    table.add_row(
        "memory_cli_capture_mode",
        (config.memory_channel_capture_modes or {}).get("cli", ""),
    )
    table.add_row(
        "memory_gateway_capture_mode",
        (config.memory_channel_capture_modes or {}).get("gateway", ""),
    )
    table.add_row("memory_consolidation_mode", config.memory_consolidation_mode)
    table.add_row(
        "memory_consolidation_after_turns",
        str(config.memory_consolidation_after_turns),
    )
    table.add_row("context_max_prompt_tokens", str(config.context_max_prompt_tokens))
    table.add_row(
        "context_compression_threshold_ratio",
        str(config.context_compression_threshold_ratio),
    )
    table.add_row("context_recent_tail_messages", str(config.context_recent_tail_messages))
    table.add_row("context_min_summary_tokens", str(config.context_min_summary_tokens))
    table.add_row("context_max_summary_tokens", str(config.context_max_summary_tokens))
    table.add_row(
        "context_semantic_memory_tokens",
        str(config.context_semantic_memory_tokens),
    )
    table.add_row(
        "context_episodic_memory_tokens",
        str(config.context_episodic_memory_tokens),
    )
    table.add_row(
        "context_procedural_memory_tokens",
        str(config.context_procedural_memory_tokens),
    )
    table.add_row(
        "context_session_context_tokens",
        str(config.context_session_context_tokens),
    )
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
    scopes = MemoryScope.default().allowed_read_scopes()
    table = Table(title="Recent Memories")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Content")
    for semantic_memory in store.list_semantic_memories(
        limit,
        scopes=scopes,
        statuses=["active"],
    ):
        table.add_row("semantic", semantic_memory.id, semantic_memory.content)
    for episodic_memory in store.list_episodic_memories(limit, scopes=scopes):
        table.add_row("episodic", episodic_memory.id, episodic_memory.summary)
    for procedural_memory in store.list_procedural_memories(limit, scopes=scopes):
        table.add_row("procedural", procedural_memory.id, procedural_memory.name)
    console.print(table)


@memory_app.command("inspect")
def memory_inspect(
    query: Annotated[
        str | None,
        typer.Argument(
            help="Inspection prompt. Defaults to 'what do you remember about me?'."
        ),
    ] = None,
    scope_key: Annotated[
        str | None,
        typer.Option("--scope-key", help="Inspect a specific memory scope key."),
    ] = None,
    include_inactive: Annotated[
        bool,
        typer.Option("--include-inactive", help="Include superseded, deleted, and conflict rows."),
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
) -> None:
    """Answer what Alpha remembers in the selected scope."""

    config = load_config()
    service = MemoryReviewService(_store(config))
    inspection = service.inspect_scope(
        query=query or "what do you remember about me?",
        scope=_memory_scope_from_option(scope_key),
        include_inactive=include_inactive,
        limit=limit,
    )
    _render_scope_inspection(inspection)


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
        scopes=MemoryScope.default().allowed_read_scopes(),
        record_access=False,
    )
    table = Table(title=f"Memory search: {query}")
    table.add_column("Type")
    table.add_column("ID")
    table.add_column("Score", justify="right")
    table.add_column("Why")
    table.add_column("Status")
    table.add_column("Confidence", justify="right")
    table.add_column("Source IDs")
    table.add_column("Content")
    for semantic_memory in context.semantic_memories:
        explanation = context.retrieval_explanations.get(f"semantic:{semantic_memory.id}")
        metadata = _retrieved_memory_metadata("semantic", semantic_memory)
        table.add_row(
            "semantic",
            semantic_memory.id,
            f"{explanation.total:.3f}" if explanation else "",
            ", ".join(explanation.reasons) if explanation else "",
            metadata["status"],
            metadata["confidence"],
            metadata["source_ids"],
            semantic_memory.content,
        )
        _print_retrieved_memory_line(
            "MemorySearch",
            "semantic",
            semantic_memory.id,
            metadata,
            explanation,
        )
    for episodic_memory in context.episodic_memories:
        explanation = context.retrieval_explanations.get(f"episodic:{episodic_memory.id}")
        metadata = _retrieved_memory_metadata("episodic", episodic_memory)
        table.add_row(
            "episodic",
            episodic_memory.id,
            f"{explanation.total:.3f}" if explanation else "",
            ", ".join(explanation.reasons) if explanation else "",
            metadata["status"],
            metadata["confidence"],
            metadata["source_ids"],
            episodic_memory.summary,
        )
        _print_retrieved_memory_line(
            "MemorySearch",
            "episodic",
            episodic_memory.id,
            metadata,
            explanation,
        )
    for procedural_memory in context.procedural_memories:
        explanation = context.retrieval_explanations.get(
            f"procedural:{procedural_memory.id}"
        )
        metadata = _retrieved_memory_metadata("procedural", procedural_memory)
        table.add_row(
            "procedural",
            procedural_memory.id,
            f"{explanation.total:.3f}" if explanation else "",
            ", ".join(explanation.reasons) if explanation else "",
            metadata["status"],
            metadata["confidence"],
            metadata["source_ids"],
            procedural_memory.name,
        )
        _print_retrieved_memory_line(
            "MemorySearch",
            "procedural",
            procedural_memory.id,
            metadata,
            explanation,
        )
    console.print(table)


@memory_app.command("diagnostics")
def memory_diagnostics(
    query: Annotated[str, typer.Argument(help="Query to explain retrieval for.")],
    session: Annotated[
        str,
        typer.Option("--session", "-s", help="Session id used for query expansion."),
    ] = "memory-diagnostics",
    scope_key: Annotated[
        str | None,
        typer.Option("--scope-key", help="Diagnose a specific memory scope key."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 8,
) -> None:
    """Explain why retrieval selected memories and their prompt budget impact."""

    config = load_config()
    service = MemoryReviewService(_store(config))
    diagnostics = service.retrieval_diagnostics(
        query=query,
        session_id=session,
        scope=_memory_scope_from_option(scope_key),
        limit=limit,
        prompt_builder=PromptBuilder(
            semantic_memory_tokens=config.context_semantic_memory_tokens,
            episodic_memory_tokens=config.context_episodic_memory_tokens,
            procedural_memory_tokens=config.context_procedural_memory_tokens,
            session_context_tokens=config.context_session_context_tokens,
        ),
    )
    _render_retrieval_diagnostics(
        diagnostics,
        budgets={
            "semantic": config.context_semantic_memory_tokens,
            "episodic": config.context_episodic_memory_tokens,
            "procedural": config.context_procedural_memory_tokens,
        },
    )


@memory_app.command()
def forget(
    memory_id: Annotated[str, typer.Argument(help="Semantic memory id to forget.")],
    reason: Annotated[
        str,
        typer.Option("--reason", help="Audit reason recorded on the deleted memory."),
    ] = "user requested forget",
) -> None:
    """Mark a semantic memory deleted without physically removing evidence."""

    config = load_config()
    store = _store(config)
    service = MemoryReviewService(store)
    try:
        service.inspect_memory(memory_id)
    except (KeyError, PermissionError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    forgotten = store.forget_semantic_memory(memory_id, reason=reason)
    console.print(
        f"Memory {forgotten.id} marked {forgotten.status}; "
        "it is excluded from retrieval and prompts."
    )


@memory_app.command()
def audit(
    memory_id: Annotated[str, typer.Argument(help="Semantic memory id to inspect.")],
) -> None:
    """Inspect semantic memory source evidence and supersession lineage."""

    config = load_config()
    service = MemoryReviewService(_store(config))
    _render_memory_audit(service.inspect_memory(memory_id))


@memory_app.command()
def consolidate() -> None:
    """Run manual memory consolidation."""

    config = load_config()
    store = _store(config)
    report = ConsolidationService(store).consolidate()
    console.print(report.render())


@memory_app.command("maintenance")
def memory_maintenance(
    stale_days: Annotated[
        int,
        typer.Option("--stale-days", help="Candidate age threshold for stale review items."),
    ] = 14,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    reject_stale: Annotated[
        bool,
        typer.Option("--reject-stale", help="Reject stale pending/edited candidates."),
    ] = False,
    cleanup_inactive_index: Annotated[
        bool,
        typer.Option(
            "--cleanup-inactive-index",
            help="Remove inactive semantic memories from the optional FTS index.",
        ),
    ] = False,
    run_consolidation: Annotated[
        bool,
        typer.Option("--consolidate", help="Run manual consolidation as part of maintenance."),
    ] = False,
    diagnostics_query: Annotated[
        str | None,
        typer.Option("--diagnostics", help="Also run retrieval diagnostics for this query."),
    ] = None,
) -> None:
    """Run memory maintenance checks without changing transcript history."""

    config = load_config()
    store = _store(config)
    service = MemoryReviewService(store)
    report = service.maintenance_report(
        stale_days=stale_days,
        limit=limit,
        reject_stale=reject_stale,
        cleanup_inactive_index=cleanup_inactive_index,
    )
    _render_maintenance_report(report)
    if run_consolidation:
        consolidation = ConsolidationService(store).consolidate()
        console.print(consolidation.render())
    if diagnostics_query:
        diagnostics = service.retrieval_diagnostics(
            query=diagnostics_query,
            limit=config.retrieval_limit,
            prompt_builder=PromptBuilder(
                semantic_memory_tokens=config.context_semantic_memory_tokens,
                episodic_memory_tokens=config.context_episodic_memory_tokens,
                procedural_memory_tokens=config.context_procedural_memory_tokens,
                session_context_tokens=config.context_session_context_tokens,
            ),
        )
        _render_retrieval_diagnostics(
            diagnostics,
            budgets={
                "semantic": config.context_semantic_memory_tokens,
                "episodic": config.context_episodic_memory_tokens,
                "procedural": config.context_procedural_memory_tokens,
            },
        )


@memory_app.command("review")
def memory_review(
    message: Annotated[
        str | None,
        typer.Argument(help="Message to extract memory candidates from."),
    ] = None,
    session: Annotated[
        str | None,
        typer.Option("--session", "-s", help="Session id for approved review source events."),
    ] = None,
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
        typer.Option("--reject-all", help="Reject and audit all extracted candidates."),
    ] = False,
    reject: Annotated[
        list[int] | None,
        typer.Option(
            "--reject",
            help="1-based candidate index to reject and audit; can be repeated.",
        ),
    ] = None,
    candidate_index: Annotated[
        int | None,
        typer.Option("--candidate", help="1-based candidate index to edit and approve."),
    ] = None,
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
    list_pending: Annotated[
        bool,
        typer.Option("--list-pending", help="List stored pending candidates."),
    ] = False,
    list_stored: Annotated[
        bool,
        typer.Option("--list-stored", help="List stored pending and edited candidates."),
    ] = False,
    candidate_id: Annotated[
        str | None,
        typer.Option(
            "--candidate-id",
            help="Stored candidate id to approve, reject, edit, or inspect.",
        ),
    ] = None,
    approve_stored: Annotated[
        bool,
        typer.Option("--approve-stored", help="Approve a stored candidate by id."),
    ] = False,
    reject_stored: Annotated[
        bool,
        typer.Option("--reject-stored", help="Reject a stored candidate by id."),
    ] = False,
    edit_stored: Annotated[
        bool,
        typer.Option("--edit-stored", help="Edit a stored candidate by id."),
    ] = False,
    inspect_stored: Annotated[
        bool,
        typer.Option(
            "--inspect-stored",
            "--inspect-source",
            "--decision-history",
            help="Show stored candidate source transcript evidence and decision history.",
        ),
    ] = False,
) -> None:
    """Preview, list, approve, reject, edit, or inspect memory candidates."""

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
    stored_actions = [
        list_pending,
        list_stored,
        approve_stored,
        reject_stored,
        edit_stored,
        inspect_stored,
    ]
    if sum(1 for action in stored_actions if action) > 1:
        raise typer.BadParameter(
            "Use only one stored review action: --approve-stored, --reject-stored, "
            "--edit-stored, or --inspect-stored."
        )
    stored_or_list_requested = any(stored_actions)
    one_shot_options_requested = (
        message is not None
        or approve_all
        or bool(approve_indices)
        or reject_all
        or bool(reject_indices)
        or candidate_index is not None
        or session is not None
    )
    invalid_edit_fields = edit_requested and not edit_stored
    invalid_list_candidate_id = candidate_id is not None and (list_pending or list_stored)
    if stored_or_list_requested and (
        one_shot_options_requested or invalid_edit_fields or invalid_list_candidate_id
    ):
        raise typer.BadParameter(
            "Stored review actions cannot be combined with MESSAGE, one-shot review "
            "flags, --candidate, unrelated edit fields, or list filters."
        )

    config = load_config()
    service = MemoryReviewService(_store(config))
    if list_pending:
        _render_stored_candidates(service.list_candidates(status="pending"))
        return
    if list_stored:
        _render_stored_candidates(service.list_reviewable_candidates())
        return
    if approve_stored or reject_stored or edit_stored or inspect_stored:
        if not candidate_id:
            raise typer.BadParameter("--candidate-id is required for stored review actions.")
        if approve_stored:
            stored = service.approve_stored(candidate_id)
            console.print(f"Approved {len(stored)} candidate(s).")
            return
        if edit_stored:
            if not edit_requested:
                raise typer.BadParameter("--edit-stored requires at least one edit field.")
            edited = service.edit_stored(
                candidate_id,
                content=edit_content,
                subject=edit_subject,
                predicate=edit_predicate,
                object_value=edit_object,
            )
            console.print(f"Edited stored candidate {edited.id}.")
            return
        if inspect_stored:
            _render_candidate_audit(service.inspect_stored(candidate_id))
            return
        rejected = service.reject_stored(candidate_id)
        console.print(f"Rejected candidate {rejected.id}.")
        return
    if message is None:
        raise typer.BadParameter("MESSAGE is required unless using stored review actions.")

    candidates = service.preview(message)
    _render_candidates(candidates)
    if not candidates:
        console.print("No memory candidates extracted.")
        return
    if reject_all:
        service.reject(
            message=message,
            session_id=session or "memory-review",
            candidates=candidates,
            reviewer="cli",
        )
        console.print(f"Rejected {len(candidates)} candidate(s).")
        return

    valid_indices = set(range(1, len(candidates) + 1))
    requested_indices = approve_indices | reject_indices
    if requested_indices - valid_indices:
        raise typer.BadParameter("Candidate indexes must refer to extracted candidates.")

    reviewed_candidates = list(candidates)
    if edit_requested:
        edited_index = candidate_index or 1
        if edited_index < 1 or edited_index > len(candidates):
            raise typer.BadParameter("--candidate must refer to an extracted candidate.")
        reviewed_candidates[edited_index - 1] = edit_candidate(
            candidates[edited_index - 1],
            content=edit_content,
            subject=edit_subject,
            predicate=edit_predicate,
            object_value=edit_object,
        )
        console.print(f"Edited candidate {edited_index}.")
        if not approve_all and not approve_indices:
            approve_indices.add(edited_index)

    if approve_all:
        selected_indices = valid_indices - reject_indices
    else:
        selected_indices = approve_indices - reject_indices
    if reject_indices:
        rejected = [
            candidate
            for index, candidate in enumerate(reviewed_candidates, start=1)
            if index in reject_indices
        ]
        service.reject(
            message=message,
            session_id=session or "memory-review",
            candidates=rejected,
            reviewer="cli",
        )
        console.print(f"Rejected {len(reject_indices)} candidate(s).")
    if selected_indices:
        selected = [
            candidate
            for index, candidate in enumerate(reviewed_candidates, start=1)
            if index in selected_indices
        ]
        stored = service.approve(
            message=message,
            session_id=session or "memory-review",
            candidates=selected,
        )
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


@memory_app.command("metrics")
def memory_metrics() -> None:
    """Show operational memory lifecycle and retrieval metrics."""

    config = load_config()
    service = MemoryReviewService(_store(config))
    _render_operational_metrics(service.operational_metrics())


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
    memory_scope = _memory_scope_from_gateway_session(store, session)
    if memory_scope is None:
        memory_scope = MemoryScope.from_source_metadata(
            session_id=session,
            source_metadata=_source_metadata(source),
        )
    context = retriever.retrieve_context(
        message,
        session_id=session,
        limit=config.retrieval_limit,
        scopes=memory_scope.allowed_read_scopes(),
        record_access=False,
    )
    retrieved_ids = {
        "episodic": [memory.id for memory in context.episodic_memories],
        "semantic": [memory.id for memory in context.semantic_memories],
        "procedural": [memory.id for memory in context.procedural_memories],
    }
    latest_ordinal = store.latest_conversation_ordinal(session)
    session_context = SessionContextManager(store).load(
        session,
        before_ordinal=latest_ordinal + 1 if latest_ordinal else None,
    )
    messages = PromptBuilder(
        semantic_memory_tokens=config.context_semantic_memory_tokens,
        episodic_memory_tokens=config.context_episodic_memory_tokens,
        procedural_memory_tokens=config.context_procedural_memory_tokens,
        session_context_tokens=config.context_session_context_tokens,
    ).build(
        message,
        context,
        session_context=session_context,
    )
    console.print(f"Session: {session}")
    console.print(f"Memory scope: {memory_scope.scope_key}")
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
    retrieved_table.add_column("Keyword", justify="right")
    retrieved_table.add_column("Scope", justify="right")
    retrieved_table.add_column("Status")
    retrieved_table.add_column("Confidence", justify="right")
    retrieved_table.add_column("Source IDs")
    retrieved_table.add_column("Reasons")
    retrieved_table.add_column("Access Count", justify="right")
    retrieved_table.add_column("Accessed At")
    retrieved_memory_map = {
        **{("semantic", memory.id): memory for memory in context.semantic_memories},
        **{("episodic", memory.id): memory for memory in context.episodic_memories},
        **{("procedural", memory.id): memory for memory in context.procedural_memories},
    }
    for memory_type, memory_ids in retrieved_ids.items():
        for memory_id in memory_ids:
            explanation = context.retrieval_explanations.get(f"{memory_type}:{memory_id}")
            components = explanation.components if explanation is not None else {}
            metadata = _retrieved_memory_metadata(
                memory_type,
                retrieved_memory_map[(memory_type, memory_id)],
            )
            retrieved_table.add_row(
                memory_type,
                memory_id,
                f"{explanation.total:.4f}" if explanation else "",
                f"{components.get('keyword', 0):.2f}" if explanation else "",
                f"{components.get('scope_priority', 0):.2f}" if explanation else "",
                metadata["status"],
                metadata["confidence"],
                metadata["source_ids"],
                ", ".join(explanation.reasons) if explanation else "",
                str(store.count_memory_access(memory_id, memory_type)),
                "",
            )
    console.print(retrieved_table)
    for memory_type, memory_ids in retrieved_ids.items():
        for memory_id in memory_ids:
            explanation = context.retrieval_explanations.get(f"{memory_type}:{memory_id}")
            access_count = str(store.count_memory_access(memory_id, memory_type))
            _print_retrieved_memory_line(
                "Retrieved",
                memory_type,
                memory_id,
                _retrieved_memory_metadata(
                    memory_type,
                    retrieved_memory_map[(memory_type, memory_id)],
                ),
                explanation,
                access_count=access_count,
            )
    for chat_message in messages:
        console.rule(chat_message["role"])
        console.print(chat_message["content"])


def main() -> None:
    """Typer entrypoint."""

    app()


if __name__ == "__main__":
    main()
