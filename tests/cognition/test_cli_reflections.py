from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app
from alpha_agent.cognition.event_log.sqlite import SQLiteEventLog
from alpha_agent.cognition.models import (
    CognitiveEventKind,
    Instant,
    NLStatement,
    Reflection,
    ReflectionId,
    ReflectionKind,
    ReflectionTarget,
    RemedyHint,
    Severity,
)
from alpha_agent.cognition.projections.reflection import ReflectionProjection
from alpha_agent.state.store import StateStore
from tests.cognition.helpers import emit


def test_cli_reflections_lists_recent_rows_with_severity_filter(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "alpha.db")
    store.initialize()
    log = SQLiteEventLog(store)
    warning = Reflection(
        id=ReflectionId("reflection:1"),
        level="L1",
        kind=ReflectionKind("unsupported-tool-call"),
        severity=Severity("warning"),
        target=ReflectionTarget("decision:decision:1"),
        finding=NLStatement("Tool use lacked support."),
        suggested_remedy=RemedyHint("Ask for evidence before calling tools."),
        created_at=Instant("2026-01-01T00:00:01+00:00"),
    )
    info = Reflection(
        id=ReflectionId("reflection:2"),
        level="L1",
        kind=ReflectionKind("feedback-surprise"),
        severity=Severity("info"),
        target=ReflectionTarget("loop_run:turn_1"),
        finding=NLStatement("Feedback surprised the loop."),
        suggested_remedy=RemedyHint("Review expectation."),
        created_at=Instant("2026-01-01T00:00:02+00:00"),
    )
    projection = ReflectionProjection(store)
    projection.apply(
        emit(
            log,
            CognitiveEventKind.REFLECTED,
            payload={
                "turn_id": "turn_1",
                "session_id": "s1",
                "reflection_count": 2,
                "reflection_ids": [str(warning.id), str(info.id)],
                "targets": [
                    {"kind": "reflection", "id": str(warning.id)},
                    {"kind": "reflection", "id": str(info.id)},
                ],
                "reflections": [warning.to_record(), info.to_record()],
            },
        )
    )

    result = CliRunner().invoke(
        app,
        ["cognition", "reflections", "--severity", "warning", "--last", "10"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0
    assert "unsupported-tool-call" in result.output
    assert "Tool use lacked support." in result.output
    assert "feedback-surprise" not in result.output


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "ALPHA_CONFIG_PATH": str(tmp_path / "config.toml"),
        "ALPHA_DB_PATH": str(tmp_path / "alpha.db"),
        "ALPHA_LOG_DIR": str(tmp_path / "logs"),
        "ALPHA_DAEMON_SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "ALPHA_DAEMON_STATUS_PATH": str(tmp_path / "daemon-status.json"),
        "ALPHA_LLM_PROVIDER": "mock",
    }
