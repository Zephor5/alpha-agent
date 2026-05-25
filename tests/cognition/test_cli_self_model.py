from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app


def test_cli_reflect_l3_self_model_show_and_history(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "alpha.db"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    runner = CliRunner()
    runner.invoke(
        app,
        [
            "cognition",
            "goals",
            "set",
            "--description",
            "seed db",
            "--target-outcome",
            "initialized",
        ],
    )

    reflect = runner.invoke(app, ["cognition", "reflect-l3", "--once"])
    show = runner.invoke(app, ["cognition", "self-model"])
    history = runner.invoke(app, ["cognition", "self-model", "history", "--last", "5"])

    assert reflect.exit_code == 0
    assert show.exit_code == 0
    assert history.exit_code == 0
    assert "reflect_l3 emitted=" in reflect.output
    assert "capabilities_self_assessed=" in show.output
    assert (
        "self_model_history=" in history.output
        or "self_model_updated event_id=" in history.output
    )
