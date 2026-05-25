from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alpha_agent.cli import app


def test_cli_goals_set_list_satisfy_abandon_and_drive_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    db_path = tmp_path / "alpha.db"
    monkeypatch.setenv("ALPHA_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("ALPHA_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHA_LLM_PROVIDER", "mock")
    runner = CliRunner()

    set_result = runner.invoke(
        app,
        [
            "cognition",
            "goals",
            "set",
            "--description",
            "answer pending user question",
            "--priority",
            "6",
            "--target-outcome",
            "clear answer sent",
        ],
    )
    list_result = runner.invoke(app, ["cognition", "goals", "list", "--active"])
    goal_id = _goal_id_from_output(set_result.output)
    drive_result = runner.invoke(app, ["cognition", "drive", "--once"])
    satisfy_result = runner.invoke(
        app,
        ["cognition", "goals", "satisfy", goal_id, "--evidence", "accepted"],
    )
    abandon_result = runner.invoke(
        app,
        ["cognition", "goals", "abandon", goal_id, "--reason", "obsolete"],
    )

    assert set_result.exit_code == 0
    assert list_result.exit_code == 0
    assert drive_result.exit_code == 0
    assert satisfy_result.exit_code == 0
    assert abandon_result.exit_code == 0
    assert "goal_set event_id=" in set_result.output
    assert f"goal={goal_id}" in list_result.output
    assert "drive triggered=true dropped=false" in drive_result.output
    assert "goal_satisfied event_id=" in satisfy_result.output
    assert "status=satisfied" in satisfy_result.output
    assert "goal_abandoned event_id=" in abandon_result.output
    assert "status=abandoned" in abandon_result.output


def _goal_id_from_output(output: str) -> str:
    for part in output.split():
        if part.startswith("goal="):
            return part.split("=", 1)[1]
    raise AssertionError(output)
