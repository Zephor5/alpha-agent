from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest

from alpha_agent.config import BashToolConfig
from alpha_agent.tools.base import ToolExecutionContext
from alpha_agent.tools.bash import BashTool
from alpha_agent.tools.shell.policy import blocked_command_reason


def _config(workdir: Path, **overrides: object) -> BashToolConfig:
    values = {
        "enabled": True,
        "default_workdir": workdir.resolve(),
        "allowed_workdirs": (workdir.resolve(),),
        "default_timeout_seconds": 5,
        "max_timeout_seconds": 10,
        "max_output_chars": 30000,
        "env_passthrough": (),
    }
    values.update(overrides)
    return BashToolConfig(**values)


def _context(tmp_path: Path, check_canceled=lambda _stage: None) -> ToolExecutionContext:
    return ToolExecutionContext(
        session_id="s1",
        tool_call_id="call_1",
        output_dir=tmp_path / "tool-results",
        check_canceled=check_canceled,
    )


def test_bash_tool_exposes_strict_schema() -> None:
    tool = BashTool()

    assert tool.name == "bash"
    assert tool.strict is True
    assert tool.parameters["required"] == ["command"]
    assert "background" not in tool.parameters["properties"]
    assert tool.parameters["additionalProperties"] is False


def test_bash_tool_echo_returns_structured_completed_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = BashTool(config=_config(workspace))

    result = tool.run({"command": "printf 'ok\\n'"}, _context(tmp_path))

    assert result.name == "bash"
    assert result.output["status"] == "completed"
    assert result.output["exit_code"] == 0
    assert result.output["stdout"] == "ok\n"
    assert result.output["stderr"] == ""
    assert result.output["truncated"] is False
    assert result.output["omitted_chars"] == 0
    assert result.output["return_code_interpretation"] is None
    assert str(workspace) not in result.output["workdir"]
    assert result.metadata["status"] == "completed"
    assert result.metadata["exit_code"] == 0
    assert result.metadata["shell"] in {"bash", "sh"}


def test_bash_tool_nonzero_exit_code_is_a_completed_tool_result(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = BashTool(config=_config(workspace))

    result = tool.run({"command": "printf 'bad' >&2; exit 7"}, _context(tmp_path))

    assert result.output["status"] == "completed"
    assert result.output["exit_code"] == 7
    assert result.output["stderr"] == "bad"
    assert result.metadata["failed"] is False


def test_bash_tool_times_out_and_kills_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = BashTool(config=_config(workspace, default_timeout_seconds=1, max_timeout_seconds=1))

    result = tool.run({"command": "sleep 5"}, _context(tmp_path))

    assert result.output["status"] == "timeout"
    assert result.output["exit_code"] is not None
    assert result.metadata["status"] == "timeout"


def test_bash_tool_timeout_kills_descendant_holding_output_pipes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = BashTool(config=_config(workspace, default_timeout_seconds=1, max_timeout_seconds=1))
    code = 'import subprocess; subprocess.Popen(["sleep", "3"]); print("done")'
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = tool.run({"command": command}, _context(tmp_path))

    assert result.output["status"] == "timeout"
    assert result.output["elapsed_ms"] < 2500


def test_bash_tool_cancels_running_command(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class CancelAfterFirstCheck:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, stage: str) -> None:
            if stage != "during_tool":
                return
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("cancel requested")

    tool = BashTool(config=_config(workspace))
    result = tool.run({"command": "sleep 5"}, _context(tmp_path, CancelAfterFirstCheck()))

    assert result.output["status"] == "canceled"
    assert result.metadata["status"] == "canceled"


def test_bash_tool_blocks_workdir_outside_allowed_roots(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    tool = BashTool(config=_config(workspace))

    result = tool.run(
        {"command": "printf nope", "workdir": str(outside)},
        _context(tmp_path),
    )

    assert result.output["status"] == "blocked"
    assert "allowed workdirs" in result.output["stderr"]


def test_bash_tool_rejects_unknown_arguments(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    tool = BashTool(config=_config(workspace))

    result = tool.run({"command": "printf ok", "background": True}, _context(tmp_path))

    assert result.output["status"] == "blocked"
    assert "Unknown bash argument" in result.output["stderr"]


def test_bash_tool_does_not_trace_or_return_absolute_workdir(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    tool = BashTool(config=_config(workspace))

    trace = tool.trace_arguments({"command": "printf ok", "workdir": str(outside)})
    result = tool.run(
        {"command": "printf ok", "workdir": str(outside)},
        _context(tmp_path),
    )

    assert str(outside) not in json.dumps(trace)
    assert result.output["status"] == "blocked"
    assert str(outside) not in json.dumps(result.output)


def test_bash_tool_trace_arguments_summarizes_large_or_secret_commands() -> None:
    tool = BashTool(secret_values=["literal-secret"])
    command = "TOKEN=literal-secret printf ok\n" + ("x" * 600)

    trace = tool.trace_arguments({"command": command, "description": "run test"})

    dumped = json.dumps(trace)
    assert "literal-secret" not in dumped
    assert trace["command_truncated"] is True
    assert trace["command_chars"] == len(command)


def test_bash_tool_cleans_redacts_and_truncates_output(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    command = (
        "printf '\\033[31mtop-secret-value\\033[0m'; "
        "i=0; while [ $i -lt 200 ]; do printf x; i=$((i+1)); done"
    )
    tool = BashTool(
        config=_config(workspace, max_output_chars=80),
        secret_values=["top-secret-value"],
    )

    result = tool.run({"command": command}, _context(tmp_path))

    assert result.output["status"] == "completed"
    assert "\x1b" not in result.output["stdout"]
    assert "[REDACTED]" in result.output["stdout"]
    assert "[output truncated:" in result.output["stdout"]
    assert result.output["truncated"] is True
    assert result.output["omitted_chars"] > 0


@pytest.mark.parametrize(
    "command",
    [
        "sudo true",
        "command sudo true",
        "env sudo true",
        "env FOO=bar sudo true",
        "exec sudo true",
        "printf ok\nsudo true",
        "printf ok\ngit reset --hard",
        "echo $(git reset --hard)",
        "echo `sudo true`",
        "if true; then sudo true; fi",
        "{ sudo true; }",
        "{ git reset --hard; }",
        "{ rm -fr /; }",
        "{ chmod -R 777 .; }",
        "function f { sudo true; }; f",
        "f(){ sudo true; }; f",
        "eval sudo true",
        'eval "sudo true"',
        "bash -lc 'sudo true'",
        "sh -c 'vim file.txt'",
        "nohup sleep 1 &",
        "vim file.txt",
        "git reset --hard",
        "git -C . reset --hard",
        "git clean -fd",
        "git -C . clean -fd",
        "chmod -R 777 .",
        "chown -R user .",
        "gh auth login --with-token",
    ],
)
def test_policy_blocks_dangerous_or_interactive_commands(command: str) -> None:
    assert blocked_command_reason(command)


@pytest.mark.parametrize(
    "command",
    ["git status", "git -C . status", "env FOO=bar printf ok", "uv run pytest --version"],
)
def test_policy_does_not_block_normal_development_commands(command: str) -> None:
    assert blocked_command_reason(command) is None


def test_bash_tool_interprets_common_nonzero_exit_codes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "data.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "left.txt").write_text("left\n", encoding="utf-8")
    (workspace / "right.txt").write_text("right\n", encoding="utf-8")
    tool = BashTool(config=_config(workspace))

    grep_result = tool.run({"command": "grep beta data.txt"}, _context(tmp_path))
    diff_result = tool.run(
        {
            "command": "diff "
            + shlex.quote("left.txt")
            + " "
            + shlex.quote("right.txt")
        },
        _context(tmp_path),
    )
    grep_error = tool.run({"command": "grep beta missing.txt"}, _context(tmp_path))

    assert grep_result.output["exit_code"] == 1
    assert grep_result.output["return_code_interpretation"] == "No matches found"
    assert diff_result.output["exit_code"] == 1
    assert diff_result.output["return_code_interpretation"] == "Files differ"
    assert grep_error.output["exit_code"] == 2
    assert grep_error.output["return_code_interpretation"] is None


def test_bash_tool_uses_clean_environment_and_opt_in_passthrough(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("ALPHA_VISIBLE_ENV", "shown")
    monkeypatch.setenv("ALPHA_TAVILY_API_KEY", "secret")
    tool = BashTool(
        config=_config(
            workspace,
            env_passthrough=("ALPHA_VISIBLE_ENV", "ALPHA_TAVILY_API_KEY"),
        )
    )

    result = tool.run(
        {
            "command": (
                "printf '%s|%s' "
                "\"${ALPHA_VISIBLE_ENV:-missing}\" "
                "\"${ALPHA_TAVILY_API_KEY:-missing}\""
            )
        },
        _context(tmp_path),
    )

    assert result.output["stdout"] == "shown|missing"
