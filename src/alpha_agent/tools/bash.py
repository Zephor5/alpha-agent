"""Opt-in local bash tool with policy, cancellation, and output governance."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from alpha_agent.config import BashToolConfig
from alpha_agent.tools.base import JSONValue, ToolExecutionContext, ToolResult
from alpha_agent.tools.shell.backend import ShellBackend
from alpha_agent.tools.shell.local import LocalShellBackend
from alpha_agent.tools.shell.output import govern_output
from alpha_agent.tools.shell.policy import BashExecutionPolicy, BashPolicyError, display_path
from alpha_agent.tools.shell.semantics import interpret_return_code

TRACE_COMMAND_CHARS = 240
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)([A-Z0-9_]*(?:API_KEY|TOKEN|SECRET|PASSWORD)[A-Z0-9_]*\s*=\s*)"
    r"([\"']?)[^ \n\"';&|]+"
)


class BashTool:
    """Execute local foreground shell commands through an explicit policy boundary."""

    name = "bash"
    description = (
        "Run a foreground local shell command for builds, tests, package management, Git, "
        "diagnostic scripts, or system inspection. Prefer specialized file/search/patch tools "
        "for file editing when available."
    )
    strict = True
    parameters: Mapping[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "description": {
                "type": "string",
                "description": "Brief purpose for trace and readable logs.",
            },
            "workdir": {
                "type": "string",
                "description": "Working directory, which must be inside an allowed workspace.",
            },
            "timeout_seconds": {
                "type": "integer",
                "minimum": 1,
                "description": "Foreground command timeout, capped by max_timeout_seconds.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        config: BashToolConfig | None = None,
        *,
        backend: ShellBackend | None = None,
        secret_values: Sequence[str] = (),
    ):
        self.config = config or BashToolConfig()
        self.backend = backend or LocalShellBackend()
        self.secret_values = tuple(str(value) for value in secret_values if str(value))

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Run a policy-approved command and return a structured JSON result."""

        policy = BashExecutionPolicy(self.config)
        try:
            prepared = policy.prepare(arguments)
        except BashPolicyError as exc:
            return self._blocked_result(str(exc), arguments)

        result = self.backend.execute(prepared.request, context)
        secrets = (*prepared.secret_values, *self.secret_values)
        governed = govern_output(
            result.stdout,
            result.stderr,
            max_chars=self.config.max_output_chars,
            secret_values=secrets,
        )
        interpretation = (
            interpret_return_code(
                prepared.request.command,
                result.exit_code,
                stderr=governed.stderr,
            )
            if result.status == "completed"
            else None
        )
        return self._result(
            status=result.status,
            exit_code=result.exit_code,
            stdout=governed.stdout,
            stderr=governed.stderr,
            elapsed_ms=result.elapsed_ms,
            workdir=prepared.request.display_workdir,
            shell=result.shell,
            truncated=governed.truncated,
            omitted_chars=governed.omitted_chars,
            return_code_interpretation=interpretation,
            error=result.error,
        )

    def trace_arguments(self, arguments: dict[str, Any]) -> Mapping[str, Any]:
        """Return trace-safe bash arguments without full heredocs or obvious secrets."""

        command = str(arguments.get("command") or "")
        redacted = self._redact_command(command)
        truncated = len(redacted) > TRACE_COMMAND_CHARS
        if truncated:
            redacted = redacted[:TRACE_COMMAND_CHARS] + "...[truncated]"
        trace: dict[str, Any] = {
            "command": redacted,
            "command_chars": len(command),
            "command_truncated": truncated,
        }
        for key in ("description", "workdir", "timeout_seconds"):
            if key in arguments:
                trace[key] = (
                    self._display_argument_workdir(arguments[key])
                    if key == "workdir"
                    else arguments[key]
                )
        return trace

    def _blocked_result(self, message: str, arguments: Mapping[str, Any]) -> ToolResult:
        workdir = self._display_argument_workdir(arguments.get("workdir"))
        return self._result(
            status="blocked",
            exit_code=None,
            stdout="",
            stderr=message,
            elapsed_ms=0,
            workdir=workdir,
            shell=None,
            truncated=False,
            omitted_chars=0,
            return_code_interpretation=None,
            error=message,
        )

    def _result(
        self,
        *,
        status: str,
        exit_code: int | None,
        stdout: str,
        stderr: str,
        elapsed_ms: int,
        workdir: str,
        shell: str | None,
        truncated: bool,
        omitted_chars: int,
        return_code_interpretation: str | None,
        error: str | None = None,
    ) -> ToolResult:
        output: dict[str, JSONValue] = {
            "status": status,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": elapsed_ms,
            "workdir": workdir,
            "truncated": truncated,
            "omitted_chars": omitted_chars,
            "return_code_interpretation": return_code_interpretation,
        }
        if status == "error" and error:
            output["error"] = error
        return ToolResult(
            name=self.name,
            output=output,
            metadata={
                "failed": False,
                "shell": shell,
                "elapsed_ms": elapsed_ms,
                "status": status,
                "exit_code": exit_code,
                "workdir": workdir,
            },
        )

    def _redact_command(self, command: str) -> str:
        redacted = command
        for secret in self.secret_values:
            if len(secret) >= 4:
                redacted = redacted.replace(secret, "[REDACTED]")
        return SECRET_ASSIGNMENT_RE.sub(r"\1\2[REDACTED]", redacted)

    def _display_argument_workdir(self, raw_workdir: Any) -> str:
        if not isinstance(raw_workdir, str) or not raw_workdir:
            return display_path(self.config.default_workdir)
        if "\x00" in raw_workdir:
            return "<workdir>"
        return display_path(Path(raw_workdir).expanduser())
