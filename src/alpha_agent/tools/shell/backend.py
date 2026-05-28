"""Shell backend contracts for local command execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from alpha_agent.tools.base import ToolExecutionContext


@dataclass(frozen=True)
class ShellRequest:
    """A policy-approved foreground shell command."""

    command: str
    workdir: Path
    display_workdir: str
    env: Mapping[str, str]
    timeout_seconds: int
    output_capture_bytes: int


@dataclass(frozen=True)
class ShellResult:
    """Raw process execution result returned by a shell backend."""

    status: str
    exit_code: int | None
    stdout: str
    stderr: str
    elapsed_ms: int
    shell: str | None = None
    error: str | None = None


class ShellBackend(Protocol):
    """Backend protocol for executing a foreground shell command."""

    def execute(
        self,
        request: ShellRequest,
        context: ToolExecutionContext,
    ) -> ShellResult:
        """Execute a command and return a structured shell result."""
