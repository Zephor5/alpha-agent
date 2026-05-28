"""Human-readable interpretations for common shell exit code conventions."""

from __future__ import annotations

import shlex
from pathlib import Path

SEARCH_COMMANDS = {"ag", "ack", "grep", "rg"}


def interpret_return_code(command: str, exit_code: int | None, *, stderr: str = "") -> str | None:
    """Explain common non-zero exit codes without changing process status."""

    if exit_code is None or exit_code == 0:
        return None
    tokens = _tokens(command)
    executable = _executable(tokens)
    if not executable:
        return None
    if executable in SEARCH_COMMANDS and exit_code == 1 and not stderr.strip():
        return "No matches found"
    if executable == "diff" and exit_code == 1:
        return "Files differ"
    if executable == "find" and exit_code == 1:
        return "Some paths could not be accessed"
    if executable in {"test", "["} and exit_code == 1:
        return "Condition evaluated to false"
    return None


def _tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _executable(tokens: list[str]) -> str | None:
    if not tokens:
        return None
    return Path(tokens[0]).name
