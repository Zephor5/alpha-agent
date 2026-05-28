"""Output cleanup and truncation for shell results."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
TRUNCATION_MARKER = "[output truncated: {omitted} chars omitted]"
MIN_SECRET_LENGTH = 4


@dataclass(frozen=True)
class GovernedOutput:
    """Cleaned shell output and truncation metadata."""

    stdout: str
    stderr: str
    truncated: bool
    omitted_chars: int


def govern_output(
    stdout: str,
    stderr: str,
    *,
    max_chars: int,
    secret_values: Iterable[str] = (),
) -> GovernedOutput:
    """Strip ANSI escapes, redact known secrets, and bound total output size."""

    cleaned_stdout = _redact(_strip_ansi(stdout), secret_values)
    cleaned_stderr = _redact(_strip_ansi(stderr), secret_values)
    total_chars = len(cleaned_stdout) + len(cleaned_stderr)
    if total_chars <= max_chars:
        return GovernedOutput(
            stdout=cleaned_stdout,
            stderr=cleaned_stderr,
            truncated=False,
            omitted_chars=0,
        )

    stdout_limit, stderr_limit = _stream_limits(cleaned_stdout, cleaned_stderr, max_chars)
    truncated_stdout, stdout_omitted = _truncate_text(cleaned_stdout, stdout_limit)
    truncated_stderr, stderr_omitted = _truncate_text(cleaned_stderr, stderr_limit)
    return GovernedOutput(
        stdout=truncated_stdout,
        stderr=truncated_stderr,
        truncated=True,
        omitted_chars=stdout_omitted + stderr_omitted,
    )


def _strip_ansi(value: str) -> str:
    return ANSI_ESCAPE_RE.sub("", value)


def _redact(value: str, secret_values: Iterable[str]) -> str:
    redacted = value
    for secret in _unique_secrets(secret_values):
        redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _unique_secrets(secret_values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for secret in secret_values:
        text = str(secret)
        if len(text) < MIN_SECRET_LENGTH or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return tuple(result)


def _stream_limits(stdout: str, stderr: str, max_chars: int) -> tuple[int, int]:
    if not stdout:
        return 0, max_chars
    if not stderr:
        return max_chars, 0
    total = len(stdout) + len(stderr)
    stdout_limit = max(1, int(max_chars * (len(stdout) / total)))
    stderr_limit = max(1, max_chars - stdout_limit)
    return stdout_limit, stderr_limit


def _truncate_text(value: str, max_chars: int) -> tuple[str, int]:
    if max_chars <= 0:
        return "", len(value)
    if len(value) <= max_chars:
        return value, 0
    marker_template = "\n" + TRUNCATION_MARKER + "\n"
    marker_overhead = len(marker_template.format(omitted=0))
    if max_chars <= marker_overhead + 2:
        return value[:max_chars], len(value) - max_chars
    body_budget = max_chars - marker_overhead
    head_chars = max(1, int(body_budget * 0.4))
    tail_chars = max(1, body_budget - head_chars)
    omitted = len(value) - head_chars - tail_chars
    marker = marker_template.format(omitted=omitted)
    return value[:head_chars] + marker + value[-tail_chars:], omitted
