"""Policy checks for the opt-in local bash tool."""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_agent.config import BashToolConfig
from alpha_agent.tools.shell.backend import ShellRequest

ALLOWED_ARGUMENTS = {"command", "description", "workdir", "timeout_seconds"}
BASE_ENV_NAMES = ("PATH", "HOME", "LANG", "LC_ALL", "SHELL", "TMPDIR")
EXACT_SECRET_ENV_NAMES = {
    "ALPHA_CODEX_ACCESS_TOKEN",
    "ALPHA_COMPATIBLE_API_KEY",
    "ALPHA_DEEPSEEK_API_KEY",
    "ALPHA_TAVILY_API_KEY",
    "TAVILY_API_KEY",
}
SECRET_ENV_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
EXECUTION_BOUNDARY_TOKENS = {";", "&&", "||", "|"}
PRIVILEGE_COMMANDS = {"doas", "pkexec", "sudo"}
BACKGROUND_COMMANDS = {"disown", "nohup", "setsid"}
INTERACTIVE_COMMANDS = {"htop", "less", "more", "nano", "top", "vi", "vim"}
DYNAMIC_COMMANDS = {"eval"}
SHELL_COMMANDS = {"bash", "sh", "zsh"}
GIT_OPTIONS_WITH_VALUE = {
    "-C",
    "-c",
    "--config-env",
    "--exec-path",
    "--git-dir",
    "--namespace",
    "--super-prefix",
    "--work-tree",
}
GIT_OPTIONS_WITH_INLINE_VALUE = tuple(f"{option}=" for option in GIT_OPTIONS_WITH_VALUE)
COMMAND_START_PATTERN = r"(?:^|[\n;&|()`{}]|\bthen\b|\bdo\b|\belse\b|\belif\b)\s*"
TEXTUAL_WRAPPER_PATTERN = (
    r"(?:(?:command|exec)(?:\s+-[A-Za-z](?:\s+\S+)?)?\s+|"
    r"env(?:\s+(?:-[^\s]+(?:\s+\S+)?|[A-Za-z_][A-Za-z0-9_]*=\S+))*\s+)*"
)
TEXTUAL_DANGEROUS_COMMAND_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?(?P<command>doas|pkexec|sudo|disown|nohup|setsid|htop|less|more|nano|top|vi|vim)\b"
)
TEXTUAL_DYNAMIC_COMMAND_RE = re.compile(
    COMMAND_START_PATTERN + TEXTUAL_WRAPPER_PATTERN + r"(?P<command>eval)\b"
)
TEXTUAL_GIT_RESET_HARD_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?git\b(?:(?![\n;&|`)]).)*\breset\b(?:(?![\n;&|`)]).)*--hard\b"
)
TEXTUAL_GIT_CLEAN_FD_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?git\b(?:(?![\n;&|`)]).)*\bclean\b(?:(?![\n;&|`)]).)*"
    + r"-(?=[A-Za-z-]*f)(?=[A-Za-z-]*d)[A-Za-z-]*\b"
)
TEXTUAL_ROOT_RM_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?rm\b(?:(?![\n;&|`)]).)*"
    + r"-(?=[A-Za-z-]*r)(?=[A-Za-z-]*f)[A-Za-z-]*\s+(?:/|\*/?)"
)
TEXTUAL_CHMOD_777_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?chmod\b(?:(?![\n;&|`)]).)*-(?:[A-Za-z-]*R|-[A-Za-z-]*recursive)\b"
    + r"(?:(?![\n;&|`)]).)*\b777\b"
)
TEXTUAL_CHOWN_RECURSIVE_RE = re.compile(
    r"(?<![\w.-])"
    + r"(?:\S+/)?chown\b(?:(?![\n;&|`)]).)*-(?:[A-Za-z-]*R|-[A-Za-z-]*recursive)\b"
)
TEXTUAL_GH_AUTH_TOKEN_RE = re.compile(
    r"(?<![\w.-])" + r"(?:\S+/)?gh\s+auth\s+login\s+--with-token\b"
)


class BashPolicyError(ValueError):
    """Raised when a bash invocation is rejected before execution."""


@dataclass(frozen=True)
class PreparedBashCommand:
    """Policy-approved shell request plus redaction inputs."""

    request: ShellRequest
    secret_values: tuple[str, ...]


class BashExecutionPolicy:
    """Validate bash tool arguments, workdir, environment, and command risk."""

    def __init__(
        self,
        config: BashToolConfig | None = None,
        *,
        environment: Mapping[str, str] | None = None,
    ):
        self.config = config or BashToolConfig()
        self.environment = dict(os.environ if environment is None else environment)
        self.allowed_workdirs = _resolve_allowed_workdirs(self.config.allowed_workdirs)

    def prepare(self, arguments: Mapping[str, Any]) -> PreparedBashCommand:
        """Return a shell request or raise a policy error."""

        unknown = sorted(set(arguments) - ALLOWED_ARGUMENTS)
        if unknown:
            names = ", ".join(unknown)
            raise BashPolicyError(f"Unknown bash argument(s): {names}")
        command = _required_string(arguments.get("command"), "command")
        reason = blocked_command_reason(command)
        if reason:
            raise BashPolicyError(reason)
        workdir = self._resolve_workdir(arguments.get("workdir"))
        timeout_seconds = self._timeout_seconds(arguments.get("timeout_seconds"))
        env = self._sanitized_env()
        return PreparedBashCommand(
            request=ShellRequest(
                command=command,
                workdir=workdir,
                display_workdir=display_path(workdir),
                env=env,
                timeout_seconds=timeout_seconds,
                output_capture_bytes=max(65536, self.config.max_output_chars * 4),
            ),
            secret_values=self.secret_values(),
        )

    def secret_values(self) -> tuple[str, ...]:
        """Return secret values known from blocked process environment names."""

        values: list[str] = []
        seen: set[str] = set()
        for name, value in self.environment.items():
            if not _is_secret_env_name(name) or not value or value in seen:
                continue
            values.append(value)
            seen.add(value)
        return tuple(values)

    def _resolve_workdir(self, raw_workdir: Any) -> Path:
        if raw_workdir is None or raw_workdir == "":
            path = self.config.default_workdir
        else:
            path = Path(_string_value(raw_workdir, "workdir")).expanduser()
        if "\x00" in str(path):
            raise BashPolicyError("workdir must not contain NUL characters")
        resolved = path.resolve()
        if not resolved.exists():
            raise BashPolicyError("workdir does not exist")
        if not resolved.is_dir():
            raise BashPolicyError("workdir must be a directory")
        inside_allowed_root = any(
            resolved == root or resolved.is_relative_to(root) for root in self.allowed_workdirs
        )
        if not inside_allowed_root:
            raise BashPolicyError("workdir must be within allowed workdirs")
        return resolved

    def _timeout_seconds(self, raw_timeout: Any) -> int:
        if raw_timeout is None or raw_timeout == "":
            timeout = self.config.default_timeout_seconds
        else:
            try:
                timeout = int(raw_timeout)
            except (TypeError, ValueError) as exc:
                raise BashPolicyError("timeout_seconds must be an integer") from exc
        if timeout < 1:
            raise BashPolicyError("timeout_seconds must be at least 1")
        return min(timeout, self.config.max_timeout_seconds)

    def _sanitized_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        for name in BASE_ENV_NAMES:
            value = self.environment.get(name)
            if value is not None and not _is_secret_env_name(name):
                env[name] = value
        env.setdefault("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        for name in self.config.env_passthrough:
            if _is_secret_env_name(name):
                continue
            value = self.environment.get(name)
            if value is not None:
                env[name] = value
        return env


def blocked_command_reason(command: str) -> str | None:
    """Return a block reason for known dangerous, privileged, or interactive commands."""

    return _blocked_command_reason(command, depth=0)


def _blocked_command_reason(command: str, *, depth: int) -> str | None:
    if "\x00" in command:
        return "Command must not contain NUL characters"
    textual_reason = _textual_block_reason(command)
    if textual_reason:
        return textual_reason
    tokens = _command_tokens(command)
    if not tokens:
        return "command is required"
    if any(token == "&" for token in tokens):
        return "Background commands are not supported by bash tool v1"
    for segment in _command_segments(tokens):
        reason = _blocked_segment_reason(segment, depth=depth)
        if reason:
            return reason
    return None


def _textual_block_reason(command: str) -> str | None:
    source = _mask_single_quoted_text(command)
    dangerous_match = TEXTUAL_DANGEROUS_COMMAND_RE.search(source)
    if dangerous_match:
        executable = Path(dangerous_match.group("command")).name
        if executable in PRIVILEGE_COMMANDS:
            return f"Privileged command is blocked: {executable}"
        if executable in BACKGROUND_COMMANDS:
            return f"Background command is blocked: {executable}"
        if executable in INTERACTIVE_COMMANDS:
            return f"Interactive command is blocked: {executable}"
    dynamic_match = TEXTUAL_DYNAMIC_COMMAND_RE.search(source)
    if dynamic_match:
        executable = Path(dynamic_match.group("command")).name
        return f"Dynamic shell command is blocked: {executable}"
    if TEXTUAL_GIT_RESET_HARD_RE.search(source):
        return "Destructive git reset --hard is blocked"
    if TEXTUAL_GIT_CLEAN_FD_RE.search(source):
        return "Destructive git clean -fd is blocked"
    if TEXTUAL_ROOT_RM_RE.search(source):
        return "Destructive root removal is blocked"
    if TEXTUAL_CHMOD_777_RE.search(source):
        return "Recursive chmod 777 is blocked"
    if TEXTUAL_CHOWN_RECURSIVE_RE.search(source):
        return "Recursive chown is blocked"
    if TEXTUAL_GH_AUTH_TOKEN_RE.search(source):
        return "Token-based interactive GitHub login is blocked"
    return None


def _mask_single_quoted_text(command: str) -> str:
    chars: list[str] = []
    in_single_quote = False
    for char in command:
        if char == "'":
            in_single_quote = not in_single_quote
            chars.append(" ")
        elif in_single_quote:
            chars.append(" ")
        else:
            chars.append(char)
    return "".join(chars)


def display_path(path: Path) -> str:
    """Return a non-machine-specific display path for tool output and traces."""

    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        relative = resolved.relative_to(cwd)
        return "." if str(relative) == "." else relative.as_posix()
    except ValueError:
        pass
    home = Path.home().resolve()
    try:
        relative = resolved.relative_to(home)
        return "~" if str(relative) == "." else "~/" + relative.as_posix()
    except ValueError:
        return "<allowed-workdir>"


def _blocked_segment_reason(segment: Sequence[str], *, depth: int) -> str | None:
    if not segment:
        return None
    executable = Path(segment[0]).name
    wrapped = _wrapped_segment(segment)
    if wrapped:
        return _blocked_segment_reason(wrapped, depth=depth)
    shell_command = _shell_c_command(segment)
    if executable in SHELL_COMMANDS and shell_command and depth < 3:
        return _blocked_command_reason(shell_command, depth=depth + 1)
    if executable in PRIVILEGE_COMMANDS:
        return f"Privileged command is blocked: {executable}"
    if executable in BACKGROUND_COMMANDS:
        return f"Background command is blocked: {executable}"
    if executable in INTERACTIVE_COMMANDS:
        return f"Interactive command is blocked: {executable}"
    if executable in DYNAMIC_COMMANDS:
        return f"Dynamic shell command is blocked: {executable}"
    if executable == "rm" and _is_root_recursive_rm(segment):
        return "Destructive root removal is blocked"
    if executable == "git":
        return _git_block_reason(segment)
    if executable == "chmod" and _has_recursive_flag(segment) and "777" in segment:
        return "Recursive chmod 777 is blocked"
    if executable == "chown" and _has_recursive_flag(segment):
        return "Recursive chown is blocked"
    if executable == "gh" and segment[:4] == ["gh", "auth", "login", "--with-token"]:
        return "Token-based interactive GitHub login is blocked"
    return None


def _git_block_reason(segment: Sequence[str]) -> str | None:
    subcommand_index = _git_subcommand_index(segment)
    if subcommand_index is None:
        return None
    subcommand = segment[subcommand_index]
    subcommand_args = segment[subcommand_index + 1 :]
    if subcommand == "reset" and "--hard" in subcommand_args:
        return "Destructive git reset --hard is blocked"
    if subcommand == "clean":
        for token in subcommand_args:
            if token.startswith("-") and "f" in token and "d" in token:
                return "Destructive git clean -fd is blocked"
    return None


def _wrapped_segment(segment: Sequence[str]) -> list[str]:
    executable = Path(segment[0]).name if segment else ""
    if executable == "command":
        return _shell_command_builtin_target(segment)
    if executable == "exec":
        return _exec_target(segment)
    if executable == "env":
        return _env_target(segment)
    return []


def _shell_command_builtin_target(segment: Sequence[str]) -> list[str]:
    index = 1
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token in {"-v", "-V"}:
            return []
        if token.startswith("-"):
            index += 1
            continue
        break
    return list(segment[index:])


def _exec_target(segment: Sequence[str]) -> list[str]:
    index = 1
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token in {"-a"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        break
    return list(segment[index:])


def _env_target(segment: Sequence[str]) -> list[str]:
    index = 1
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token in {"-0", "-i", "--ignore-environment", "--null"}:
            index += 1
            continue
        if token in {"-C", "-S", "-u", "--chdir", "--split-string", "--unset"}:
            index += 2
            continue
        if token.startswith(("--chdir=", "--split-string=", "--unset=")):
            index += 1
            continue
        if "=" in token and not token.startswith("-"):
            index += 1
            continue
        break
    return list(segment[index:])


def _shell_c_command(segment: Sequence[str]) -> str | None:
    if not segment or Path(segment[0]).name not in SHELL_COMMANDS:
        return None
    index = 1
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token.startswith("-") and "c" in token:
            return segment[index + 1] if index + 1 < len(segment) else None
        if token.startswith("-"):
            index += 1
            continue
        break
    return None


def _git_subcommand_index(segment: Sequence[str]) -> int | None:
    index = 1
    while index < len(segment):
        token = segment[index]
        if token == "--":
            index += 1
            break
        if token in GIT_OPTIONS_WITH_VALUE:
            index += 2
            continue
        if token.startswith(GIT_OPTIONS_WITH_INLINE_VALUE):
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        return index
    return index if index < len(segment) else None


def _is_root_recursive_rm(segment: Sequence[str]) -> bool:
    has_recursive = False
    has_force = False
    targets: list[str] = []
    for token in segment[1:]:
        if token.startswith("-"):
            has_recursive = has_recursive or "r" in token or "R" in token
            has_force = has_force or "f" in token
        else:
            targets.append(token)
    return has_recursive and has_force and any(target in {"/", "/*"} for target in targets)


def _has_recursive_flag(segment: Sequence[str]) -> bool:
    return any(token in {"-R", "-r", "--recursive"} for token in segment[1:])


def _command_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return []


def _command_segments(tokens: Sequence[str]) -> list[list[str]]:
    segments: list[list[str]] = [[]]
    for token in tokens:
        if token in EXECUTION_BOUNDARY_TOKENS:
            if segments[-1]:
                segments.append([])
            continue
        segments[-1].append(token)
    return [segment for segment in segments if segment]


def _required_string(value: Any, key: str) -> str:
    text = _string_value(value, key).strip()
    if not text:
        raise BashPolicyError(f"{key} is required")
    return text


def _string_value(value: Any, key: str) -> str:
    if not isinstance(value, str):
        raise BashPolicyError(f"{key} must be a string")
    if "\x00" in value:
        raise BashPolicyError(f"{key} must not contain NUL characters")
    return value


def _resolve_allowed_workdirs(paths: Sequence[Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        root = path.expanduser().resolve()
        if root in seen:
            continue
        resolved.append(root)
        seen.add(root)
    if not resolved:
        raise BashPolicyError("tools.bash.allowed_workdirs must not be empty")
    return tuple(resolved)


def _is_secret_env_name(name: str) -> bool:
    normalized = name.strip().upper()
    return normalized in EXACT_SECRET_ENV_NAMES or normalized.endswith(SECRET_ENV_SUFFIXES)
