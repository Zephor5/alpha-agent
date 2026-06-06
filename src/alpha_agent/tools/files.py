"""Local file tools."""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_agent.config import FileToolConfig
from alpha_agent.tools.base import (
    JSONValue,
    ToolAvailability,
    ToolExecutionContext,
    ToolResult,
    ToolSpec,
)

FILE_LIST_TOOL_NAME = "file_list"
FILE_PATCH_TOOL_NAME = "file_patch"
FILE_READ_TOOL_NAME = "file_read"
FILE_SEARCH_TOOL_NAME = "file_search"

MAX_CONTEXT_LINES = 5
MAX_SEARCH_LINE_CHARS = 500
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)


class FileToolError(ValueError):
    """Raised when a file tool request violates local file policy."""


@dataclass(frozen=True)
class _ResolvedPath:
    path: Path
    display: str


class _FileToolBase:
    def __init__(self, config: FileToolConfig | None = None):
        self.config = config or FileToolConfig()
        self.allowed_roots = _normalized_roots(self.config.allowed_roots)
        self.write_roots = _normalized_roots(
            self.config.write_roots,
            required=False,
            label="tools.files.write_roots",
        )

    def check_available(self) -> ToolAvailability:
        """Return whether local file reading is enabled."""

        if not self.config.enabled:
            return ToolAvailability.unavailable("tools.files.enabled is false")
        if not self.allowed_roots:
            return ToolAvailability.unavailable("tools.files.allowed_roots is empty")
        return ToolAvailability()

    def _resolve_argument_path(
        self,
        value: Any,
        *,
        default: str | None = None,
        must_exist: bool = True,
    ) -> _ResolvedPath:
        raw = default if value is None or value == "" else value
        if not isinstance(raw, str):
            raise FileToolError("path must be a string")
        if "\x00" in raw:
            raise FileToolError("path must not contain NUL characters")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.allowed_roots[0] / candidate
        resolved = candidate.resolve(strict=must_exist)
        if not _is_inside_allowed(resolved, self.allowed_roots):
            raise FileToolError("path is outside tools.files.allowed_roots")
        policy_path = candidate if candidate.is_symlink() else resolved
        return _ResolvedPath(path=policy_path, display=self._display_path(policy_path))

    def _display_path(self, path: Path) -> str:
        for root in self.allowed_roots:
            if path == root:
                return "."
            if path.is_relative_to(root):
                return path.relative_to(root).as_posix()
        return path.name


class FileListTool(_FileToolBase):
    """List directory entries inside configured roots."""

    @property
    def spec(self) -> ToolSpec:
        """Return the file listing spec derived from current config."""

        return ToolSpec(
            name=FILE_LIST_TOOL_NAME,
            description=(
                "List files and directories under configured allowed roots. Returns compact "
                "metadata only and skips common large internal directories."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path relative to an allowed root.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to recursively list nested entries.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional fnmatch glob applied to relative paths or names.",
                    },
                    "max_entries": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.config.max_list_entries,
                        "description": "Maximum entries to return.",
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=True,
            destructive=False,
            concurrency_safe=True,
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Return a stable JSON directory listing."""

        del context
        resolved = self._resolve_argument_path(arguments.get("path"), default=".")
        if resolved.path.is_symlink():
            raise FileToolError("symlink directories are not followed")
        if not resolved.path.is_dir():
            raise FileToolError("path must be a directory")
        recursive = bool(arguments.get("recursive", False))
        pattern = _optional_text(arguments.get("glob"))
        max_entries = _bounded_int(
            arguments.get("max_entries"),
            default=self.config.max_list_entries,
            minimum=1,
            maximum=self.config.max_list_entries,
            field_name="max_entries",
        )

        entries: list[JSONValue] = []
        truncated = False
        for child in self._iter_entries(resolved.path, recursive=recursive):
            display = self._display_path(child)
            if pattern and not _matches_glob(display, child.name, pattern):
                continue
            if len(entries) >= max_entries:
                truncated = True
                break
            entries.append(_entry_payload(child, display))

        return ToolResult(
            name=self.spec.name,
            output={
                "path": resolved.display,
                "entries": entries,
                "truncated": truncated,
            },
            metadata={"entry_count": len(entries), "truncated": truncated},
        )

    def _iter_entries(self, root: Path, *, recursive: bool) -> Iterator[Path]:
        if not recursive:
            for child in sorted(root.iterdir(), key=lambda item: item.name):
                if self._entry_allowed(child):
                    yield child
            return

        yield from self._iter_recursive_entries(root)

    def _iter_recursive_entries(self, root: Path) -> Iterator[Path]:
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            if not self._entry_allowed(child):
                continue
            yield child
            if (
                child.is_dir()
                and not child.is_symlink()
                and child.name not in DEFAULT_EXCLUDED_DIRS
            ):
                yield from self._iter_recursive_entries(child)

    def _entry_allowed(self, path: Path) -> bool:
        if path.name in DEFAULT_EXCLUDED_DIRS and path.is_dir():
            return False
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            return False
        return _is_inside_allowed(resolved, self.allowed_roots)


class FileReadTool(_FileToolBase):
    """Read text files inside configured roots."""

    @property
    def spec(self) -> ToolSpec:
        """Return the file reading spec derived from current config."""

        return ToolSpec(
            name=FILE_READ_TOOL_NAME,
            description=(
                "Read a UTF-8 text file from configured allowed roots. Supports line ranges "
                "or character limits and rejects binary files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Text file path relative to an allowed root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based first line to read.",
                    },
                    "end_line": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Optional 1-based inclusive last line to read.",
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.config.max_read_chars,
                        "description": "Maximum content characters to return.",
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=True,
            destructive=False,
            concurrency_safe=True,
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Return text content and stable file metadata."""

        del context
        resolved = self._resolve_argument_path(arguments.get("path"))
        if resolved.path.is_symlink():
            raise FileToolError("symlink files are not followed")
        if not resolved.path.is_file():
            raise FileToolError("path must be a file")
        data = _read_file_bytes(resolved.path, max_file_bytes=self.config.max_file_bytes)
        content = _decode_text(data)
        lines = content.splitlines(keepends=True)
        requested_start = _optional_positive_int(arguments.get("start_line"), "start_line") or 1
        requested_end = _optional_positive_int(arguments.get("end_line"), "end_line")
        if requested_end is not None and requested_end < requested_start:
            raise FileToolError("end_line must be greater than or equal to start_line")
        max_chars = _bounded_int(
            arguments.get("max_chars"),
            default=self.config.max_read_chars,
            minimum=1,
            maximum=self.config.max_read_chars,
            field_name="max_chars",
        )

        start_index = min(requested_start - 1, len(lines))
        end_index = len(lines) if requested_end is None else min(requested_end, len(lines))
        selected = "".join(lines[start_index:end_index])
        line_truncated = requested_end is not None and requested_end < len(lines)
        char_truncated = len(selected) > max_chars
        if char_truncated:
            selected = selected[:max_chars]
        returned_end_line = _returned_end_line(
            selected,
            start_line=requested_start,
            has_lines=bool(lines),
            empty_end_line=min(start_index, len(lines)) if lines else 0,
        )

        payload: dict[str, JSONValue] = {
            "path": resolved.display,
            "content": selected,
            "start_line": requested_start,
            "end_line": returned_end_line,
            "truncated": line_truncated or char_truncated,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        return ToolResult(
            name=self.spec.name,
            output=payload,
            metadata={
                "path": resolved.display,
                "size": len(data),
                "truncated": bool(payload["truncated"]),
            },
        )


class FileSearchTool(_FileToolBase):
    """Search UTF-8 text files inside configured roots."""

    @property
    def spec(self) -> ToolSpec:
        """Return the file search spec derived from current config."""

        return ToolSpec(
            name=FILE_SEARCH_TOOL_NAME,
            description=(
                "Search UTF-8 text files under configured allowed roots. Returns bounded "
                "literal substring matches with optional surrounding context."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal text to search for, case-insensitive.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path relative to an allowed root.",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional fnmatch glob applied to relative paths or names.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_CONTEXT_LINES,
                        "description": "Number of surrounding lines to include.",
                    },
                    "max_matches": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.config.max_search_matches,
                        "description": "Maximum matches to return.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=True,
            destructive=False,
            concurrency_safe=True,
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Return bounded search matches."""

        del context
        query = str(arguments.get("query") or "")
        if not query:
            raise FileToolError("query is required")
        resolved = self._resolve_argument_path(arguments.get("path"), default=".")
        pattern = _optional_text(arguments.get("glob"))
        context_lines = _bounded_int(
            arguments.get("context_lines"),
            default=0,
            minimum=0,
            maximum=MAX_CONTEXT_LINES,
            field_name="context_lines",
        )
        max_matches = _bounded_int(
            arguments.get("max_matches"),
            default=self.config.max_search_matches,
            minimum=1,
            maximum=self.config.max_search_matches,
            field_name="max_matches",
        )

        matches: list[JSONValue] = []
        truncated = False
        for file_path in self._iter_search_files(resolved.path, pattern):
            try:
                text = _decode_text(
                    _read_file_bytes(file_path, max_file_bytes=self.config.max_file_bytes)
                )
            except FileToolError:
                if resolved.path.is_file():
                    raise
                continue
            lines = text.splitlines()
            for line_index, line in enumerate(lines):
                if query.lower() not in line.lower():
                    continue
                if len(matches) >= max_matches:
                    truncated = True
                    break
                matches.append(
                    {
                        "path": self._display_path(file_path),
                        "line_number": line_index + 1,
                        "line": _truncate_line(line),
                        "context": _context_payload(lines, line_index, context_lines),
                    }
                )
            if truncated:
                break

        return ToolResult(
            name=self.spec.name,
            output={
                "query": query,
                "path": resolved.display,
                "matches": matches,
                "truncated": truncated,
            },
            metadata={"match_count": len(matches), "truncated": truncated},
        )

    def _iter_search_files(self, path: Path, pattern: str | None) -> list[Path]:
        if path.is_symlink():
            return []
        if path.is_file():
            if pattern and not _matches_glob(self._display_path(path), path.name, pattern):
                return []
            return [path]
        if not path.is_dir():
            raise FileToolError("path must be a file or directory")
        files: list[Path] = []
        self._collect_search_files(path, pattern, files)
        return files

    def _collect_search_files(
        self,
        directory: Path,
        pattern: str | None,
        files: list[Path],
    ) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name in DEFAULT_EXCLUDED_DIRS and child.is_dir():
                continue
            try:
                resolved = child.resolve(strict=True)
            except OSError:
                continue
            if not _is_inside_allowed(resolved, self.allowed_roots):
                continue
            if child.is_dir() and not child.is_symlink():
                self._collect_search_files(child, pattern, files)
                continue
            if child.is_file():
                display = self._display_path(child)
                if pattern and not _matches_glob(display, child.name, pattern):
                    continue
                files.append(child)


class FilePatchTool(_FileToolBase):
    """Apply structured text edits inside configured write roots."""

    @property
    def spec(self) -> ToolSpec:
        """Return the file patching spec derived from current config."""

        return ToolSpec(
            name=FILE_PATCH_TOOL_NAME,
            description=(
                "Apply small structured edits to a UTF-8 text file inside configured write "
                "roots. Requires expected_sha256 for existing files to avoid overwriting "
                "concurrent changes."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Text file path relative to a configured write root.",
                    },
                    "expected_sha256": {
                        "type": "string",
                        "description": (
                            "Current file SHA-256 for existing files. Must be empty or "
                            "omitted when creating a new file."
                        ),
                    },
                    "create_if_missing": {
                        "type": "boolean",
                        "description": "Whether to create the file when it does not exist.",
                    },
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_line": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "description": "1-based first original line to replace.",
                                },
                                "end_line": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "description": (
                                        "1-based inclusive original line to replace. Use "
                                        "start_line = end_line + 1 to insert."
                                    ),
                                },
                                "replacement": {
                                    "type": "string",
                                    "description": "Replacement text for this edit.",
                                },
                            },
                            "required": ["start_line", "end_line", "replacement"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["path", "edits"],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=False,
            destructive=True,
            concurrency_safe=False,
            requires_user_interaction=False,
        )

    def check_available(self) -> ToolAvailability:
        """Return whether local file patching is enabled."""

        availability = super().check_available()
        if not availability.available:
            return availability
        if not self.config.patch_enabled:
            return ToolAvailability.unavailable("tools.files.patch_enabled is false")
        if not self.write_roots:
            return ToolAvailability.unavailable("tools.files.write_roots is empty")
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        """Apply validated structured edits and return compact patch metadata."""

        del context
        resolved = self._resolve_patch_path(
            arguments.get("path"),
            create_if_missing=_optional_bool(
                arguments.get("create_if_missing", False),
                "create_if_missing",
            ),
        )
        current_data: bytes | None
        before_sha256: str | None
        if resolved.path.exists():
            if resolved.path.is_symlink():
                raise FileToolError("symlink files are not patched")
            if not resolved.path.is_file():
                raise FileToolError("path must be a file")
            current_data = _read_file_bytes(
                resolved.path,
                max_file_bytes=self.config.max_file_bytes,
            )
            before_sha256 = hashlib.sha256(current_data).hexdigest()
            expected_sha256 = _required_sha256(arguments.get("expected_sha256"))
            if expected_sha256 != before_sha256:
                raise FileToolError("expected_sha256 does not match current file content")
            before_text = _decode_text(current_data)
        else:
            new_file_expected_sha256 = _optional_text(arguments.get("expected_sha256"))
            if new_file_expected_sha256:
                raise FileToolError("expected_sha256 must be empty when creating a new file")
            current_data = None
            before_sha256 = None
            before_text = ""

        original_lines = before_text.splitlines(keepends=True)
        edits = _parse_patch_edits(arguments.get("edits"), line_count=len(original_lines))
        new_lines = list(original_lines)
        for edit in reversed(edits):
            start_index = edit.start_line - 1
            end_index = edit.end_line
            new_lines[start_index:end_index] = edit.replacement.splitlines(keepends=True)
        after_text = "".join(new_lines)
        after_data = after_text.encode("utf-8")
        _decode_text(after_data)
        if len(after_data) > self.config.max_file_bytes:
            raise FileToolError("patched file is too large to write")

        resolved.path.write_bytes(after_data)
        after_sha256 = hashlib.sha256(after_data).hexdigest()
        diff = _bounded_unified_diff(
            before_text,
            after_text,
            path=resolved.display,
            max_chars=self.config.max_output_chars,
        )
        output: dict[str, JSONValue] = {
            "path": resolved.display,
            "before_sha256": before_sha256,
            "after_sha256": after_sha256,
            "bytes_written": len(after_data),
            "line_count": len(after_text.splitlines()),
            "applied_edits": len(edits),
            "diff": diff,
        }
        return ToolResult(
            name=self.spec.name,
            output=output,
            metadata={
                "path": resolved.display,
                "created": current_data is None,
                "bytes_written": len(after_data),
                "applied_edits": len(edits),
            },
        )

    def _resolve_patch_path(
        self,
        value: Any,
        *,
        create_if_missing: bool,
    ) -> _ResolvedPath:
        if not isinstance(value, str) or not value:
            raise FileToolError("path must be a non-empty string")
        if "\x00" in value:
            raise FileToolError("path must not contain NUL characters")
        if not self.write_roots:
            raise FileToolError("tools.files.write_roots is empty")
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.write_roots[0] / candidate
        if candidate.is_symlink():
            raise FileToolError("symlink files are not patched")
        self._reject_symlink_ancestors(candidate)
        if candidate.exists():
            resolved = candidate.resolve(strict=True)
            if not _is_inside_allowed(resolved, self.write_roots):
                raise FileToolError("path is outside tools.files.write_roots")
            return _ResolvedPath(path=resolved, display=self._display_write_path(resolved))

        if not create_if_missing:
            raise FileToolError("create_if_missing must be true to create a new file")
        parent = candidate.parent
        if not parent.exists():
            raise FileToolError("parent directory must exist")
        if parent.is_symlink():
            raise FileToolError("symlink directories are not patched")
        parent_resolved = parent.resolve(strict=True)
        if not parent_resolved.is_dir():
            raise FileToolError("parent path must be a directory")
        resolved = candidate.resolve(strict=False)
        if not _is_inside_allowed(parent_resolved, self.write_roots) or not _is_inside_allowed(
            resolved,
            self.write_roots,
        ):
            raise FileToolError("path is outside tools.files.write_roots")
        return _ResolvedPath(path=resolved, display=self._display_write_path(resolved))

    def _reject_symlink_ancestors(self, candidate: Path) -> None:
        for ancestor in reversed(candidate.parents):
            if ancestor.is_symlink():
                raise FileToolError("symlink ancestors are not patched")
            if not ancestor.exists():
                return

    def _display_write_path(self, path: Path) -> str:
        for root in self.write_roots:
            if path == root:
                return "."
            if path.is_relative_to(root):
                return path.relative_to(root).as_posix()
        return self._display_path(path)


@dataclass(frozen=True)
class _PatchEdit:
    start_line: int
    end_line: int
    replacement: str


def _normalized_roots(
    roots: tuple[Path, ...],
    *,
    required: bool = True,
    label: str = "tools.files.allowed_roots",
) -> tuple[Path, ...]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        path = root.expanduser().resolve()
        if path not in seen:
            resolved.append(path)
            seen.add(path)
    if required and not resolved:
        raise FileToolError(f"{label} must not be empty")
    return tuple(resolved)


def _is_inside_allowed(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in allowed_roots)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_bool(value: Any, field_name: str) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise FileToolError(f"{field_name} must be a boolean")
    return value


def _bounded_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
    field_name: str,
) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < minimum:
        raise FileToolError(f"{field_name} must be at least {minimum}")
    return min(value, maximum)


def _optional_positive_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < 1:
        raise FileToolError(f"{field_name} must be at least 1")
    return value


def _required_sha256(value: Any) -> str:
    text = _optional_text(value)
    if text is None:
        raise FileToolError("expected_sha256 is required for existing files")
    normalized = text.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise FileToolError("expected_sha256 must be a SHA-256 hex digest")
    return normalized


def _parse_patch_edits(value: Any, *, line_count: int) -> list[_PatchEdit]:
    if not isinstance(value, list) or not value:
        raise FileToolError("edits must be a non-empty array")
    edits: list[_PatchEdit] = []
    previous_start = 0
    previous_covered_end = 0
    for raw_edit in value:
        if not isinstance(raw_edit, dict):
            raise FileToolError("each edit must be an object")
        start_line = _required_line_int(raw_edit.get("start_line"), "start_line", minimum=1)
        end_line = _required_line_int(raw_edit.get("end_line"), "end_line", minimum=0)
        replacement = raw_edit.get("replacement")
        if not isinstance(replacement, str):
            raise FileToolError("replacement must be a string")
        if "\x00" in replacement:
            raise FileToolError("binary files are not allowed")
        if end_line < start_line - 1:
            raise FileToolError("end_line must be at least start_line - 1")
        is_insertion = start_line == end_line + 1
        if is_insertion:
            if start_line > line_count + 1:
                raise FileToolError("insertion line must be at most line_count + 1")
            covered_end = end_line
        else:
            if end_line > line_count:
                raise FileToolError("end_line must be at most line_count")
            covered_end = end_line
        if start_line < previous_start:
            raise FileToolError("edits must be in original file order")
        if start_line == previous_start or start_line <= previous_covered_end:
            raise FileToolError("edits must not overlap")
        edits.append(
            _PatchEdit(
                start_line=start_line,
                end_line=end_line,
                replacement=replacement,
            )
        )
        previous_start = start_line
        previous_covered_end = covered_end
    return edits


def _required_line_int(value: Any, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError(f"{field_name} must be an integer")
    if value < minimum:
        raise FileToolError(f"{field_name} must be at least {minimum}")
    return value


def _bounded_unified_diff(
    before_text: str,
    after_text: str,
    *,
    path: str,
    max_chars: int,
) -> str:
    diff = "".join(
        difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=f"{path}\tbefore",
            tofile=f"{path}\tafter",
            n=3,
        )
    )
    limit = max(200, max_chars - 1000)
    if len(diff) <= limit:
        return diff
    return diff[:limit] + "\n... diff truncated ...\n"


def _read_file_bytes(path: Path, *, max_file_bytes: int) -> bytes:
    size = path.stat().st_size
    if size > max_file_bytes:
        raise FileToolError("file is too large to read")
    return path.read_bytes()


def _decode_text(data: bytes) -> str:
    if b"\x00" in data:
        raise FileToolError("binary files are not allowed")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FileToolError("binary files are not allowed") from exc


def _matches_glob(display_path: str, name: str, pattern: str) -> bool:
    return fnmatch.fnmatch(display_path, pattern) or fnmatch.fnmatch(name, pattern)


def _entry_payload(path: Path, display_path: str) -> dict[str, JSONValue]:
    item_type = _path_type(path)
    stat = path.lstat() if item_type == "symlink" else path.stat()
    return {
        "path": display_path,
        "type": item_type,
        "size": stat.st_size if item_type == "file" else None,
        "mtime": stat.st_mtime,
        "truncated": False,
    }


def _truncate_line(line: str) -> str:
    if len(line) <= MAX_SEARCH_LINE_CHARS:
        return line
    return line[:MAX_SEARCH_LINE_CHARS]


def _returned_end_line(
    content: str,
    *,
    start_line: int,
    has_lines: bool,
    empty_end_line: int,
) -> int:
    if not has_lines:
        return 0
    if not content:
        return empty_end_line
    newline_count = content.count("\n")
    if content.endswith("\n"):
        return start_line + newline_count - 1
    return start_line + newline_count


def _path_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _context_payload(
    lines: list[str],
    match_index: int,
    context_lines: int,
) -> dict[str, JSONValue]:
    if context_lines == 0:
        return {"before": [], "after": []}
    before_start = max(0, match_index - context_lines)
    after_end = min(len(lines), match_index + context_lines + 1)
    before: list[JSONValue] = [
        {"line_number": index + 1, "line": _truncate_line(lines[index])}
        for index in range(before_start, match_index)
    ]
    after: list[JSONValue] = [
        {"line_number": index + 1, "line": _truncate_line(lines[index])}
        for index in range(match_index + 1, after_end)
    ]
    return {"before": before, "after": after}
