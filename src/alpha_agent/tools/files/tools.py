"""Concrete local file tools."""

from __future__ import annotations

import fnmatch
from collections.abc import Callable
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
from alpha_agent.tools.files.config import (
    create_parent_dirs_enabled,
    max_glob_results,
    max_read_lines,
    max_search_results,
)
from alpha_agent.tools.files.errors import FileToolError
from alpha_agent.tools.files.io import (
    atomic_write_text,
    bounded_unified_diff,
    path_locks,
    read_text_file,
    sha256_bytes,
)
from alpha_agent.tools.files.patching import (
    PlannedWrite,
    apply_range_edits,
    apply_replace,
    parse_range_edits,
    plan_patch_text,
    resolve_patch_text_paths,
)
from alpha_agent.tools.files.paths import (
    DEFAULT_EXCLUDED_DIRS,
    ResolvedPath,
    display_path,
    normalized_roots,
    path_type,
    resolve_read_path,
    resolve_write_path,
)
from alpha_agent.tools.files.searching import rg_files, rg_search
from alpha_agent.tools.files.state import invalidate_read_ledger, read_ledger
from alpha_agent.tools.files.validation import (
    bounded_int,
    optional_bool,
    optional_non_negative_int,
    optional_sha256,
    required_sha256,
    required_text,
    syntax_validation,
)

FILE_GLOB_TOOL_NAME = "file_glob"
FILE_PATCH_TOOL_NAME = "file_patch"
FILE_READ_TOOL_NAME = "file_read"
FILE_SEARCH_TOOL_NAME = "file_search"
FILE_WRITE_TOOL_NAME = "file_write"

MAX_CONTEXT_LINES = 5


class _FileToolBase:
    def __init__(self, config: FileToolConfig | None = None):
        self.config = config or FileToolConfig()
        self.allowed_roots = normalized_roots(self.config.allowed_roots)
        self.write_roots = normalized_roots(
            self.config.write_roots,
            required=False,
            label="tools.files.write_roots",
        )

    def check_available(self) -> ToolAvailability:
        if not self.config.enabled:
            return ToolAvailability.unavailable("file tools are disabled for this session")
        if not self.allowed_roots:
            return ToolAvailability.unavailable(
                "file tools have no readable workspace roots configured"
            )
        return ToolAvailability()

    def _display(self, path: Path) -> str:
        return display_path(path.resolve(strict=False), self.allowed_roots)

    def _display_write(self, path: Path) -> str:
        return display_path(path.resolve(strict=False), self.write_roots)


class FileGlobTool(_FileToolBase):
    """Find files by path/name pattern."""

    @property
    def spec(self) -> ToolSpec:
        limit = max_glob_results(self.config)
        return ToolSpec(
            name=FILE_GLOB_TOOL_NAME,
            description="Find files by path/name pattern under the readable file workspace.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern, default *."},
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file or directory to search; defaults to '.'."
                        ),
                    },
                    "max_depth": {"type": "integer", "minimum": 1},
                    "limit": {"type": "integer", "minimum": 1, "maximum": limit},
                    "offset": {"type": "integer", "minimum": 0},
                    "sort": {"type": "string", "enum": ["mtime_desc", "path_asc"]},
                    "include_dirs": {"type": "boolean"},
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
        del context
        resolved = resolve_read_path(arguments.get("path"), roots=self.allowed_roots, default=".")
        if not resolved.path.is_dir() and not resolved.path.is_file():
            raise FileToolError("path must be a file or directory")
        pattern = arguments.get("pattern", "*")
        if not isinstance(pattern, str) or not pattern:
            raise FileToolError("pattern must be a non-empty string")
        max_depth = optional_non_negative_int(arguments.get("max_depth"), "max_depth")
        if max_depth == 0:
            raise FileToolError("max_depth must be at least 1")
        limit = bounded_int(
            arguments.get("limit"),
            default=max_glob_results(self.config),
            minimum=1,
            maximum=max_glob_results(self.config),
            field_name="limit",
        )
        offset = optional_non_negative_int(arguments.get("offset"), "offset") or 0
        sort = arguments.get("sort", "mtime_desc")
        if sort not in {"mtime_desc", "path_asc"}:
            raise FileToolError("sort must be one of: mtime_desc, path_asc")
        include_dirs = optional_bool(arguments.get("include_dirs"), "include_dirs")
        if max_depth == 1 or resolved.path.is_file():
            all_entries = self._local_browse(
                resolved.path, pattern=pattern, include_dirs=include_dirs
            )
            total_count: int | None = len(all_entries)
        else:
            try:
                all_entries = [
                    path
                    for path in rg_files(resolved.path, pattern=pattern, max_depth=max_depth)
                    if path.exists()
                ]
            except FileToolError as exc:
                return ToolResult(
                    name=self.spec.name,
                    output={
                        "path": resolved.display,
                        "pattern": pattern,
                        "files": [],
                        "limit": limit,
                        "offset": offset,
                        "next_offset": None,
                        "truncated": False,
                        "error": str(exc),
                    },
                    metadata={"unavailable": True},
                )
            if include_dirs:
                all_entries.extend(
                    self._recursive_dirs(resolved.path, max_depth=max_depth, pattern=pattern)
                )
            total_count = len(all_entries)
        all_entries = _sorted_entries(all_entries, display=self._display, sort=sort)
        selected = all_entries[offset : offset + limit]
        next_offset = offset + len(selected) if offset + len(selected) < len(all_entries) else None
        return ToolResult(
            name=self.spec.name,
            output={
                "path": resolved.display,
                "pattern": pattern,
                "files": [_entry_payload(path, self._display(path)) for path in selected],
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "next_offset": next_offset,
                "truncated": next_offset is not None,
            },
            metadata={"entry_count": len(selected), "truncated": next_offset is not None},
        )

    def _local_browse(self, path: Path, *, pattern: str, include_dirs: bool) -> list[Path]:
        if path.is_file():
            return [path] if _matches_glob(self._display(path), path.name, pattern) else []
        entries: list[Path] = []
        for child in path.iterdir():
            if child.name in DEFAULT_EXCLUDED_DIRS and child.is_dir():
                continue
            if child.is_dir() and not include_dirs:
                continue
            if not child.is_dir() and not child.is_file() and not child.is_symlink():
                continue
            display = self._display(child)
            if _matches_glob(display, child.name, pattern):
                entries.append(child)
        return entries

    def _recursive_dirs(self, root: Path, *, max_depth: int | None, pattern: str) -> list[Path]:
        dirs: list[Path] = []
        root_depth = len(root.parts)
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name, reverse=True)
            except OSError:
                continue
            for child in children:
                if not child.is_dir() or child.name in DEFAULT_EXCLUDED_DIRS:
                    continue
                depth = len(child.parts) - root_depth
                if max_depth is not None and depth > max_depth:
                    continue
                stack.append(child)
                if _matches_glob(self._display(child), child.name, pattern):
                    dirs.append(child)
        return dirs


class FileReadTool(_FileToolBase):
    """Read a bounded view of a text file."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=FILE_READ_TOOL_NAME,
            description="Read a bounded, line-addressable UTF-8 text file view.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file path inside the readable workspace."
                        ),
                    },
                    "offset": {"type": "integer", "minimum": 1},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": max_read_lines(self.config),
                    },
                    "max_chars": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self.config.max_read_chars,
                    },
                    "format": {"type": "string", "enum": ["line_numbered", "plain"]},
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
        resolved = resolve_read_path(arguments.get("path"), roots=self.allowed_roots)
        data, content = read_text_file(resolved.path, max_file_bytes=self.config.max_file_bytes)
        stat = resolved.path.stat()
        file_hash = sha256_bytes(data)
        offset = bounded_int(
            arguments.get("offset"), default=1, minimum=1, maximum=10**9, field_name="offset"
        )
        limit = bounded_int(
            arguments.get("limit"),
            default=max_read_lines(self.config),
            minimum=1,
            maximum=max_read_lines(self.config),
            field_name="limit",
        )
        max_chars = bounded_int(
            arguments.get("max_chars"),
            default=self.config.max_read_chars,
            minimum=1,
            maximum=self.config.max_read_chars,
            field_name="max_chars",
        )
        fmt = arguments.get("format", "line_numbered")
        if fmt not in {"line_numbered", "plain"}:
            raise FileToolError("format must be one of: line_numbered, plain")
        ledger_key = (resolved.path, offset, limit, max_chars, fmt, stat.st_size, stat.st_mtime_ns)
        ledger = read_ledger(context)
        if ledger is not None and ledger_key in ledger:
            cached_payload = dict(ledger[ledger_key])
            cached_payload["deduplicated"] = True
            cached_payload["content"] = "<same file view already read this turn>"
            return ToolResult(
                name=self.spec.name,
                output=cached_payload,
                metadata={"deduplicated": True},
            )

        lines = content.splitlines(keepends=True)
        total_lines = len(lines)
        start_index = min(offset - 1, total_lines)
        selected_lines = lines[start_index : start_index + limit]
        line_truncated = start_index + len(selected_lines) < total_lines
        if fmt == "line_numbered":
            selected = "".join(
                f"{line_no:6d}\t{line}"
                for line_no, line in enumerate(selected_lines, start=start_index + 1)
            )
        else:
            selected = "".join(selected_lines)
        char_truncated = len(selected) > max_chars
        if char_truncated:
            selected = selected[:max_chars]
        returned_lines = len(selected_lines)
        next_offset = offset + returned_lines if line_truncated else None
        payload: dict[str, JSONValue] = {
            "path": resolved.display,
            "content": selected,
            "offset": offset,
            "limit": limit,
            "returned_lines": returned_lines,
            "total_lines": total_lines,
            "next_offset": next_offset,
            "sha256": file_hash,
            "size": len(data),
            "truncated": line_truncated or char_truncated,
            "format": fmt,
        }
        if ledger is not None:
            ledger[ledger_key] = dict(payload)
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
    """Search file contents with ripgrep."""

    @property
    def spec(self) -> ToolSpec:
        limit = max_search_results(self.config)
        return ToolSpec(
            name=FILE_SEARCH_TOOL_NAME,
            description="Search file contents using ripgrep.",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "mode": {"type": "string", "enum": ["regex", "literal"]},
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative file or directory to search; defaults to '.'."
                        ),
                    },
                    "glob": {"type": "string"},
                    "type": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                    "case_sensitive": {"type": "boolean"},
                    "context": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT_LINES},
                    "before_context": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_CONTEXT_LINES,
                    },
                    "after_context": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": MAX_CONTEXT_LINES,
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": limit},
                    "offset": {"type": "integer", "minimum": 0},
                    "multiline": {"type": "boolean"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=True,
            destructive=False,
            concurrency_safe=True,
        )

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        del context
        pattern = required_text(arguments.get("pattern"), "pattern")
        resolved = resolve_read_path(arguments.get("path"), roots=self.allowed_roots, default=".")
        output_mode = arguments.get("output_mode", "files_with_matches")
        if output_mode not in {"content", "files_with_matches", "count"}:
            raise FileToolError("output_mode must be one of: content, files_with_matches, count")
        mode = arguments.get("mode", "regex")
        if mode not in {"regex", "literal"}:
            raise FileToolError("mode must be one of: regex, literal")
        context_lines = bounded_int(
            arguments.get("context"),
            default=0,
            minimum=0,
            maximum=MAX_CONTEXT_LINES,
            field_name="context",
        )
        before_context = bounded_int(
            arguments.get("before_context"),
            default=context_lines,
            minimum=0,
            maximum=MAX_CONTEXT_LINES,
            field_name="before_context",
        )
        after_context = bounded_int(
            arguments.get("after_context"),
            default=context_lines,
            minimum=0,
            maximum=MAX_CONTEXT_LINES,
            field_name="after_context",
        )
        limit = bounded_int(
            arguments.get("limit"),
            default=max_search_results(self.config),
            minimum=1,
            maximum=max_search_results(self.config),
            field_name="limit",
        )
        offset = optional_non_negative_int(arguments.get("offset"), "offset") or 0
        payload = rg_search(
            resolved.path,
            pattern=pattern,
            mode=mode,
            glob=arguments.get("glob") if isinstance(arguments.get("glob"), str) else None,
            file_type=arguments.get("type") if isinstance(arguments.get("type"), str) else None,
            output_mode=output_mode,
            case_sensitive=optional_bool(arguments.get("case_sensitive"), "case_sensitive"),
            before_context=before_context,
            after_context=after_context,
            limit=limit,
            offset=offset,
            multiline=optional_bool(arguments.get("multiline"), "multiline"),
            max_file_bytes=self.config.max_file_bytes,
            display=self._display,
        )
        payload.update({"path": resolved.display, "pattern": pattern, "output_mode": output_mode})
        return ToolResult(
            name=self.spec.name,
            output=payload,
            metadata={
                "truncated": bool(payload.get("truncated")),
                "unavailable": "error" in payload,
            },
        )


class FilePatchTool(_FileToolBase):
    """Apply targeted edits inside writable file workspaces."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=FILE_PATCH_TOOL_NAME,
            description="Apply range, exact replacement, or strict patch-text edits.",
            parameters={
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["range", "replace", "patch_text"]},
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative target path inside a writable workspace."
                        ),
                    },
                    "expected_sha256": {"type": "string"},
                    "create_if_missing": {"type": "boolean"},
                    "edits": {"type": "array"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                    "patch": {"type": "string"},
                },
                "required": ["mode"],
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
        availability = super().check_available()
        if not availability.available:
            return availability
        if not self.config.patch_enabled:
            return ToolAvailability.unavailable("file writing is disabled for this session")
        if not self.write_roots:
            return ToolAvailability.unavailable(
                "file writing has no writable workspace roots configured"
            )
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        mode = arguments.get("mode")
        if mode == "range":
            resolved = self._resolve_patch_lock_target(arguments)
            with path_locks([resolved.path]):
                planned = [self._plan_range(arguments)]
                return _write_planned_locked(
                    self.spec.name,
                    planned,
                    context=context,
                    max_file_bytes=self.config.max_file_bytes,
                    max_output_chars=self.config.max_output_chars,
                )
        elif mode == "replace":
            resolved = self._resolve_patch_lock_target(arguments)
            with path_locks([resolved.path]):
                planned = [self._plan_replace(arguments)]
                return _write_planned_locked(
                    self.spec.name,
                    planned,
                    context=context,
                    max_file_bytes=self.config.max_file_bytes,
                    max_output_chars=self.config.max_output_chars,
                )
        elif mode == "patch_text":
            patch = required_text(arguments.get("patch"), "patch")
            paths = resolve_patch_text_paths(patch, write_roots=self.write_roots)
            with path_locks(paths):
                planned = plan_patch_text(
                    patch,
                    write_roots=self.write_roots,
                    max_file_bytes=self.config.max_file_bytes,
                )
                return _write_planned_locked(
                    self.spec.name,
                    planned,
                    context=context,
                    max_file_bytes=self.config.max_file_bytes,
                    max_output_chars=self.config.max_output_chars,
                )
        else:
            raise FileToolError("mode must be one of: range, replace, patch_text")

    def _resolve_patch_lock_target(self, arguments: dict[str, Any]) -> ResolvedPath:
        return resolve_write_path(
            arguments.get("path"),
            roots=self.write_roots,
            create_if_missing=optional_bool(
                arguments.get("create_if_missing"),
                "create_if_missing",
            ),
        )

    def _plan_range(self, arguments: dict[str, Any]) -> PlannedWrite:
        resolved, before_text, before_sha256, created = self._read_write_target(arguments)
        if not created and required_sha256(arguments.get("expected_sha256")) != before_sha256:
            raise FileToolError("expected_sha256 does not match current file content")
        edits = parse_range_edits(
            arguments.get("edits"), line_count=len(before_text.splitlines(keepends=True))
        )
        after_text = apply_range_edits(before_text, edits)
        return PlannedWrite(resolved, before_text, after_text, before_sha256, created, len(edits))

    def _plan_replace(self, arguments: dict[str, Any]) -> PlannedWrite:
        resolved, before_text, before_sha256, created = self._read_write_target(arguments)
        if created:
            raise FileToolError("replace mode requires an existing file")
        if required_sha256(arguments.get("expected_sha256")) != before_sha256:
            raise FileToolError("expected_sha256 does not match current file content")
        old_string = required_text(arguments.get("old_string"), "old_string")
        new_string = arguments.get("new_string")
        if not isinstance(new_string, str):
            raise FileToolError("new_string must be a string")
        after_text, count = apply_replace(
            before_text,
            old_string,
            new_string,
            replace_all=optional_bool(arguments.get("replace_all"), "replace_all"),
        )
        return PlannedWrite(resolved, before_text, after_text, before_sha256, False, count)

    def _read_write_target(
        self, arguments: dict[str, Any]
    ) -> tuple[ResolvedPath, str, str | None, bool]:
        create = optional_bool(arguments.get("create_if_missing"), "create_if_missing")
        resolved = resolve_write_path(
            arguments.get("path"),
            roots=self.write_roots,
            create_if_missing=create,
        )
        if resolved.path.exists():
            data, before_text = read_text_file(
                resolved.path, max_file_bytes=self.config.max_file_bytes
            )
            return resolved, before_text, sha256_bytes(data), False
        if optional_sha256(arguments.get("expected_sha256")):
            raise FileToolError("expected_sha256 must be empty when creating a new file")
        return resolved, "", None, True


class FileWriteTool(_FileToolBase):
    """Create or intentionally replace a whole file."""

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=FILE_WRITE_TOOL_NAME,
            description="Create a new file or overwrite a whole file with hash guard.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Workspace-relative target path inside a writable workspace."
                        ),
                    },
                    "content": {"type": "string"},
                    "expected_sha256": {"type": "string"},
                    "create_if_missing": {"type": "boolean"},
                    "overwrite": {"type": "boolean"},
                    "create_parent_dirs": {"type": "boolean"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
            max_result_size_chars=self.config.max_output_chars,
            toolset="file",
            read_only=False,
            destructive=True,
            concurrency_safe=False,
        )

    def check_available(self) -> ToolAvailability:
        availability = super().check_available()
        if not availability.available:
            return availability
        if not self.config.patch_enabled:
            return ToolAvailability.unavailable("file writing is disabled for this session")
        if not self.write_roots:
            return ToolAvailability.unavailable(
                "file writing has no writable workspace roots configured"
            )
        return ToolAvailability()

    def run(self, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        content = arguments.get("content")
        if not isinstance(content, str):
            raise FileToolError("content must be a string")
        if "\x00" in content:
            raise FileToolError("content must not contain NUL characters")
        create_parent_dirs = optional_bool(
            arguments.get("create_parent_dirs"), "create_parent_dirs"
        )
        if create_parent_dirs and not create_parent_dirs_enabled(self.config):
            raise FileToolError(
                "create_parent_dirs cannot be used in this session; "
                "choose an existing parent directory or retry with create_parent_dirs false",
                code="file_parent_creation_disabled",
                details={
                    "argument": "create_parent_dirs",
                    "retry": (
                        "Write to a path whose parent directory already exists, "
                        "or call the tool with create_parent_dirs false."
                    ),
                },
            )
        resolved = self._resolve_write_lock_target(arguments, create_parent_dirs=create_parent_dirs)
        with path_locks([resolved.path]):
            planned = [
                self._plan_write(arguments, content=content, create_parent_dirs=create_parent_dirs)
            ]
            return _write_planned_locked(
                self.spec.name,
                planned,
                context=context,
                max_file_bytes=self.config.max_file_bytes,
                max_output_chars=self.config.max_output_chars,
            )

    def _resolve_write_lock_target(
        self, arguments: dict[str, Any], *, create_parent_dirs: bool
    ) -> ResolvedPath:
        return resolve_write_path(
            arguments.get("path"),
            roots=self.write_roots,
            create_if_missing=optional_bool(
                arguments.get("create_if_missing"), "create_if_missing", default=True
            ),
            create_parent_dirs=create_parent_dirs,
        )

    def _plan_write(
        self,
        arguments: dict[str, Any],
        *,
        content: str,
        create_parent_dirs: bool,
    ) -> PlannedWrite:
        resolved = self._resolve_write_lock_target(
            arguments,
            create_parent_dirs=create_parent_dirs,
        )
        if resolved.path.exists():
            if not optional_bool(arguments.get("overwrite"), "overwrite"):
                raise FileToolError("overwrite must be true to replace an existing file")
            data, before_text = read_text_file(
                resolved.path, max_file_bytes=self.config.max_file_bytes
            )
            before_sha256 = sha256_bytes(data)
            if required_sha256(arguments.get("expected_sha256")) != before_sha256:
                raise FileToolError("expected_sha256 does not match current file content")
            created = False
        else:
            if optional_sha256(arguments.get("expected_sha256")):
                raise FileToolError("expected_sha256 must be empty when creating a new file")
            before_text = ""
            before_sha256 = None
            created = True
        return PlannedWrite(resolved, before_text, content, before_sha256, created, 1)


def _write_planned_locked(
    tool_name: str,
    planned: list[PlannedWrite],
    *,
    context: ToolExecutionContext,
    max_file_bytes: int,
    max_output_chars: int,
) -> ToolResult:
    paths = [item.resolved.path for item in planned]
    if len(paths) != len(set(paths)):
        raise FileToolError("duplicate write paths are not allowed")
    preflight_validations = _preflight_planned_writes(planned, max_file_bytes=max_file_bytes)
    may_have_written_paths: list[Path] = []
    outputs: list[dict[str, JSONValue]] = []
    try:
        for item, validation in zip(planned, preflight_validations, strict=True):
            may_have_written_paths.append(item.resolved.path)
            bytes_written, after_sha256 = atomic_write_text(
                item.resolved.path,
                item.after_text,
                max_file_bytes=max_file_bytes,
            )
            outputs.append(
                {
                    "path": item.resolved.display,
                    "before_sha256": item.before_sha256,
                    "after_sha256": after_sha256,
                    "bytes_written": bytes_written,
                    "created": item.created,
                    "applied_edits": item.applied_edits,
                    "diff": bounded_unified_diff(
                        item.before_text,
                        item.after_text,
                        path=item.resolved.display,
                        max_chars=max_output_chars,
                    ),
                    "validation": validation,
                    "warnings": [],
                }
            )
    finally:
        if may_have_written_paths:
            invalidate_read_ledger(context, may_have_written_paths)
    files_created: list[JSONValue] = [str(row["path"]) for row in outputs if row["created"]]
    files_modified: list[JSONValue] = [
        str(row["path"]) for row in outputs if not row["created"]
    ]
    first = outputs[0] if len(outputs) == 1 else {}
    bytes_written_total = sum(_json_int(row["bytes_written"]) for row in outputs)
    applied_edits_total = sum(_json_int(row["applied_edits"]) for row in outputs)
    validation_items: list[JSONValue] = []
    for row in outputs:
        row_validation = row["validation"]
        if isinstance(row_validation, list):
            validation_items.extend(row_validation)
    output: dict[str, JSONValue] = {
        "files_modified": files_modified,
        "files_created": files_created,
        "bytes_written": bytes_written_total,
        "applied_edits": applied_edits_total,
        "diff": "\n".join(str(row["diff"]) for row in outputs),
        "validation": validation_items,
        "warnings": [],
    }
    if len(outputs) == 1:
        output.update(
            {
                "path": first["path"],
                "before_sha256": first["before_sha256"],
                "after_sha256": first["after_sha256"],
            }
        )
    return ToolResult(
        name=tool_name,
        output=output,
        metadata={
            "files_modified": len(files_modified),
            "files_created": len(files_created),
            "bytes_written": bytes_written_total,
        },
    )


def _preflight_planned_writes(
    planned: list[PlannedWrite], *, max_file_bytes: int
) -> list[list[JSONValue]]:
    validations: list[list[JSONValue]] = []
    for item in planned:
        data = item.after_text.encode("utf-8")
        if len(data) > max_file_bytes:
            raise FileToolError("file is too large to write")
        validations.append(syntax_validation(item.resolved.path, item.after_text))
    return validations


def _matches_glob(display: str, name: str, pattern: str) -> bool:
    return fnmatch.fnmatch(display, pattern) or fnmatch.fnmatch(name, pattern)


def _entry_payload(path: Path, display: str) -> dict[str, JSONValue]:
    item_type = path_type(path)
    stat = path.lstat() if item_type == "symlink" else path.stat()
    return {
        "path": display,
        "type": item_type,
        "size": stat.st_size if item_type == "file" else None,
        "mtime": stat.st_mtime,
    }


def _json_int(value: JSONValue) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolError("internal write counter must be an integer")
    return value


def _sorted_entries(
    paths: list[Path],
    *,
    display: Callable[[Path], str],
    sort: str,
) -> list[Path]:
    if sort == "path_asc":
        return sorted(paths, key=display)
    return sorted(paths, key=lambda item: (-item.stat().st_mtime, display(item)))
