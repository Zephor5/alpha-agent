"""Path policy helpers for local file tools."""

from __future__ import annotations

import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_agent.tools.files.errors import FileToolError
from alpha_agent.tools.files.validation import required_text

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


@dataclass(frozen=True)
class ResolvedPath:
    path: Path
    display: str


READ_BOUNDARY_MESSAGE = (
    "path is outside the readable file workspace; use a relative path inside the workspace"
)
WRITE_BOUNDARY_MESSAGE = (
    "path is outside the writable file workspace; use a relative path inside a writable workspace"
)
READ_BOUNDARY_DETAILS = {
    "argument": "path",
    "retry": (
        "Use a workspace-relative path, or inspect available files with file_glob using path '.'."
    ),
}
WRITE_BOUNDARY_DETAILS = {
    "argument": "path",
    "retry": "Use a relative target path under a writable workspace.",
}


def normalized_roots(
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
        scope = _root_scope(label)
        raise FileToolError(
            f"{scope} file workspace roots must not be empty",
            code=f"file_{scope}_workspace_roots_empty",
            details={"scope": scope},
        )
    return tuple(resolved)


def is_inside_allowed(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def display_path(path: Path, roots: tuple[Path, ...]) -> str:
    for root in roots:
        if path == root:
            return "."
        if path.is_relative_to(root):
            return path.relative_to(root).as_posix()
    return path.name


def resolve_read_path(
    value: Any,
    *,
    roots: tuple[Path, ...],
    default: str | None = None,
    must_exist: bool = True,
) -> ResolvedPath:
    raw = default if value in (None, "") else value
    text = required_text(raw, "path")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    try:
        resolved = candidate.resolve(strict=must_exist)
    except FileNotFoundError as exc:
        suggestions = _missing_suggestions(candidate, roots)
        suffix = f"; suggestions: {', '.join(suggestions)}" if suggestions else ""
        raise FileToolError(f"path does not exist{suffix}") from exc
    if not is_inside_allowed(resolved, roots):
        raise FileToolError(
            READ_BOUNDARY_MESSAGE,
            code="file_path_outside_readable_workspace",
            details=READ_BOUNDARY_DETAILS,
        )
    return ResolvedPath(path=resolved, display=display_path(resolved, roots))


def resolve_write_path(
    value: Any,
    *,
    roots: tuple[Path, ...],
    create_if_missing: bool,
    create_parent_dirs: bool = False,
    reject_absolute: bool = False,
) -> ResolvedPath:
    text = required_text(value, "path")
    candidate = Path(text).expanduser()
    if reject_absolute and candidate.is_absolute():
        raise FileToolError("patch paths must not be machine-specific absolute paths")
    if not candidate.is_absolute():
        candidate = roots[0] / candidate
    if candidate.is_symlink():
        raise FileToolError("symlink files are not patched")
    reject_symlink_ancestors(candidate)

    if candidate.exists():
        resolved = candidate.resolve(strict=True)
        if not is_inside_allowed(resolved, roots):
            raise _write_boundary_error()
        return ResolvedPath(path=resolved, display=display_path(resolved, roots))

    if not create_if_missing:
        raise FileToolError("create_if_missing must be true to create a new file")
    parent = candidate.parent
    resolved = candidate.resolve(strict=False)
    if not parent.exists():
        if not create_parent_dirs:
            raise FileToolError("parent directory must exist")
        parent_resolved = parent.resolve(strict=False)
        if not is_inside_allowed(parent_resolved, roots) or not is_inside_allowed(
            resolved,
            roots,
        ):
            raise _write_boundary_error()
        parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink():
        raise FileToolError("symlink directories are not patched")
    parent_resolved = parent.resolve(strict=True)
    if not parent_resolved.is_dir():
        raise FileToolError("parent path must be a directory")
    if not is_inside_allowed(parent_resolved, roots) or not is_inside_allowed(resolved, roots):
        raise _write_boundary_error()
    return ResolvedPath(path=resolved, display=display_path(resolved, roots))


def _write_boundary_error() -> FileToolError:
    return FileToolError(
        WRITE_BOUNDARY_MESSAGE,
        code="file_path_outside_writable_workspace",
        details=WRITE_BOUNDARY_DETAILS,
    )


def _root_scope(label: str) -> str:
    return "writable" if "write" in label else "readable"


def reject_symlink_ancestors(candidate: Path) -> None:
    ancestors = list(candidate.parents)
    for ancestor in reversed(ancestors):
        if ancestor.is_symlink():
            raise FileToolError("symlink ancestors are not patched")
        if not ancestor.exists():
            return


def reject_device_path(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError as exc:
        raise FileToolError(f"path cannot be statted: {exc}") from exc
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
        raise FileToolError("device paths are not allowed")


def path_type(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "directory"
    if path.is_file():
        return "file"
    return "other"


def _missing_suggestions(candidate: Path, roots: tuple[Path, ...]) -> list[str]:
    suggestions: list[str] = []
    parent = candidate.parent
    if parent.exists() and parent.is_dir():
        target = candidate.name.lower()
        for child in sorted(parent.iterdir(), key=lambda item: item.name)[:200]:
            name = child.name.lower()
            if target in name or name in target or name[:3] == target[:3]:
                suggestions.append(display_path(child.resolve(strict=False), roots))
            if len(suggestions) >= 5:
                return suggestions
    suggestions.extend(
        suggestion
        for suggestion in _global_missing_suggestions(candidate, roots)
        if suggestion not in suggestions
    )
    return suggestions


def _global_missing_suggestions(candidate: Path, roots: tuple[Path, ...]) -> list[str]:
    target_name = candidate.name.lower()
    if not target_name:
        return []
    suggestions: list[str] = []
    visited_dirs = 0
    max_dirs = 1000
    for root in roots:
        stack = [root]
        while stack and len(suggestions) < 5 and visited_dirs < max_dirs:
            current = stack.pop()
            visited_dirs += 1
            try:
                children = sorted(current.iterdir(), key=lambda item: item.name, reverse=True)
            except OSError:
                continue
            for child in children:
                if child.is_dir():
                    if _excluded_suggestion_dir(child, root):
                        continue
                    stack.append(child)
                    continue
                if child.name.lower() == target_name:
                    suggestions.append(display_path(child.resolve(strict=False), roots))
                    if len(suggestions) >= 5:
                        break
    return suggestions


def _excluded_suggestion_dir(path: Path, root: Path) -> bool:
    if path.name in DEFAULT_EXCLUDED_DIRS:
        return True
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return len(relative.parts) >= 2 and relative.parts[:2] == ("docs", "develop_record")
